import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper
from ibapi.contract import Contract
from threading import Thread
import time
import random 
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.relativedelta import relativedelta
import asyncio
import streamlit as st
import streamlit.components.v1 as components
import concurrent.futures
import subprocess
import os
import json

# Intentar obtener el bucle de eventos; si no existe, crear uno nuevo
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Future, Forex, MarketOrder, LimitOrder, Contract

try:
    sym_pro = pd.read_csv("symbol_problems.csv")
except:
    sym_pro = pd.DataFrame([], columns=["symbol", "fecha_ven", "multiplier", "trading_class", "exchange"])
    sym_pro.to_csv("symbol_problems.csv", index=False)

def generate_random_id(length=3):
    """
    Generate a random ID with a specified minimum length.

    Args:
        length (int): Minimum number of digits for the ID (default is 3).
    
    Returns:
        str: Randomly generated ID as a string.
    """
    if length < 3:
        raise ValueError("The minimum length for the ID must be at least 3.")
    
    # Generate a random number with the specified number of digits
    min_value = 10**(length - 1)  # Minimum value with the given length
    max_value = 10**length - 1   # Maximum value with the given length
    return int(random.randint(min_value, max_value))

def format_number(num):
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M".rstrip('0').rstrip('.')
    elif num >= 1_000:
        return f"{num / 1_000:.1f}k".rstrip('0').rstrip('.')
    else:
        return str(num)

def idm_size_calculator(idm_size):
        idm_mapping = {
            1: 1,
            2: 1.20,
            3: 1.48,
            4: 1.56,
            5: 1.70,
            6: 1.90,
            7: 2.10
        }
        
        # Asignar 2.20 para idm_size entre 8 y 14
        if idm_size in range(8, 15):
            return 2.20
        elif idm_size in range(15, 25):
            return 2.20
        elif idm_size in range(25, 30):
            return 2.40
        elif idm_size >= 30:
            return 2.50
        
        
        # Obtener el valor de IDM del diccionario o devolver None si no se encuentra
        return idm_mapping.get(idm_size, None)
    

def get_forecast_scalars_from_ewma_pairs(ewmac_values):
    # Datos de ejemplo
    forecast_scalars = {
        'EWMAC Pair': ['EWMAC 2,8', 'EWMAC 4,16', 'EWMAC 8,32', 'EWMAC 16,64', 'EWMAC 32,128', 'EWMAC 64,256'],
        'Forecast Scalar': [10.6, 7.5, 5.3, 3.75, 2.65, 1.87]
    }

    # Crear DataFrame
    df = pd.DataFrame(forecast_scalars)

    # Lista para almacenar los Forecast Scalars seleccionados
    selected_forecast_scalars = []

    # Procesar cada valor en ewmac_values
    for ewmac_value in ewmac_values:
        ewmac_pair = f"EWMAC {ewmac_value},{ewmac_value * 4}"
        forecast_scalar = df.loc[df['EWMAC Pair'] == ewmac_pair, 'Forecast Scalar'].values

        # Añadir a la lista si se encuentra el valor
        if len(forecast_scalar) > 0:
            selected_forecast_scalars.append(forecast_scalar[0])

    return selected_forecast_scalars

# Definir la matriz de correlación basada en EWMAs tabla 57
def get_ewmac_correlations(ewmac_values):
    """
    Genera una cadena de correlaciones EWMAC seleccionadas basada en los valores proporcionados.

    Args:
        ewmac_values (list): Lista de valores EWMAC para seleccionar.

    Returns:
        str: Cadena de correlaciones separada por comas.
    """
    # Diccionario de correlaciones EWMAC
    ewmac_correlations = {
        'EW 2, 8': [1.0, 0.90, 0.60, 0.35, 0.20, 0.15],
        'EW 4, 16': [0.90, 1.0, 0.90, 0.60, 0.40, 0.20],
        'EW 8, 32': [0.60, 0.90, 1.0, 0.90, 0.65, 0.45],
        'EW 16, 64': [0.35, 0.60, 0.90, 1.0, 0.90, 0.70],
        'EW 32, 128': [0.20, 0.40, 0.65, 0.90, 1.0, 0.90],
        'EW 64, 256': [0.15, 0.20, 0.45, 0.70, 0.90, 1.0]
    }

    # Crear DataFrame de correlaciones
    df = pd.DataFrame(
        ewmac_correlations, 
        index=['EW 2', 'EW 4', 'EW 8', 'EW 16', 'EW 32', 'EW 64']
    )

    # Generar los nombres de las EWMAC seleccionadas
    selected_ewmas = [f"EW {value}" for value in ewmac_values]

    # Filtrar filas y columnas correspondientes
    start_index = df.index.get_loc(selected_ewmas[0])
    end_index = df.index.get_loc(selected_ewmas[-1]) + 1
    filtered_df = df.iloc[start_index:end_index, start_index:end_index]

    # Crear una máscara para eliminar correlaciones duplicadas y valores redundantes
    for i in range(filtered_df.shape[0]):
        for j in range(i + 1, filtered_df.shape[1]):
            filtered_df.iloc[j, i] = np.nan

    # Extraer las correlaciones válidas (excluyendo 1.0 y NaN)
    ewmas_correlations = filtered_df[filtered_df != 1.0].stack().dropna().values
    
    return tuple(ewmas_correlations)

# Define the table of weights based on Table 8
def get_ewmac_weights(num_ewma, correlations):
    """
    Calcula los pesos basados en el número de activos y sus correlaciones.

    Args:
        num_ewma (int): Número de activos.
        correlations (tuple): Tupla de correlaciones entre los activos.

    Returns:
        list: Lista de pesos proporcionales para los activos.
    """

    # Verificar el tipo de correlaciones
    if not isinstance(correlations, tuple):
        raise ValueError("Las correlaciones deben ser una tupla.")

    if not all(isinstance(c, (int, float)) for c in correlations):
        raise ValueError("Todas las correlaciones deben ser números flotantes o enteros.")

    # Definir la tabla de pesos basada en el ejemplo original
    weights_data = {
        'Assets': [
            '1', '2', 'Identical Correlations', 
            'Three Assets 0.0, 0.5, 0.0', 'Three Assets 0.0, 0.9, 0.0',
            'Three Assets 0.5, 0.0, 0.5', 'Three Assets 0.0, 0.5, 0.9',
            'Three Assets 0.9, 0.0, 0.9', 'Three Assets 0.5, 0.9, 0.5',
            'Three Assets 0.9, 0.5, 0.9'
        ],
        'Weights': [
            '100', '50 each', 'Equal weights', 
            '30, 40, 30', '27, 46, 27', '37, 26, 37',
            '45, 45, 10', '39, 22, 39', '29, 42, 29', '42, 16, 42'
        ],
        'Correlations': [
            None, None, 'Identical', 
            (0.0, 0.5, 0.0), (0.0, 0.9, 0.0), (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.9), (0.9, 0.0, 0.9), (0.5, 0.9, 0.5), (0.9, 0.5, 0.9)
        ]
    }

    # Crear un DataFrame para buscar coincidencias
    weights_df = pd.DataFrame(weights_data)

    # Caso 1: Un solo activo
    if num_ewma == 1:
        return [1.0]  # 100% peso en el único activo

    # Caso 2: Dos activos (siempre igual peso)
    if num_ewma == 2:
        return [0.5, 0.5]  # Igual peso

    # Caso 3: Identical Correlations
    if correlations == "Identical":
        return [1 / num_ewma] * num_ewma  # Igual peso para correlaciones idénticas

    # Caso 4: Tres activos
    if num_ewma == 3:
        # Buscar una coincidencia exacta en la tabla
        exact_match = weights_df[(weights_df['Assets'].str.startswith('Three Assets')) & 
                                 (weights_df['Correlations'] == correlations)]
        if not exact_match.empty:
            weights_str = exact_match['Weights'].values[0]
            return [float(w) / 100 for w in weights_str.split(',')]  # Convertir a proporciones

        # Si no hay coincidencia exacta, buscar la más cercana
        min_distance = float('inf')
        closest_weight = None
        for _, row in weights_df[weights_df['Assets'].str.startswith('Three Assets')].iterrows():
            row_correlations = row['Correlations']
            if row_correlations is not None:
                # Asegurarnos de que las correlaciones son comparables (tipo float)
                distance = np.sqrt(sum((np.array(row_correlations) - np.array(correlations)) ** 2))
                if distance < min_distance:
                    min_distance = distance
                    closest_weight = row['Weights']

        # Si se encuentra la más cercana, devolverla
        if closest_weight:
            return [float(w) / 100 for w in closest_weight.split(',')]

        # Si no hay match, igual distribución
        return [1.0 / num_ewma] * num_ewma

    # Caso 5: Cuatro o más activos (repartición igual)
    if num_ewma >= 4:
        return [1.0 / num_ewma] * num_ewma  # Igual peso

    # Caso inválido
    raise ValueError("Entrada inválida. Revisa el número de activos o correlaciones.")

def generate_fdm_ewmac_and_carry(ewmac_values, ewma_weights, ewmacs_final_weight, carry_final_weight, carry_value):
    """
    Genera el cálculo final basado en las correlaciones y pesos seleccionados:
    1 / SQRT(MMULT(TRANSPOSE(weights), MMULT(correlations_matrix, weights))).

    Args:
        ewmac_values (list): Lista de valores EWMAC seleccionados (ejemplo: [8, 16, 32]).
        ewma_weights (list): Pesos específicos para cada EWMA en la misma posición que ewmac_values.
        ewmacs_final_weight (float): Peso final común para todas las EWMAs.
        carry_final_weight (float): Peso final asignado al Carry.
        carry_value (float): Valor del Carry a incluir en la tabla.

    Returns:
        float: Resultado del cálculo.
    """
    # Diccionario de correlaciones EWMAC
    ewmac_correlations = {
        'EW 2, 8': [1.0, 0.90, 0.60, 0.35, 0.20, 0.15, 0.25], 
        'EW 4, 16': [0.90, 1.0, 0.90, 0.60, 0.40, 0.20, 0.25], 
        'EW 8, 32': [0.60, 0.90, 1.0, 0.90, 0.65, 0.45, 0.25], 
        'EW 16, 64': [0.35, 0.60, 0.90, 1.0, 0.90, 0.70, 0.25], 
        'EW 32, 128': [0.20, 0.40, 0.65, 0.90, 1.0, 0.90, 0.25], 
        'EW 64, 256': [0.15, 0.20, 0.45, 0.70, 0.90, 1.0, 0.25], 
        'Carry': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, carry_value]
    }

    # Crear DataFrame
    df = pd.DataFrame(
        ewmac_correlations, 
        index=['EW 2, 8', 'EW 4, 16', 'EW 8, 32', 'EW 16, 64', 'EW 32, 128', 'EW 64, 256', 'Carry']
    )

    # Generar los nombres de los pares EWMAC seleccionados
    selected_ewmas = [f"EW {value}, {value * 4}" for value in ewmac_values] + ['Carry']

    # Filtrar filas y columnas correspondientes
    filtered_df = df.loc[selected_ewmas, selected_ewmas]

    # Calcular los pesos
    weights = np.array([
        ewma_weights[i] * ewmacs_final_weight for i in range(len(ewmac_values))
    ] + [carry_final_weight])

    # Extraer la matriz de correlaciones (sin la columna de pesos)
    correlations_matrix = filtered_df.values

    # Calcular 1 / SQRT(MMULT(TRANSPOSE(weights), MMULT(correlations_matrix, weights)))
    result = 1 / np.sqrt(np.dot(weights.T, np.dot(correlations_matrix, weights)))

    return result

def get_fdm_value_from_ema_pair( emas_pair: list):
        emas_pair = str(emas_pair)
        # Diccionario con los rangos de pares EMA y sus correspondientes FDM
        data_emas_pair = pd.DataFrame(
            {
                "options" : [
                    "EWMAC2, 4, 8, 16, 32 and 64",
                    "EWMAC4, 8, 16, 32 and 64",
                    "EWMAC8, 16, 32 and 64",
                    "EWMAC16, 32 and 64",
                    "EWMAC32 and 64",
                    "EWMAC64"
                ],
                "weights" : [
                    0.167,
                    0.2,
                    0.25,
                    0.333,
                    0.50,
                    1.0
                ],
                "fdm":[
                    1.26,
                    1.19,
                    1.13,
                    1.08,
                    1.03,
                    1.0
                ],
                "emas_pair":
                    [
                        str([(N, 4*N) for N in [2,4,8,16,32,64]]),
                        str([(N, 4*N) for N in [4,8,16,32,64]]),
                        str([(N, 4*N) for N in [8,16,32,64]]),
                        str([(N, 4*N) for N in [16,32,64]]),
                        str([(N, 4*N) for N in [32,64]]),
                        str([(N, 4*N) for N in [64]])
                    ]
            }
        )
        df_return = data_emas_pair[data_emas_pair["emas_pair"] == emas_pair]
        
        if len(df_return):
            return {
                "weight":df_return["weights"].iloc[0],
                "fdm": df_return["fdm"].iloc[0]
            }
        else:
            return "EMA pair not found in predefined categories"


# Clase para conexión a IB
class IBApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Lista para almacenar los datos históricos
 
    def error(self, reqId, errorCode, errorString):
        print(f"Error {reqId}: {errorCode} - {errorString}")
    
    @iswrapper
    def historicalData(self, reqId, bar):
        """Callback para manejar los datos históricos"""
        print(bar)
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
 
    @iswrapper
    def historicalDataEnd(self, reqId, start, end):
        """Callback que se llama cuando los datos históricos han terminado de descargarse"""
        print(f"Historical data download complete from {start} to {end}")
        self.disconnect()  # Desconectar cuando se completan los datos
 
    def get_historical_data_fut(self, 
                            symbol,
                            sec_type,
                            exchange,
                            currency,
                            last_trade_date,
                            duration,
                            bar_size,
                            multiplier,
                            trading_class,
                            end_time=""):
        """Solicitar datos históricos desde la API de IB"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        contract.multiplier = multiplier
        contract.tradingClass = trading_class
        contract.lastTradeDateOrContractMonth = last_trade_date  # Fecha de vencimiento del futuro
 
        # Solicitar datos históricos
        self.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=end_time,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",  # TRADES para futuros
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
    def get_historical_data_contfut(self, 
                            id,
                            sec_type,
                            exchange,
                            currency,
                            #last_trade_date,
                            duration,
                            bar_size,
                            multiplier,
                            trading_class,
                            end_time=""):
        """Solicitar datos históricos desde la API de IB"""
        contract = Contract()
        contract.symbol = trading_class
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        contract.multiplier = multiplier
        #contract.tradingClass = trading_class
        #contract.lastTradeDateOrContractMonth = last_trade_date  # Fecha de vencimiento del futuro
 
        # Solicitar datos históricos
        self.reqHistoricalData(
            reqId=id,
            contract=contract,
            endDateTime=end_time,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",  # TRADES para futuros
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

def run_loop(app):
    app.run()
def download_continous_future_data(
                          multiplier,
                          trading_class,
                          exchange="GLOBEX",
                          currency="USD",
                          duration="1 D",
                          bar_size="1 min",
                          end_time=None,
                          ip= '127.0.0.1',
                          port=7496):
    """Función principal para descargar datos históricos de futuros y devolver un DataFrame"""
    app = IBApp()
    
    random_id = generate_random_id()
    
    # Conectarse a TWS o IB Gateway
    app.connect(ip, port, clientId=random_id)
 
    # Iniciar la aplicación en un hilo separado
    api_thread = Thread(target=run_loop, args=(app,))
    api_thread.start()
 
    # Definir la fecha de fin explícita si no está definida
    if end_time is None:
        from datetime import datetime
        end_time = datetime.now().strftime("%Y%m%d %H:%M:%S UTC")
 
    # Solicitar datos históricos para el futuro
    app.get_historical_data_contfut(random_id,
                            "CONTFUT",
                            exchange,
                            currency,
                            #last_trade_date,
                            duration,
                            bar_size,
                            multiplier,
                            trading_class)
 
    # Esperar a que se descarguen los datos
    time.sleep(10)  # Aumenta el tiempo de espera si es necesario
 
    # Convertir los datos en un DataFrame de pandas
    df = pd.DataFrame(app.data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
 
    # Desconectar la aplicación y detener el hilo
    app.disconnect()
    api_thread.join()
 
    return df

def download_futures_data(symbol,
                          last_trade_date,
                          multiplier,
                          trading_class,
                          exchange="GLOBEX",
                          currency="USD",
                          duration="1 D",
                          bar_size="1 min",
                          end_time=None,
                          sectype="FUT",
                          ip= '127.0.0.1',
                          port=7496):
    """Función principal para descargar datos históricos de futuros y devolver un DataFrame"""
    app = IBApp()
    
    random_id = generate_random_id()
    
    # Conectarse a TWS o IB Gateway
    app.connect(ip, port, clientId=random_id)
 
    # Iniciar la aplicación en un hilo separado
    api_thread = Thread(target=run_loop, args=(app,))
    api_thread.start()
 
    # Definir la fecha de fin explícita si no está definida
    if end_time is None:
        from datetime import datetime
        end_time = datetime.now().strftime("%Y%m%d %H:%M:%S UTC")
 
    # Solicitar datos históricos para el futuro
    app.get_historical_data_fut(symbol,
                            sectype,
                            exchange,
                            currency,
                            last_trade_date,
                            duration,
                            bar_size,
                            multiplier,
                            trading_class,
                            end_time=end_time)
 
    # Esperar a que se descarguen los datos
    time.sleep(7)  # Aumenta el tiempo de espera si es necesario
 
    # Convertir los datos en un DataFrame de pandas
    df = pd.DataFrame(app.data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
 
    # Desconectar la aplicación y detener el hilo
    app.disconnect()
    api_thread.join()
 
    return df

#Funcion para buscar tablas de correlacion para pesos de activos
def get_correlation_tables():
    """
    Retorna las tablas de correlaciones como DataFrames.

    Returns:
        dict: Diccionario con las tablas de correlaciones etiquetadas.
    """
    # Tabla 50: Correlations across super-asset classes
    table_50 = pd.DataFrame(
        [
            [1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 1, 0.1, 0.1, 0.6],
            [0.1, 0.1, 1, 0.25, 0.2],
            [0.1, 0.1, 0.25, 1, 0.1],
            [0.1, 0.6, 0.2, 0.1, 1]
        ],
        index=["Rates", "Equities", "FX", "Commodities", "Volatility"],
        columns=["Rates", "Equities", "FX", "Commodities", "Volatility"]
    )

    # Tabla 51: Correlation across asset classes
    table_51 = pd.DataFrame(
        [
            [1, 0.5, np.nan, np.nan, np.nan],
            [0.5, 1, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1, 0.2, 0.25],
            [np.nan, np.nan, 0.2, 1, 0.35],
            [np.nan, np.nan, 0.25, 0.35, 1]
        ],
        index=["Bonds (R)", "STIR (R)", "Agricultural (C)", "Metal (C)", "Energy (C)"],
        columns=["Bonds", "STIR", "Agricultural", "Metal", "Energy"]
    )

    # Tabla 52: Correlations within commodity asset classes
    table_52 = pd.DataFrame(
        [
            [1, 0.4, 0.25, np.nan, np.nan, np.nan, np.nan],
            [0.4, 1, 0.15, np.nan, np.nan, np.nan, np.nan],
            [0.25, 0.15, 1, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 1, 0.25, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 0.25, 1, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 1, 0.5],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.5, 1]
        ],
        index=["Grains (A)", "Softs (A)", "Livestock (A)", "Oil (E)", "Gas (E)", "Precious metals (M)", "Base metals (M)"],
        columns=["Grains", "Softs", "Livestock", "Oil", "Gas", "Precious metals", "Base metals"]
    )

    # Tabla 53: Correlations for regions within financial asset classes
    table_53 = pd.DataFrame(
        [0.35, 0.35, 0.5, 0.5, 0.15],
        index=[
            "Emerging and developed market bonds",
            "Emerging and developed market STIR",
            "Emerging and developed market equities",
            "Emerging and developed market volatility",
            "Emerging and developed FX rates"
        ],
        columns=["Correlation"]
    )

    # Tabla 54: Correlations within regions and sub-asset classes
    table_54 = pd.DataFrame(
        [0.75, 0.75, 0.75, 0.75, 0.7, 0.7, 0.8],
        index=[
            "For bonds in same region, different countries",
            "For equities in same region, different countries",
            "For FX rates in same region, different rates against USD",
            "For volatility in same region, different countries",
            "For commodities in same sub-asset class, different products",
            "For equities in same country, different industry",
            "For equities in same industry, different firms"
        ],
        columns=["Correlation"]
    )

    # Tabla 55: Correlations for bonds or bond futures of different durations
    table_55 = pd.DataFrame(
        [
            [1, 0.8, 0.65, 0.5, 0.5],
            [0.8, 1, 0.85, 0.8, 0.75],
            [0.65, 0.85, 1, 0.85, 0.8],
            [0.5, 0.8, 0.85, 1, 0.9],
            [0.5, 0.75, 0.8, 0.9, 1]
        ],
        index=["2 year", "5 year", "10 year", "20 year", "30 year"],
        columns=["2 year", "5 year", "10 year", "20 year", "30 year"]
    )

    # Retornar todas las tablas como un diccionario
    return {
        "table_50": table_50,
        "table_51": table_51,
        "table_52": table_52,
        "table_53": table_53,
        "table_54": table_54,
        "table_55": table_55
    }

#Funcion para caluclo IDM Systematic Trading Book
def idm_calculator(instruments, sectors, weights, table):
    """
    Agrupa instrumentos en categorías principales según sus sectores, añade una columna de pesos,
    multiplica las correlaciones por 0.7, y calcula el resultado final.

    Args:
        instruments (list): Lista de nombres de instrumentos.
        sectors (list): Lista de sectores asociados a cada instrumento.
        weights (list): Lista de pesos asignados a cada instrumento.
        table (pd.DataFrame): Tabla de correlaciones (table_50).

    Returns:
        float: Resultado del cálculo final.
    """
    print(instruments)
    print(sectors)
    # Validación de entradas
    if len(instruments) != len(sectors):
        raise ValueError("La longitud de instrumentos y sectores debe ser la misma.")
    if len(instruments) != len(weights):
        raise ValueError("La longitud de instrumentos y pesos debe ser la misma.")

    # Definir el mapeo de sectores a categorías principales
    sector_to_category = {
        "STIR": "Rates", "Bond": "Rates",
        "Sector": "Equities", "Housing": "Equities", "Equity": "Equities",
        "Metals": "Commodities", "OilGas": "Commodities", "Ags": "Commodities",
        "Other": "Commodities", "odityIndex": "Commodities", 
        "Vol": "Volatility",
        "FX": "FX"
    }

    # Mapear instrumentos a categorías principales
    category_map = {instrument: sector_to_category[sector] for instrument, sector in zip(instruments, sectors)}

    # Validar que las categorías mapeadas estén en la tabla
    unique_categories = list(set(category_map.values()))
    missing_categories = [cat for cat in unique_categories if cat not in table.index or cat not in table.columns]
    if missing_categories:
        raise ValueError(f"Las siguientes categorías no están en la tabla: {missing_categories}")

    # Crear la matriz de correlaciones entre instrumentos
    instrument_correlation = pd.DataFrame(index=instruments, columns=instruments, data=0.0)
    for i, inst1 in enumerate(instruments):
        for j, inst2 in enumerate(instruments):
            cat1 = category_map[inst1]
            cat2 = category_map[inst2]
            instrument_correlation.at[inst1, inst2] = table.at[cat1, cat2]

    # Multiplicar correlaciones por 0.7 excepto los valores iguales a 1
    instrument_correlation = instrument_correlation.applymap(lambda x: x * 0.7 if x != 1 else x)

    # Convertir la matriz de correlaciones y los pesos a NumPy para el cálculo final
    correlation_matrix = instrument_correlation.values
    weight_vector = np.array(weights).reshape(-1, 1)  # Convertir a columna

    # Calcular el resultado final
    # Fórmula: 1 / sqrt(w^T * C * w)
    intermediate = np.matmul(np.matmul(weight_vector.T, correlation_matrix), weight_vector)
    result = 1 / np.sqrt(intermediate[0, 0])

    return result

#Asignación pesos por grupo
def assign_group_weights(assets, correlations=None):
    """
    Asigna pesos a los activos dentro de un grupo en función de la Tabla 8 y las correlaciones.

    Args:
        assets (list): Lista de activos en el grupo.
        correlations (list): Lista de correlaciones entre pares de activos.

    Returns:
        dict: Pesos asignados a cada activo.
    """
    if len(assets) == 1:  # Caso de un solo activo
        return {assets[0]: 1.0}

    if len(assets) == 2:  # Caso de dos activos
        return {assets[0]: 0.5, assets[1]: 0.5}

    if len(assets) == 3 and correlations:  # Caso de tres activos
        AB, AC, BC = correlations[:3]
        # Reglas de la Tabla 8
        if AB == AC == BC:
            return {asset: 1 / 3 for asset in assets}
        if AB == 0.0 and AC == 0.5 and BC == 0.0:
            return {assets[0]: 0.3, assets[1]: 0.4, assets[2]: 0.3}
        if AB == 0.0 and AC == 0.9 and BC == 0.0:
            return {assets[0]: 0.27, assets[1]: 0.46, assets[2]: 0.27}
        if AB == 0.5 and AC == 0.0 and BC == 0.5:
            return {assets[0]: 0.37, assets[1]: 0.26, assets[2]: 0.37}
        if AB == 0.0 and AC == 0.5 and BC == 0.5:
            return {assets[0]: 0.45, assets[1]: 0.45, assets[2]: 0.1}
        if AB == 0.9 and AC == 0.9 and BC == 0.0:
            return {assets[0]: 0.39, assets[1]: 0.22, assets[2]: 0.39}
        if AB == 0.5 and AC == 0.9 and BC == 0.5:
            return {assets[0]: 0.29, assets[1]: 0.42, assets[2]: 0.29}
        if AB == 0.9 and AC == 0.5 and BC == 0.9:
            return {assets[0]: 0.42, assets[1]: 0.16, assets[2]: 0.42}

    # Dividir equitativamente si no hay reglas específicas
    return {asset: 1 / len(assets) for asset in assets}

#Asignacion pesos por sector
def calculate_group_weights(contracts, sectors, regions, tables):
    """
    Calcula los pesos de los activos dentro de sectores utilizando las reglas de la Tabla 8 y las correlaciones jerárquicas.

    Args:
        contracts (list): Lista de contratos/activos seleccionados.
        sectors (dict): Diccionario de sectores con activos.
        regions (dict): Diccionario que asocia activos con regiones (e.g., 'ASIA', 'EMEA', 'US').
        tables (dict): Tablas de correlaciones jerárquicas (e.g., Table 50-55).

    Returns:
        dict: Pesos calculados para los activos seleccionados.
    """
    def select_table(sector, region):
        """
        Selecciona la tabla adecuada para un sector y una región.

        Args:
            sector (str): Sector del activo.
            region (str): Región del activo.

        Returns:
            pd.DataFrame: La tabla de correlaciones correspondiente.
        """
        if sector in ["Rates", "Equities", "FX", "Commodities", "Volatility"]:
            return tables["table_50"]
        elif sector in ["Bonds (R)", "STIR (R)", "Agricultural (C)", "Metal (C)", "Energy (C)"]:
            return tables["table_51"]
        elif sector in ["Grains (A)", "Softs (A)", "Livestock (A)", "Oil (E)", "Gas (E)", "Precious metals (M)", "Base metals (M)"]:
            return tables["table_52"]
        elif region in ["ASIA", "EMEA", "US"]:
            return tables["table_53"]
        elif "same region" in sector or "sub-asset class" in sector:
            return tables["table_54"]
        elif sector in ["2 year", "5 year", "10 year", "20 year", "30 year"]:
            return tables["table_55"]
        return None

    sector_weights = {}
    
    for sector, assets in sectors.items():
        selected_assets = [asset for asset in assets if asset in contracts]
        if not selected_assets:
            continue

        # Obtener la región del primer activo del sector (asumiendo una región para el sector)
        region = regions.get(selected_assets[0], "Unknown")

        # Seleccionar la tabla adecuada
        sector_table = select_table(sector, region)
        if sector_table is None:
            print(f"No se encontró una tabla para el sector '{sector}' y la región '{region}'")
            continue

        # Calcular correlaciones para los activos seleccionados
        sector_correlations = []
        for i, asset1 in enumerate(selected_assets):
            for j, asset2 in enumerate(selected_assets):
                if i < j:  # Evitar duplicados
                    corr = sector_table.loc[asset1, asset2] if asset1 in sector_table.index and asset2 in sector_table.columns else 0.0
                    sector_correlations.append(corr)

        # Aplicar las reglas de la Tabla 8
        weights = assign_group_weights(selected_assets, sector_correlations)

        # Asignar los pesos calculados
        for asset, weight in weights.items():
            sector_weights[asset] = weight

    return sector_weights

#Asignacion pesos final
def calculate_weights_with_correlation_and_table8(contracts, sectors, regions, tables):
    """
    Calcula los pesos de los activos dentro de grupos utilizando las reglas de la Tabla 8 y las correlaciones jerárquicas.

    Args:
        contracts (list): Lista de contratos/activos seleccionados.
        sectors (dict): Diccionario de sectores con activos.
        regions (dict): Diccionario que asocia activos con regiones (e.g., 'ASIA', 'EMEA', 'US').
        tables (dict): Tablas de correlaciones jerárquicas (e.g., Table 50-55).

    Returns:
        dict: Pesos calculados para los activos seleccionados.
    """
    # Definir grupos principales de activos
    asset_groups = {
        "Rates": {"STIR", "Bond"},
        "Equities": {"Sector", "Housing", "Equity"},
        "Commodities": {"Metals", "OilGas", "Ags", "Other", "CommodityIndex"},
        "Volatility": {"Vol"},
        "FX": {"FX"}
    }

    # Paso 1: Dividir el 100% entre los asset groups en formato decimal
    active_groups = {group: group_sectors for group, group_sectors in asset_groups.items()
                     if any(sector in group_sectors for sector in sectors)}
    total_active_groups = len(active_groups)
    weight_per_group = 1 / total_active_groups  # En decimal

    # Asignar peso base a cada grupo de activos
    group_weights = {group: weight_per_group for group in active_groups.keys()}

    # Paso 2: Calcular pesos dentro de cada grupo utilizando la función descrita
    final_weights = {}
    for group, group_sectors in active_groups.items():
        # Filtrar sectores relevantes del grupo
        relevant_sectors = {sector: assets for sector, assets in sectors.items() if sector in group_sectors}
        if not relevant_sectors:
            continue

        # Calcular pesos dentro del grupo
        sector_weights = calculate_group_weights(contracts, relevant_sectors, regions, tables)

        # Ajustar pesos según el peso del grupo
        for asset, weight in sector_weights.items():
            final_weights[asset] = (weight_per_group * weight)  # Mantener en formato decimal

    # Paso 3: Ajustar pesos para asegurar que la suma sea 1
    total_weight = sum(final_weights.values())
    if total_weight != 1.0:
        final_weights = {asset: weight / total_weight for asset, weight in final_weights.items()}

    return final_weights

# Función para calcular sector_weights
def calculate_weights(contracts, sectors):
    sector_weights = {}
    total_sectors = sum(1 for sector in sectors if any(asset in contracts for asset in sectors[sector]))
    base_weight = 100 / total_sectors

    for sectors, assets in sectors.items():
        if any(asset in contracts for asset in assets):
            asset_weight = base_weight / len(assets)
            for asset in assets:
                if asset in contracts:
                    sector_weights[asset] = asset_weight / 100  # Convertir a formato decimal
    return sector_weights

# Function to calculate the EWMA standard deviation directly
def calculate_ewma_std(returns, lambda_param):
    """
    Calculate EWMA standard deviation directly using the recursive variance formula.
    
    Parameters:
    - returns: A numpy array of returns
    - lambda_param: The smoothing parameter
    
    Returns:
    - std_dev: A numpy array of EWMA standard deviation values
    """
    ewma_variance = np.zeros(len(returns))
    ewma_variance[0] = returns[0] ** 2  # Initialize with the square of the first return
    for t in range(1, len(returns)):
        ewma_variance[t] = lambda_param * (returns[t] ** 2) + ((1 - lambda_param) * ewma_variance[t - 1])
    return np.sqrt(ewma_variance)

# Function to calculate the slow EWMA
def calculate_ewma_slow(prices, lambda_slow):
    """
    Calculate the EWMA (Slow) for a given price series and smoothing factor.

    Parameters:
    - prices: A numpy array or pandas series of prices.
    - lambda_slow: The smoothing parameter for the slow EWMA.

    Returns:
    - A numpy array of the EWMA (Slow) values.
    """
    ewma_slow = np.zeros(len(prices))
    print(len(ewma_slow))
    ewma_slow[0] = prices[0]  # Initialize with the first price
    for t in range(1, len(ewma_slow)):
        ewma_slow[t] = lambda_slow * prices[t] + ((1 - lambda_slow) * ewma_slow[t - 1])
    return ewma_slow

# Function to calculate the fast EWMA
def calculate_ewma_fast(prices, lambda_fast):
    """
    Calculate the EWMA (Fast) for a given price series and smoothing factor.

    Parameters:
    - prices: A numpy array or pandas series of prices.
    - lambda_fast: The smoothing parameter for the fast EWMA.

    Returns:
    - A numpy array of the EWMA (Fast) values.
    """
    ewma_fast = np.zeros(len(prices))
    print(len(ewma_fast))
    ewma_fast[0] = prices[0]  # Initialize with the first price
    for t in range(1, len(ewma_fast)):
        ewma_fast[t] = lambda_fast * prices[t] + ((1 - lambda_fast) * ewma_fast[t - 1])
    return ewma_fast

def calculate_month_difference(date1, date2):
# Convertir las cadenas a objetos datetime
    def parse_date(date):
        # Determina el formato según la longitud de la cadena
        if len(date) == 6:  # Formato 'YYYYMM'
            return datetime.strptime(date, '%Y%m')
        elif len(date) == 8:  # Formato 'YYYYMMDD'
            return datetime.strptime(date, '%Y%m%d')
        else:
            raise ValueError(f"Invalid date format: {date}")

    # Convertir las cadenas a objetos datetime
    date1 = parse_date(date1)
    date2 = parse_date(date2)

    # Calcular la diferencia en años y meses
    years_diff = date2.year - date1.year
    months_diff = date2.month - date1.month

    # Calcular el total de meses
    total_months = abs(years_diff * 12 + months_diff)

    return total_months

def generate_futures_symbol(symbol, date):
    """
    Generate the futures symbol based on the instrument symbol and expiration date.
    Args:
    symbol (str): The base symbol of the futures instrument (e.g., "ES").
    date (str): The expiration date in the format "YYYYMMDD" (e.g., "20241220").
    Returns:
    str: The full futures symbol (e.g., "ESZ4").
    """
    # Mapping of months to their respective codes
    months = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J',
        5: 'K', 6: 'M', 7: 'N', 8: 'Q',
        9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }
    # Extract the year and month from the date
    year = int(date[:4])  # First 4 characters are the year
    month = int(date[4:6])  # Next 2 characters are the month
    # Get the month code and last digit of the year
    month_code = months[month]
    year_digit = str(year)[-1]  # Get the last digit of the year
    # Combine symbol, month code, and year digit
    return f"{symbol}{month_code}{year_digit}"


def fetch_data_with_retries(download_function, max_retries=10, retry_delay=2, **kwargs):
    """
    Tries to fetch data multiple times until successful or retries are exhausted.

    Args:
        download_function (callable): Function to download data.
        max_retries (int): Maximum number of retries (default is 5).
        retry_delay (int): Delay in seconds between retries (default is 2).
        **kwargs: Arguments to pass to the download function.

    Returns:
        data: The downloaded data or None if unsuccessful.
    """
    attempts = 0
    data = None

    while attempts < max_retries:
        try:
            data = download_function(**kwargs)
            if len(data) > 0:  # Successful download
                break
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")

        attempts += 1
        print(attempts)
        time.sleep(retry_delay)  # Wait before retrying

    if data is None:
        print(f"Failed to fetch data after {max_retries} attempts with arguments {kwargs}.")
    else:
        print(f"Successfully fetched data on attempt {attempts + 1}.")

    return data


# Procesar cálculos por activo
def process_asset(asset_name, 
                  contract_info,
                  sector_weights,
                  look_back_period,
                  forecast_scalars,
                  ewma_weights,
                  fdm_value,
                  multipliers,
                  daily_cash_vol_target,
                  idm_value,
                  ewmacs,
                  ewmacs_final_weight,
                  carry_final_weight,
                  ip= '127.0.0.1',
                  port=7496):
    # Descargar datos para EWMA
    instrument = contract_info[asset_name]['instrument']
    multiplier=contract_info[asset_name]['carry_symbols']['multiplier']
    trading_class=asset_name
    exchange=contract_info[asset_name]['exchange']
    currency=contract_info[asset_name]['currency']
    
    instruments_currency = {
        'EUR':'EUR',
        'AUD':'AUD',
        'CAD':'CAD',
        'JPY':'JPY',
        'HKD':'HKD',
        'CHF':'CHF',
        'CNH':'CNH',
        'GBP':'GBP',
        'INR':'INR',
        'MXP':'MXP',
        'KRW':'KRWUSD',
        'SGD':'SGD',
        'SEK':'SEK'
    }
    if currency == 'USD':
        fx_rate = 1
    elif currency== 'HKD':
        df_fx = None
        df_fx= get_ib_data_cfd('HKD', 
                                'IDEALPRO', 
                                101,
                                ip,
                                port)
        price= df_fx['Close'].iloc[-1]
        jpy= load_data_from_db(db_name = "data/zenit_future_instrument.db",
                                    table_name= 'JPY',
                                    last_row = True)["Close"].iloc[0]
        fx_rate=  price*jpy
    else:
        table_name = instruments_currency.get(currency)
        fx_rate = load_data_from_db(db_name = "data/zenit_future_instrument.db",
                                    table_name= table_name,
                                    last_row = True)["Close"].iloc[0]
    
    
    print(f"{multiplier} - {trading_class} - {exchange}- {currency}")
    
    
    max_retries = 10
    retry_delay = 3
    attempts = 0
    ewma_data = None

    while attempts < max_retries:
        try:
            ewma_data = load_data_from_db(db_name="data/zenit_future_instrument.db", 
                                            table_name=instrument)
            ewma_data = ewma_data.dropna(subset=["Close"], axis=0).reset_index(drop=True)
            if len(ewma_data) > 0:  # Successful download
                attempts = 0
                break
        except Exception as e:
            print(f"Attempt {attempts + 1} failed for {asset_name}: {e}")
        
        attempts += 1
        print(attempts)
        time.sleep(retry_delay)  # Wait before retrying

    if ewma_data is None:
        print(f"Failed to fetch EWMA data for {asset_name} after {max_retries} attempts.")
    else:
        print(f"Successfully fetched EWMA data for {asset_name} on attempt {attempts + 1}.")
    
    if ewma_data.empty:
        print(f"No se encontraron datos para {asset_name} (EWMA).")
        new_row = {
            "symbol": asset_name,
            "fecha_ven": "ewma_data",
            "multiplier": contract_info[asset_name]['carry_symbols']['multiplier'],
            "trading_class": contract_info[asset_name]['trading_class'],
            "exchange": contract_info[asset_name]['exchange']
        }

        # Agregar la nueva fila al DataFrame
        sym_pro.loc[len(sym_pro)] = new_row
        sym_pro.to_csv("symbol_problems.csv", index=False)
        return pd.DataFrame()

    # Calcular std.EWMA 36d
    ewma_data['Returns'] = ewma_data['Close'].diff().fillna(0)
    lambda_param = 2 / (look_back_period + 1)
    ewma_std_col = f'EWMAC_Std_{look_back_period}D'
    ewma_data[ewma_std_col] = calculate_ewma_std(ewma_data['Returns'], lambda_param)
    current_price=ewma_data['Close'].iloc[-1]
    ewmac_m=[(value, value*4) for value in ewmacs ]
   # Forecast y pesos
    ewma_forecast_data = []
    weighted_forecast = 0
    for i, (short_span, long_span) in enumerate(ewmac_m):
        ewma_data[f'EWMAC_{short_span}'] = calculate_ewma_fast(ewma_data['Close'].values, lambda_fast=(2 / (short_span + 1)))
        ewma_data[f'EWMAC_{long_span}'] = calculate_ewma_slow(ewma_data['Close'].values, lambda_slow=(2 / (long_span + 1)))
        ewma_data['EWMAC_Diff'] = ewma_data[f'EWMAC_{short_span}'] - ewma_data[f'EWMAC_{long_span}']
        ewma_data['Vol adjusted'] = ewma_data['EWMAC_Diff'] / (ewma_data[ewma_std_col].iloc[-1])
                
        forecast_ewma = ewma_data['Vol adjusted'] * forecast_scalars[i]
        #print(f"Data {instrument} forecast {forecast_ewma.tail(60)}")
        forecast_ewma = np.clip(forecast_ewma, -20, 20)
        
        final_forecast_weight = ewma_weights[i] * ewmacs_final_weight
        weighted_forecast += final_forecast_weight * forecast_ewma.iloc[-1]
        ewma_forecast_data.append({
            'Trading Rules & Variations': f'EWMAC {short_span}/{long_span}',
            'Forecast': round(forecast_ewma.iloc[-1], 14),
            'Final Forecast Weights': f"{float(final_forecast_weight * 100)}%"
        })
    
    # Carry
    # Descargar nearer_data con reintentos
    nearer_data = get_ib_data_instrument(asset_name, 
                                         exchange, 
                                         id=datetime.now().microsecond+1,
                                         date_ven=contract_info[asset_name]['carry_symbols']['last_trade_day_now'],
                                         multiplier=multiplier,
                                         currency=currency,
                                         ip= ip,
                                        port=port)
    
    
    if nearer_data is not None:
        print(f"***** Data carry 1 {asset_name}")

    # Descargar current_data con reintentos
    current_data = get_ib_data_instrument(asset_name, 
                                         exchange, 
                                         id=datetime.now().microsecond+1,
                                         date_ven=contract_info[asset_name]['carry_symbols']['last_trade_day_after'],
                                         multiplier=multiplier,
                                         currency=currency,
                                         ip= ip,
                                         port=port)
    if current_data is not None:
        print(f"***** Data carry 2 {asset_name}")
            
    if nearer_data is None:
        print(f"No se encontraron datos para Carry actual de {asset_name}.")
        new_row = {
            "symbol": asset_name,
            "fecha_ven": contract_info[asset_name]['carry_symbols']['last_trade_day_now'],
            "multiplier": contract_info[asset_name]['carry_symbols']['multiplier'],
            "trading_class": contract_info[asset_name]['trading_class'],
            "exchange": contract_info[asset_name]['exchange']
        }

        # Agregar la nueva fila al DataFrame
        sym_pro.loc[len(sym_pro)] = new_row
        sym_pro.to_csv("symbol_problems.csv", index=False)
        nearer_data = ewma_data
        
    if current_data is None:
        print(f"No se encontraron datos para Carry proximo contrato de {asset_name}.")
        new_row = {
            "symbol": asset_name,
            "fecha_ven": contract_info[asset_name]['carry_symbols']['last_trade_day_after'],
            "multiplier": contract_info[asset_name]['carry_symbols']['multiplier'],
            "trading_class": contract_info[asset_name]['trading_class'],
            "exchange": contract_info[asset_name]['exchange']
        }

        # Agregar la nueva fila al DataFrame
        sym_pro.loc[len(sym_pro)] = new_row
        sym_pro.to_csv("symbol_problems.csv", index=False)
        current_data = ewma_data
    
    nearer_price_carry = nearer_data['Close'].iloc[-1]
    current_price_carry = current_data['Close'].iloc[-1]
    std_carry = f'EWMAC_Std_{look_back_period}D'
    current_data['Returns'] = current_data['Close'].diff().fillna(0)
    current_data[std_carry]= calculate_ewma_std(current_data['Returns'], lambda_param)
    months_difference = calculate_month_difference(contract_info[asset_name]['carry_symbols']['last_trade_day_now']
                                                   , contract_info[asset_name]['carry_symbols']['last_trade_day_after'])
    distance_between_contracts = months_difference / 12
    price_differential = nearer_price_carry - current_price_carry
    net_expected_return_price_units = price_differential / distance_between_contracts
    annualised_std_dev = current_data[std_carry].iloc[-1] * np.sqrt(256)
    raw_carry = net_expected_return_price_units / annualised_std_dev
    carry_forecast = np.clip(30 * raw_carry, -20, 20)
    carry_weight = carry_final_weight

    # Combinar forecasts
    combined_forecast = np.clip((weighted_forecast + carry_weight * carry_forecast) * fdm_value, -20, 20)

# Cálculos adicionales
    multiplier = multipliers.get(asset_name, 1)
    block_value = round(0.01 * current_price * multiplier, 16)
    price_volatility = round((ewma_data[ewma_std_col].iloc[-1]/current_price)*100, 16)
    instrument_currency_vol = round(block_value * price_volatility * fx_rate, 16)
    volatility_scalar = round(daily_cash_vol_target / instrument_currency_vol, 16)
    subsystem_position = round((combined_forecast * volatility_scalar) / 10, 16)

 # Agregar carry a las filas del forecast
    ewma_forecast_data.append({
        'Trading Rules & Variations': 'Preferred Carry Rule',
        'Forecast': round(carry_forecast, 2),
        'Final Forecast Weights': f"{float(carry_weight * 100)}%"
    })
    
     # Crear dataframe final en el orden correcto basado en las imágenes
     #ewma_forecast_data_df = pd.DataFrame(ewma_forecast_data)

    # Crear el resultado para un activo en formato vertical
    result = {
        'Metric': [
            'Asset Class', 'Currency', 'Daily Cash Vol Target', 'Price','Multiplier', 'FX Rate', 
            'EWMAC 2/8', 'EWMAC 4/16', 'EWMAC 8/32', 'EWMAC 16/64', 'EWMAC 32/128', 
            'EWMAC 64/256', 'Preferred Carry Rule', 'Combine Forecast', 'STD (%) del precio', 
            'Price Volatility', 'Subsystem Position', 'Weights', 'IDM', 'Portfolio Instrument Position'
        ],
        asset_name: [
            contract_info[asset_name]['asset_class'],
            contract_info[asset_name]['currency'],
            round(daily_cash_vol_target, 2),
            round(ewma_data['Close'].iloc[-1],16),
            multiplier,
            round(fx_rate,13),
            ewma_forecast_data[0]['Forecast'],
            ewma_forecast_data[1]['Forecast'],
            ewma_forecast_data[2]['Forecast'],
            ewma_forecast_data[3]['Forecast'],
            ewma_forecast_data[4]['Forecast'],
            ewma_forecast_data[5]['Forecast'],
            round(carry_forecast,16),
            round(combined_forecast,16),
            round(ewma_data[ewma_std_col].iloc[-1] / current_price, 16),
            price_volatility,
            subsystem_position,
            round(sector_weights.get(asset_name, 0),16),  # Weight en decimal
            round(idm_value,13),
            round(subsystem_position * sector_weights.get(asset_name, 0) * idm_value, 0)
        ]
    }

    # Convertir a DataFrame y establecer Metric como índice
    result_df = pd.DataFrame(result).set_index('Metric')

    return result_df


def create_event_loop():
    """Crea un bucle de eventos en el hilo actual."""
    asyncio.set_event_loop(asyncio.new_event_loop())
    return asyncio.get_event_loop()

def process_multiple_assets(contracts,
                            sectors,
                            look_back_period,
                            forecast_scalars,
                            ewma_weights,
                            fdm,
                            multipliers,
                            daily_cash_vol_target,
                            portfolio_name,
                            ewmacs,
                            ewmacs_final_weight,
                            carry_final_weight,
                            ip="127.0.0.1",
                            port=7496,
                            max_workers=10,
                            account="DU7186453"):
    today_str_date = datetime.now().date().strftime("%Y%m%d")
    results_df = pd.DataFrame()
    # Calcular pesos por sector y región
    region = {x: contracts[x]['region'] for x in contracts}
    correlation_tables = get_correlation_tables()
    table_50 = correlation_tables["table_50"]
    sector_weights = calculate_weights_with_correlation_and_table8(contracts, sectors, region, correlation_tables)

    instruments_list = [contracts[x]['instrument'] for x in contracts]
    weights = [sector_weights[x] for x in sector_weights]
    sectors_for_idm = [contracts[x]['asset_class'] for x in contracts]
    idm_value = idm_calculator(instruments_list, sectors_for_idm, weights, table_50)

    def run_process_asset(asset_name):
        """Envuelve la ejecución de process_asset dentro de un bucle de eventos."""
        loop = create_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = process_asset(
                asset_name,
                contracts,
                sector_weights,
                look_back_period,
                forecast_scalars,
                ewma_weights,
                fdm,
                multipliers,
                daily_cash_vol_target,
                idm_value,
                ewmacs,
                ewmacs_final_weight,
                carry_final_weight,
                ip=ip,
                port=port
            )
        finally:
            loop.close()
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_asset = {
            executor.submit(run_process_asset, asset_name): asset_name
            for asset_name in contracts.keys()
        }

        for future in as_completed(future_to_asset):
            asset_name = future_to_asset[future]
            try:
                result = future.result()
                if not result.empty:
                    results_df = pd.concat([results_df, result], axis=1)
            except Exception as e:
                print(f'Error procesando el instrumento {asset_name}: {e}')
    csv_data = results_df.copy()  
    csv_data_t = csv_data.T
    
    # Agregar el índice como una columna en el DataFrame
    csv_data_t.reset_index(inplace=True)
    csv_data_t.rename(columns={'index':'Instrument'}, inplace=True)
    actual_pos = get_ib_positions_by_account(account_id=account,
                                                ip=ip,
                                                port=port)
    if len(actual_pos) > 0:
        df_actual_pos = pd.DataFrame(actual_pos)
        df_actual_pos = df_actual_pos[df_actual_pos['Instrument'].isin(contracts.keys())].reset_index(drop=True)
    else:
        for symbol in contracts:
            symbol_data = {
                    "Instrument": contracts[symbol]["instrument"],
                    "SecType": "FUT",
                    "Currency": contracts[symbol]["currency"],
                    "Position": 0,
                    "AverageCost": 0,
                    "ExpirationDate": contracts[symbol]["carry_symbols"]["last_trade_day_now"],
                }
            actual_pos.append(symbol_data)
        df_actual_pos = pd.DataFrame(actual_pos)
    
    # Agrupar por instrumento y sumar las posiciones
    grouped_positions = df_actual_pos.groupby("Instrument")["Position"].sum().reset_index()

    # Renombrar columnas para mayor claridad
    grouped_positions.columns = ["Instrument", "Position"]
    # Merge entre los DataFrames
    merged_df = pd.merge(csv_data_t[['Instrument', 'Portfolio Instrument Position']], 
                            grouped_positions, 
                            on='Instrument', 
                            how='outer')

    # Reemplazar NaN por 0 en las posiciones
    merged_df.fillna({'Portfolio Instrument Position': 0, 'Position': 0}, inplace=True)
    merged_df['Portfolio Instrument Position'] = merged_df['Portfolio Instrument Position'].astype(float).astype(int)
    merged_df["is_widget"] = True
    merged_df.to_csv(f'data/{account}_{portfolio_name}_compare_{today_str_date}.csv')
    results_df.to_csv(f'data/{account}_{portfolio_name}_{today_str_date}.csv')



def process_asset_threaded(asset_name, contracts, sector_weights, look_back_period, forecast_scalars, ewma_weights, fdm, multipliers, daily_cash_vol_target, idm_value):
    """
    Wrapper function for threading to process an asset.
    """
    return process_asset(asset_name, contracts, sector_weights, look_back_period, forecast_scalars, ewma_weights, fdm, multipliers, daily_cash_vol_target, idm_value)

def execute_in_threads(contracts, sectors, look_back_period, forecast_scalars, ewma_weights, fdm, multipliers, daily_cash_vol_target, idm_value):
    """
    Executes the processing of assets in multiple threads.
    """
    results_df = pd.DataFrame()
    sector_weights = calculate_weights(contracts, sectors)
    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Submit tasks to threads
        futures = {
            executor.submit(
                process_asset_threaded,
                asset_name, contracts, sector_weights, look_back_period,
                forecast_scalars, ewma_weights, fdm, multipliers,
                daily_cash_vol_target, idm_value
            ): asset_name for asset_name in contracts.keys()
        }

        # Collect results as they complete
        for future in futures:
            result = future.result()
            print(result)
            if len(result):
                results_df = pd.concat([results_df, result], ignore_index=True)
                
    results_df.to_csv('resultados_IB.csv', index=False)
    return results_df


def load_data_from_db(db_name="data/zenit_future_instrument.db", 
                      table_name="closing_prices",
                      last_row=False):
    """
    Loads data from a specified table in an SQLite database and returns it as a Pandas DataFrame.
    
    Parameters:
    db_name (str): Name of the SQLite database file.
    table_name (str): Name of the table to load data from.
    
    Returns:
    pd.DataFrame: The data from the table as a DataFrame.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_name)
        if not last_row:
            # Query the table and load it into a DataFrame
            query = f"SELECT * FROM '{table_name}'"
            
        else:
            # Query to get the last row
            query = f"SELECT * FROM '{table_name}' ORDER BY ROWID DESC LIMIT 1"
        
        dataframe = pd.read_sql_query(query, conn)
        
        print(f"Data successfully loaded from table: {table_name}")
        return dataframe
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    finally:
        # Close the database connection
        conn.close()
        

def update_column_value(table_name, 
                        column_name, 
                        new_value, 
                        condition,
                        database_name="data/zenit_future_instrument.db"):
    """
    Actualiza una columna en una tabla de la base de datos SQLite.

    Args:
        database_name (str): Nombre de la base de datos SQLite.
        table_name (str): Nombre de la tabla.
        column_name (str): Nombre de la columna a actualizar.
        new_value (any): Nuevo valor para la columna.
        condition (str): Condición para filtrar las filas a actualizar (ejemplo: "id=1").
    """
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()

        # Formar la consulta SQL de actualización
        query = f"""
        UPDATE {table_name}
        SET {column_name} = ?
        WHERE {condition};
        """
        # Ejecutar la consulta con el nuevo valor
        cursor.execute(query, (new_value,))
        conn.commit()

        print(f"Columna '{column_name}' actualizada correctamente en la tabla '{table_name}'.")
    
    except sqlite3.Error as e:
        print(f"Error al actualizar la columna: {e}")
    
    finally:
        # Cerrar la conexión a la base de datos
        if conn:
            conn.close()
            
def convert_bars_to_dataframe(bars):
    # Crear el DataFrame a partir de los datos de las barras
    df = pd.DataFrame([{
        'date': bar.date,
        'Open': bar.open,
        'High': bar.high,
        'Low': bar.low,
        'Close': bar.close,
        'Volume': bar.volume,
        'Average': bar.average,
        'BarCount': bar.barCount
    } for bar in bars])

    # Asegurarse de que la columna de fecha sea del tipo datetime
    df['date'] = pd.to_datetime(df['date'])

    # Configurar la fecha como índice
    df.set_index('date', inplace=True)

    return df

def convert_bars_to_dataframe_db(bars):
    # Crear el DataFrame a partir de los datos de las barras
    df = pd.DataFrame([{
        'DATE': bar.date,
        'Close': bar.close,
        'Volume': bar.volume
    } for bar in bars])

    # Asegurarse de que la columna de fecha sea del tipo datetime
    df['DATE'] = pd.to_datetime(df['DATE'])

    return df

def calculate_date_difference(start_date_str, end_date_str):
    """
    Calculate the difference between two dates in terms of months, days, or years.

    Parameters:
    start_date_str (str): Start date in the format "YYYY-MM-DD".
    end_date_str (str): End date in the format "YYYY-MM-DD".

    Returns:
    str: The difference in "M" (months), "D" (days), or "Y" (years).
    """
    # Parse dates
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Calculate the time difference
    delta = relativedelta(end_date, start_date)

    # Determine the most significant unit to represent the difference
    if delta.years > 0:
        return f"{delta.years} Y"
    elif delta.months > 0:
        return f"{delta.months} M"
    else:
        days_diff = (end_date - start_date).days
        return f"{days_diff} D"


def insert_dataframe_to_table(table_name, dataframe, db_name="data/zenit_future_instrument.db"):
    """
    Insert data from a DataFrame into a specified table in an SQLite database.
    
    Parameters:
    db_name (str): Name of the SQLite database file.
    table_name (str): Name of the table to insert data into.
    dataframe (pd.DataFrame): DataFrame with columns "DATE" and "Close".
    """
    try:
        # Verificar que las columnas requeridas están presentes
        if not {"DATE", "Close"}.issubset(dataframe.columns):
            raise ValueError("The DataFrame must contain 'DATE' and 'Close' columns.")

        # Convertir DataFrame a lista de tuplas
        data_to_insert = list(dataframe.itertuples(index=False, name=None))

        # Conectar a la base de datos
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Insertar los datos
        cursor.executemany(
            f"INSERT INTO '{table_name}' (DATE, Close, Volume) VALUES (?, ?, ?)", data_to_insert
        )
        conn.commit()  # Confirmar los cambios
        print(f"Data successfully inserted into table: {table_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cerrar la conexión a la base de datos
        conn.close()
        
        
def get_ib_data_instrument(symbol, 
                           exchange, 
                           id,
                           date_ven,
                           multiplier,
                           currency,
                           ip= '127.0.0.1',
                           port=7496):
    # Conexión a TWS o IB Gateway
    ib = IB()
    ib.connect(ip, port, clientId=id)
    contract = Future(symbol=symbol, exchange=exchange)
    contract.lastTradeDateOrContractMonth = date_ven
    contract.multiplier=multiplier
    contract.currency=currency
    
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d 23:59:59')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_date,
        durationStr='1 Y',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    # Desconectar de la API
    ib.disconnect()
    
    if bars:
        df = convert_bars_to_dataframe_db(bars)

        # Mostrar un resumen de los datos descargados
        return df
    else:
        print("ocurrio un error al descargar este contrato")

       
def get_ib_data_cfd(symbol, 
                    exchange, 
                    client_id,
                    ip= '127.0.0.1',
                    port=7496):
    """
    Obtiene datos históricos de un instrumento CFD utilizando IB API.
 
    Parameters:
    - symbol (str): El símbolo del CFD.
    - exchange (str): El intercambio donde se opera el CFD.
    - client_id (int): ID único para la conexión del cliente.
 
    Returns:
    - pd.DataFrame: DataFrame con los datos históricos del CFD.
    """
   
    # Conexión a TWS o IB Gateway
    ib = IB()
    ib.connect(ip, port, clientId=client_id)
 
    # Configuración del contrato CFD
    contract = Forex(symbol=symbol, exchange=exchange)
 
    # Definir fecha final y descargar datos históricos
    end_date = datetime.today()
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_date,
        durationStr='1 D',
        barSizeSetting='1 day',
        whatToShow='MIDPOINT',  # O 'TRADES', según los datos disponibles
        useRTH=False,  # Incluye datos fuera de horas de mercado
        formatDate=1
    )
 
    # Desconectar de la API
    ib.disconnect()
 
    # Convertir los datos a un DataFrame
    if bars:
        df = convert_bars_to_dataframe_db(bars)
        
        return df
    else:
        print(f"No se obtuvieron datos para el CFD: {symbol} en {exchange}")


def create_portafolio_table(username, 
                            portfolio_name, 
                            symbols):
    """
    Crea una tabla `portafolio` en la base de datos `data/zenit_future_instrument.db`
    y agrega un nuevo portafolio si el nombre no está duplicado para el usuario.

    :param username: Nombre del usuario.
    :param portfolio_name: Nombre del portafolio.
    :param symbols: Lista de símbolos a insertar.
    """
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect("data/zenit_future_instrument.db")
        cursor = conn.cursor()

        # Crear la tabla `portafolio` si no existe
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portafolio (
            username TEXT NOT NULL,
            nombre_portafolio TEXT NOT NULL UNIQUE,
            simbolos TEXT NOT NULL
        )
        """)

        # Comprobar si el portafolio ya existe para este usuario
        cursor.execute("""
        SELECT COUNT(*) FROM portafolio
        WHERE username = ? AND nombre_portafolio = ?
        """, (username, portfolio_name))
        exists = cursor.fetchone()[0]

        if exists > 0:
            st.error(f"El portafolio '{portfolio_name}' ya existe para el usuario '{username}'.")
            conn.close()
            return

        # Convertir la lista de símbolos en una cadena separada por comas
        symbols_str = ",".join(symbols)

        # Insertar datos en la tabla
        cursor.execute("""
        INSERT INTO portafolio (username, nombre_portafolio, simbolos)
        VALUES (?, ?, ?)
        """, (username, portfolio_name, symbols_str))

        # Confirmar cambios y cerrar conexión
        conn.commit()
        conn.close()

        st.success("Datos insertados correctamente.")
    except sqlite3.Error as e:
        st.error(f"Error al interactuar con la base de datos: {e}")

def get_user_portfolios(username):
    """
    Obtiene la lista de portafolios creados por un usuario.

    :param username: Nombre del usuario.
    :return: Lista de portafolios.
    """
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect("data/zenit_future_instrument.db")
        cursor = conn.cursor()

        # Consultar los portafolios del usuario
        cursor.execute("""
        SELECT nombre_portafolio, simbolos FROM portafolio
        WHERE username = ?
        """, (username,))
        portfolios = [row for row in cursor.fetchall()]

        conn.close()
        return portfolios
    except sqlite3.Error as e:
        st.error(f"Error al interactuar con la base de datos: {e}")
        return []
    
# Función para eliminar un portafolio de la base de datos
def delete_portfolio(portfolio_name):
    try:
        conn = sqlite3.connect("data/zenit_future_instrument.db")
        cursor = conn.cursor()

        cursor.execute("DELETE FROM portafolio WHERE nombre_portafolio = ?", (portfolio_name,))
        conn.commit()
        conn.close()
        st.success(f"El portafolio '{portfolio_name}' ha sido eliminado correctamente.")
    except sqlite3.Error as e:
        st.error(f"Error al eliminar el portafolio: {e}")


@st.experimental_dialog("Portafolio Simbolos", width="large")
def show_portfolio_symbols(portfolio_name, symbols):
    symbols_list = "\n".join(symbols)
    components.html(
        f"""
        <div style="height: 80%; width: 100%; overflow: auto; background-color: white; padding: 10px; border: 1px solid #ccc;">
            <h2>Símbolos del portafolio: {portfolio_name}</h2>
            <h3>Cantidad de instrumentos: {len(symbols)}</h3>
            <textarea style="width: 100%; height: 100%; font-size: 16px;" >{symbols_list}</textarea>
        </div>
        """,
        height=500,
        width=500,
    )
    
def generate_dicts(req_symbol_list,
                   symbol_info):
    # Generar el diccionario de contratos automáticamente
    contracts = {}
    for symbol in req_symbol_list:
        try:
            # Filtrar la información para el símbolo actual
            symbol_data = symbol_info[symbol_info["broker_symbol"] == symbol].iloc[0]
            instrument = symbol_info[symbol_info["broker_symbol"] == symbol]["instrument"].iloc[0]
            # Limpiar y convertir valores
            point_size = float(str(symbol_data['pointsize']).replace(',', '').strip())
            multiplier = float(str(symbol_data['broker_ibmultiplier']).replace(',', '').strip())

            # Construir la entrada del diccionario
            contracts[symbol] = {
                'description': symbol_data['full_description'].strip(),
                'ewma_ticker': symbol,  # Contrato continuo
                'instrument': instrument,
                'region': symbol_data['region'],
                'asset_class': symbol_data['assetclass'],
                "carry_symbols": {
                    "last_trade_day_now": str(int(symbol_data['expiration_actual'])),
                    "last_trade_day_after": str(int(symbol_data['expiration_next'])),
                    "multiplier": int(multiplier) if multiplier > 1 else multiplier
                },
                'exchange': symbol_data['broker_exchange'].strip(),
                'currency': symbol_data['currency'].strip(),
                'trading_class': symbol_data['trading_class'].strip(),
                'point_size': int(point_size) if point_size > 1 else point_size
            }
        except Exception as e:
            print(f"Error processing sector for symbol {symbol}: {e}")

    #print(contracts)
    # Multiplicadores
    multipliers = {asset: contracts[asset]['point_size'] for asset in contracts.keys()}

    # Definición de sectores
    # Generar el diccionario de sectores automáticamente
    sectors = {}
    for symbol in req_symbol_list:
        try:
            # Filtrar la información para el símbolo actual
            symbol_data = symbol_info[symbol_info["broker_symbol"] == symbol].iloc[0]

            # Obtener la clase de activo (AssetClass)
            asset_class = symbol_data['assetclass'].strip()

            # Agregar el símbolo al sector correspondiente
            if asset_class not in sectors:
                sectors[asset_class] = []
            sectors[asset_class].append(symbol)
        except Exception as e:
            print(f"Error processing sector for symbol {symbol}: {e}")
    return contracts, multipliers,sectors


def run_update_script(ip, port, symbols):
    """Función para ejecutar el script en un hilo separado."""
    try:
        subprocess.run(['python', 'update_instrument_bd.py', '--ip', ip, '--port', str(port),'--symbols', str(symbols)], check=True)
        st.success("El proceso de actualización se ha completado correctamente.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error al ejecutar el script: {e}")
        
        

def run_update_position_script(ip, 
                               port, 
                               contracts, 
                               sectors, 
                               look_back_period, 
                               forecast_scalars, 
                               ewma_weights, fdm, 
                               multipliers, 
                               daily_cash_vol_target, 
                               portfolio_name, ewmacs, 
                               ewmacs_final_weight, carry_final_weight,
                               account):
    try:
        # Serializar los dicts a JSON
        contracts_json = json.dumps(contracts)
        sectors_json = json.dumps(sectors)
        multipliers_json = json.dumps(multipliers)
        forecast_scalars_json = json.dumps(forecast_scalars)
        ewma_weights_json = json.dumps(ewma_weights)
        ewmacs_json = json.dumps(ewmacs)

        subprocess.run([
            'python', 'IB.py',
            '--ip', ip,
            '--port', str(port),
            '--contracts', contracts_json,
            '--sectors', sectors_json,
            '--look_back_period', str(look_back_period),
            '--forecast_scalars', forecast_scalars_json,
            '--ewma_weights', ewma_weights_json,
            '--fdm', str(fdm),
            '--multipliers', multipliers_json,
            '--daily_cash_vol_target', str(daily_cash_vol_target),
            '--portfolio_name', portfolio_name,
            '--ewmacs', ewmacs_json,
            '--ewmacs_final_weight', str(ewmacs_final_weight),
            '--carry_final_weight', str(carry_final_weight),
            '--account', account
        ], check=True)
    except Exception as e:
        print(f"Error ejecutando el script: {e}")


def load_portfolio_csv(selected_portfolio, username, message=True):
    # Construir el path del archivo
    today_str_date = datetime.now().date().strftime("%Y%m%d")
    csv_path = f"data/{selected_portfolio}_{username}_{today_str_date}.csv"
    
    # Verificar si el archivo existe
    if os.path.exists(csv_path):
        try:
            # Cargar el archivo CSV
            csv_data = pd.read_csv(csv_path)
            if message:
                st.success(f"Posiciones generadas satisfactoriamente")
            return csv_data
        except Exception as e:
            # Manejar errores durante la carga
            st.error(f"Error al cargar el archivo: {e}")
            return None
    else:
        # Mostrar mensaje de advertencia si el archivo no existe
        #st.warning(f"El archivo {csv_path} no existe.")
        return None
        
def get_ib_positions_by_account(account_id="DU7186453", 
                                ip="127.0.0.1", 
                                port=7497, 
                                client_id=datetime.now().microsecond):
    """
    Extrae las posiciones actuales de una cuenta específica desde Interactive Brokers.

    Args:
        account_id (str): ID de la cuenta (por ejemplo, "DU7186453").
        ip (str): Dirección IP del servidor TWS o IB Gateway.
        port (int): Puerto del servidor TWS o IB Gateway.
        client_id (int): ID del cliente para la conexión a IB.

    Returns:
        list[dict]: Lista de posiciones actuales con detalles de cada símbolo de la cuenta especificada.
    """
    ib = IB()
    try:
        # Conexión a IB
        ib.connect(ip, port, clientId=client_id)
        
        # Obtener todas las posiciones
        all_positions = ib.positions()
        
        # Filtrar posiciones por la cuenta específica
        positions_data = []
        for pos in all_positions:
            if pos.account == account_id:
                contract_details = ib.reqContractDetails(pos.contract)
                expiration_date = None
                if contract_details and contract_details[0].contract.lastTradeDateOrContractMonth:
                    expiration_date = contract_details[0].contract.lastTradeDateOrContractMonth
                
                symbol_data = {
                    "Instrument": pos.contract.symbol,
                    "SecType": pos.contract.secType,
                    "Currency": pos.contract.currency,
                    "Position": pos.position,
                    "AverageCost": pos.avgCost,
                    "ExpirationDate": expiration_date,
                }
                positions_data.append(symbol_data)
        
        return positions_data
    
    except Exception as e:
        print(f"Error al obtener posiciones de la cuenta {account_id}: {e}")
        return []
    finally:
        # Desconexión de IB
        ib.disconnect()
        
def calculate_operations(row):
    N = row["Portfolio Instrument Position"]
    C = row["Position"]
    if abs(C - N) > 0.1 * abs(N):  # Condición simplificada para desviación del 10%
        return int(N - C)  # Operar la diferencia entre N y C
    return 0  # No operar si dentro del rango permitido

    

def execute_order(symbol, 
                  exchange, 
                  secType, 
                  action, 
                  quantity, 
                  price=None, account="DU123456", ip="127.0.0.1", port=7496):
    """
    Ejecuta una orden en Interactive Brokers usando ib_insync.

    Args:
        symbol (str): Símbolo del activo, por ejemplo, 'ES'.
        exchange (str): Mercado en el que se opera, por ejemplo, 'GLOBEX'.
        secType (str): Tipo de instrumento, por ejemplo, 'FUT' o 'STK'.
        action (str): 'BUY' o 'SELL'.
        quantity (int): Número de contratos o acciones.
        price (float, optional): Precio límite para órdenes limitadas; None para órdenes de mercado.
        account (str): ID de la cuenta en IB, por ejemplo, 'DU123456'.
        ip (str): Dirección IP del servidor TWS o Gateway.
        port (int): Puerto del servidor TWS o Gateway.

    Returns:
        dict: Información de la orden ejecutada o error.
    """
    ib = IB()

    try:
        # Conectar a IB
        ib.connect(ip, port, clientId=int(account[-4:]))  # clientId único basado en los últimos dígitos de la cuenta

        # Crear contrato
        contract = Contract(
            symbol=symbol,
            exchange=exchange,
            secType=secType,
            currency="USD"
        )

        # Crear la orden (MarketOrder o LimitOrder según el precio)
        if price is not None:
            order = LimitOrder(action, quantity, price)
        else:
            order = MarketOrder(action, quantity)

        # Enviar la orden
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)  # Esperar para asegurar la ejecución
        trade_status = trade.orderStatus.status

        # Desconectar
        ib.disconnect()

        return {
            "status": trade_status,
            "order_id": trade.order.permId,
            "symbol": symbol,
            "quantity": quantity,
            "action": action,
            "price": price if price is not None else "Market Order"
        }

    except Exception as e:
        ib.disconnect()
        return {"error": str(e)}



def run_buy_sell_position_script(ip, 
                               port, 
                               contracts, 
                               symbol_list, 
                               account):
    try:
        # Serializar los dicts a JSON
        contracts_json = json.dumps(contracts)
        symbol_list_json = json.dumps(symbol_list)
        order_validity="GTC"
        subprocess.run([
            'python', 'execute_orders.py',
            '--ip', ip,
            '--port', str(port),
            '--contracts', contracts_json,
            '--symbol_list', symbol_list_json,
            '--account', account,
            '--order_validity', order_validity
        ], check=True)
    except Exception as e:
        print(f"Error ejecutando el script: {e}")
