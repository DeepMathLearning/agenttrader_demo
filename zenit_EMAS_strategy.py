from api_interface import Main, BarData, PriceInformation, str_to_bool, convert_date_time
import datetime
import time
from threading import Timer
import argparse
import logging
import pandas as pd
import numpy as np
import pytz
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import yfinance as yf
import ta
from tqdm import tqdm
from openorder import orden_status
import math


logger = logging.getLogger()
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO)

class BotZenitEMAS1055(Main):

    def __init__(self, 
                ip, 
                port, 
                client, 
                symbol, 
                secType, 
                currency, 
                exchange, 
                quantity,     
                account,
                interval,
                accept_trade,
                multiplier,
                trading_class,
                lastTradeDateOrContractMonth,
                is_paper,
                order_type="LIMIT", 
                order_validity="DAY"
                 ):
        Main.__init__(self, ip, port, client)
        self.action1 = "BUY"
        self.action2 = "SELL"
        self.ip = ip
        self.port = port
        self.interval = interval
        self.interval1 = None
        self.bar_size = self.convert_to_seconds()
        self.historical_bar_size = "{} secs".format(self.bar_size)
        self.is_paper = is_paper
        self.prices = pd.Series() 
        self.start_time = lambda: datetime.datetime.combine(datetime.datetime.now().date(), datetime.time(9))
        self.stop_time = lambda: datetime.datetime.combine(datetime.datetime.now().date(), datetime.time(17))
        self.pnl_time = lambda: datetime.datetime.combine(datetime.datetime.now().date(), datetime.time(15, 1))
        self.account = account
        self.quantity = quantity
        self.contract = self.CONTRACT_CONFIG()
        self.contract.symbol = symbol
        self.contract.secType = secType
        self.contract.currency = currency
        self.contract.exchange = exchange
        if secType == 'FUT':
            self.contract.multiplier = multiplier
            self.contract.tradingClass = trading_class
            self.contract.lastTradeDateOrContractMonth = lastTradeDateOrContractMonth
        self.min_tick = 0.00005
        self.reqContractDetails(10004, self.contract)

        #risk indicators
        self.today = datetime.datetime.now().date() # 
       
        self.order_type = order_type    

        self.average_purchase_price = 0

        # Datos para la estrategia
        self.accept_trade = accept_trade
        
        # Position
        self.open_position = False

        self.cant_cont = 0 # Cantidad de contratos adquiridos

        self.order_validity = order_validity
        self.current_price = 0
        self.positions1 = {}
        self.pnl_symbol = {}
        self.is_short = False
        self.is_long = False
        self.fdata = None
        self.time_to_wait = 0
        self.open_trade_price = 0
        self.f_pos = False # Close Firts position
        self.s_pos = False # Close Second position
        self.volatilidad = 0 
        self.stop_activate = False
        if self.contract.secType == 'FUT':
            self.symbol = self.contract.tradingClass
        else:
            self.symbol = self.contract.symbol
        
        if self.symbol in ['ES']:
            if self.interval == '1m':
                self.tics_size=2
            elif self.interval == '5m':
                self.tics_size=4
            elif self.interval == '15m':
                self.tics_size=8
        else:
            if self.interval == '1m':
                self.tics_size=12
            elif self.interval == '5m':
                self.tics_size=18
            elif self.interval == '15m':
                self.tics_size=24
        
        self.tics_price=self.tics_size*0.25

        self.candle_force = None
        self.barsize = None

        self.trades_info = {'action':[], 'time':[], 'price':[]}
        self.hora_ejecucion = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.traling_stop_activate = False
        self.cant_cont_init = None
        self.cont_1 = int(self.quantity)

    def main(self):
        self.initial_balance = self.get_account_balance()
        self.highest_balance = self.initial_balance
        logger.info(f"Initical balance {self.initial_balance}")

        seconds_to_wait = (self.start_time() - datetime.datetime.now() + datetime.timedelta(days=1)).total_seconds()
        Timer(seconds_to_wait, self.main).start()

        seconds_to_wait = (self.pnl_time() - datetime.datetime.now()).total_seconds()
        Timer(seconds_to_wait, self.get_pnl).start()


        if not self.isConnected():
            self.reconnect()
        unique_id = self.get_unique_id()
        logger.info('unique_id',unique_id)
        self.market_data[unique_id] = PriceInformation(self.contract)
        
        if self.is_paper:
            self.reqMarketDataType(3)  # Delayed data
        else:
            self.reqMarketDataType(1)  # Live data
        
        
        self.reqMktData(unique_id, self.contract, "", False, False, [])
        self.loop(unique_id)
        self.cancelMktData(unique_id)
    
    def request_firts_position(self):
        self.reqPositions()
        time.sleep(1)
        try:
            print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
        except:
            self.positions1[self.symbol] = {
                "position": 0,
                "averageCost": 0
            }
            
            print(f'**** POSITION {self.positions1[self.symbol]["position"]}')

        self.cant_cont = self.positions1[self.symbol]["position"]

        if self.cant_cont_init is None:
            # if self.cant_cont_init is None:
            # Aumentamos la cantidad de contratos adquiridos
            self.cant_cont_init = self.cant_cont
        else:
            if (self.cant_cont > 0) and (self.cont_1 is not None):
                self.cant_cont_init = self.cant_cont - self.cont_1
            elif (self.cant_cont < 0) and (self.cont_1 is not None):
                self.cant_cont_init = self.cant_cont + self.cont_1
            else:
                self.cant_cont_init = self.cant_cont
        
        print(f'#-----------------------> POSICION INICIAL {self.cant_cont_init}')

    def position_gestion(self):
        logger.info('GESTIONANDO POSICIONES')
        self.reqPositions()
        time.sleep(5)
        try:
            print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
        except:
            self.positions1[self.symbol] = {
                "position": 0,
                "averageCost": 0
            }
            # self.positions1[self.symbol]["position"] = 0
            print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
        # Aumentamos la cantidad de contratos adquiridos
        self.cant_cont = self.positions1[self.symbol]["position"]

        if self.cant_cont < 0:
            self.open_trade_price = float(self.positions1[self.symbol]["averageCost"]) / float(self.contract.multiplier)
            self.open_position = True
            self.fdata.loc[self.fdata[-1:].index[0], 'Open_position'] = -1
            self.trades_info['action'].append('Sell')
            self.trades_info['time'].append(self.fdata[-1:].index[0])
            self.trades_info['price'].append(self.open_trade_price)
        elif self.cant_cont > 0:
            self.open_trade_price = float(self.positions1[self.symbol]["averageCost"]) / float(self.contract.multiplier)
            self.open_position = True
            self.fdata.loc[self.fdata[-1:].index[0], 'Open_position'] = 1
            self.trades_info['action'].append('Buy')
            self.trades_info['time'].append(self.fdata[-1:].index[0])
            self.trades_info['price'].append(self.open_trade_price)

    def loop(self, req_id):

        if self.fdata is None:
            self.get_data(req_id)
            logger.info(f'Se descargó data historica, tamaño {self.fdata.shape}')
        
        if self.redondear_marca_de_tiempo(str(datetime.datetime.now())) in list(self.fdata.index):    
            self.fdata = self.fdata[~self.fdata.index.duplicated(keep=False)]
        
        # self.fdata['Retornos'] = self.fdata['Close'].pct_change()

        # # Calcular la volatilidad histórica (en este caso, se asume 252 días laborables en un año)
        # self.volatilidad = self.fdata['Retornos'].std() * (252**0.5)
        # self.volatilidad = self.volatilidad/2

        if 'Open_position' not in self.fdata.columns:
            self.fdata['Open_position'] = 0
            logger.info('Se agregao indicador Open_position')

        #self.position_gestion()
        
        while True:

            self.estrategy_jemir()
            self.plot_strategy_jemir()
            self.html_generate() 
            self.candle_force_determine()
            print(f'-*-**-*-*-*-*-*-*- TIPO DE TENDENCIA {self.candle_force}')
            print(f'-*-**-*-*-*-*-*-*- DATA {self.fdata.tail(5)}')
            # time.sleep(self.bar_size)
            #self.buy(self.open_trade_price, float(self.quantity), "MARKET")
            #self.reqPnLUpdates(req_id, self.account)
            # Bid Price y Ask Price durante un minuto
            try:
                # Obtén la hora actual
                hora_actual = time.localtime()
                time_part_space = 3600/self.bar_size
                segundos_actuales = (hora_actual.tm_min * 60) + hora_actual.tm_sec
                parte_decimal, _ = math.modf((3600 - segundos_actuales) / self.bar_size)
                segundos_hasta_siguiente_tiempo = int(parte_decimal * self.bar_size)     
                
                if segundos_hasta_siguiente_tiempo > 0:
                    tiempo_de_espera = min(self.bar_size, segundos_hasta_siguiente_tiempo)
                else:
                    tiempo_de_espera = self.bar_size
                datos_prices = []
                logger.info(f'Esperando {tiempo_de_espera} segundos para agregar precios')
                for i in tqdm(range(tiempo_de_espera)):
                    time.sleep(1)
                    try:
                        if self.is_paper:
                            price = (self.market_data[req_id].DelayedBid + self.market_data[req_id].DelayedAsk) / 2
                            market_price = self.market_data[req_id].DelayedAsk
                            vol = self.market_data[req_id].DelayedVolume
                        else:
                            print('LIVE DATA')
                            price = (self.market_data[req_id].Bid + self.market_data[req_id].Ask) / 2
                            market_price = self.market_data[req_id].Ask 
                            vol = self.market_data[req_id].NotDefined
                        logger.info(f'PRECIO ------------> ${price}')
                        logger.info(f'PRECIO DE MERCADO--> ${market_price}')
                        datos_prices.append(price)
                    except TypeError:
                        logger.info(f"Error TypeError, the price is None")
                    
                    if (not self.open_position) or (not self.stop_activate):
                        if (self.accept_trade in ['short100', 'short']) :
                            if (
                                (price < self.fdata['EMA_10'][-1]) and 
                                (self.fdata['EMA_10'][-1] < self.fdata['EMA_55'][-1]) or
                                (self.candle_force == 'bajista')
                                ):
                                break
                                
                        elif (self.accept_trade in ['long100', 'long']):
                            if (
                                (price > self.fdata['EMA_10'][-1]) and 
                                (self.fdata['EMA_10'][-1] > self.fdata['EMA_55'][-1]) or
                                (self.candle_force == 'alcista')
                                ):
                                break

                    if self.open_position:
                        if (self.accept_trade in ['short100', 'short']) and (self.open_trade_price > 0):
                            if (
                                price <= self.open_trade_price-self.tics_price and
                                (not self.f_pos or self.s_pos)
                                ):
                                break
                            # elif (
                            #     self.interval in ['5m', '15m'] and
                            #     price < (self.open_trade_price-(self.tics_price/2)) and
                            #     (not self.f_pos or self.s_pos)
                            #     ):
                            #     break
                            elif price >= self.open_trade_price+self.tics_price:
                                break
                                
                        elif (self.accept_trade in ['long100', 'long']) and (self.open_trade_price > 0):
                            if (
                                price >= self.open_trade_price+self.tics_price and
                                (not self.f_pos or self.s_pos)
                                ):
                                break
                            # elif (
                            #     self.interval in ['5m', '15m'] and
                            #     price < (self.open_trade_price+(self.tics_price/2)) and
                            #     (not self.f_pos or self.s_pos)
                            #     ):
                            #     break
                            elif price <= self.open_trade_price-self.tics_price:
                                break
                        
                new_price_info = pd.DataFrame({
                        'Open':[datos_prices[0]], 
                        'High':[max(datos_prices)], 
                        'Low':[min(datos_prices)], 
                        'Close':[datos_prices[-1]], 
                        'Adj Close':[datos_prices[-1]], 
                        'Volume':[vol],
                        },
                        index=[self.redondear_marca_de_tiempo(str(datetime.datetime.now()- datetime.timedelta(minutes=int(self.bar_size / 60)))+'+00:00')])
                new_price_info['Short_Exit'] = 0
                new_price_info['Open_position'] = 0
                if len(new_price_info) > 0:
                    
                    self.fdata = pd.concat([self.fdata,new_price_info])
                    self.fdata = self.fdata.groupby(self.fdata.index).agg({'Open': 'min', 
                                                                           'High': 'max', 
                                                                           'Low': 'min', 
                                                                           'Close': 'max',
                                                                           'Short_Exit': 'max',
                                                                           'Open_position': 'max'
                                                                           })

                    self.candle_force_determine()
                    print(f'-*-**-*-*-*-*-*-*- TIPO DE TENDENCIA {self.candle_force}')
                    print(f'-*-**-*-*-*-*-*-*- DATA {self.fdata.tail(5)}')
                    
                    self.estrategy_jemir()

                    logger.info('CALCULO DE MÉTRICAS ')
                    logger.info(f'Ultima Vela {self.fdata[-1:].T}')
                    
                    # if self.time_to_wait >= self.bar_size * 2:
                    self.strategy_metrics_jemir(price, req_id)
                    logger.info('ANALIZANDO ESTRATEGIA')
                    
                    self.request_firts_position()             
                        
            except Exception as e:
                logger.info(f'{e}')

    
    def convert_to_seconds(self):
        time_units = {
            'm': 60,     # minutos
            'h': 3600,   # horas
            'd': 86400,  # días
            'wk': 604800,  # semanas
            'mo': 2628000,  # meses (aproximadamente)
            '3mo': 7884000,  # 3 meses (aproximadamente)
        }

        if self.interval in time_units:
            return time_units[self.interval]

        if self.interval.endswith('m'):
            minutes = int(self.interval[:-1])
            return minutes * 60

        if self.interval.endswith('h'):
            hours = int(self.interval[:-1])
            return hours * 3600

        if self.interval.endswith('d'):
            days = int(self.interval[:-1])
            return days * 86400

        raise ValueError("Intervalo de tiempo no válido")
    
    def update_positions(self):
        self.positions1[self.symbol] = {
                        "position": 0,
                        "averageCost": 0
                    }
        self.reqPositions()
        time.sleep(3)

        print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
        self.cant_cont = self.positions1[self.symbol]["position"]

    def redondear_marca_de_tiempo(self, marca_de_tiempo):
        """
        Redondea una marca de tiempo según el formato especificado y la zona horaria.

        Args:
            marca_de_tiempo (str o Timestamp): La marca de tiempo que se va a redondear.
            formato (str): El formato de redondeo ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '1wk', '1mo', '3mo').
            zona_horaria (str): La zona horaria para la marca de tiempo redondeada (por defecto, 'America/New_York').

        Returns:
            Timestamp: La marca de tiempo redondeada en la zona horaria especificada.
        """
        formato = self.interval
        zona_horaria='America/New_York'
        if isinstance(marca_de_tiempo, str):
            # Convierte la cadena a una marca de tiempo si es una cadena
            marca_de_tiempo = pd.to_datetime(marca_de_tiempo)
        
        formatos_validos = ['1m','2m', '5m', '10m','15m', '30m', '60m', '90m', '1h', '1d', '4h','1wk', '1mo', '3mo']
        
        if formato not in formatos_validos:
            raise ValueError("Formato no válido. Use uno de los formatos siguientes: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '1wk', '1mo', '3mo'.")
        
        # Convierte a minutos, horas, días o meses según el formato especificado
        if formato.endswith('m'):
            minutos = int(formato[:-1])
            marca_de_tiempo_redondeada = marca_de_tiempo.round(f'{minutos}T')
        elif formato.endswith('h'):
            horas = int(formato[:-1])
            marca_de_tiempo_redondeada = marca_de_tiempo.round(f'{horas}H')
        elif formato.endswith('d'):
            dias = int(formato[:-1])
            marca_de_tiempo_redondeada = marca_de_tiempo.round(f'{dias}D')
        elif formato.endswith('wk'):
            semanas = int(formato[:-2])
            marca_de_tiempo_redondeada = marca_de_tiempo.round(f'{semanas}W')
        elif formato.endswith('mo'):
            meses = int(formato[:-2])
            marca_de_tiempo_redondeada = marca_de_tiempo.round(f'{meses}M')
        
        # Establece la zona horaria
        # marca_de_tiempo_redondeada = marca_de_tiempo_redondeada.tz_localize(pytz.UTC)
        # marca_de_tiempo_redondeada = marca_de_tiempo_redondeada.astimezone(pytz.timezone(zona_horaria))
        
        return marca_de_tiempo_redondeada

    def get_data(self, req_id):
                    
        if self.interval == '1m':
            self.barsize = "1 min"  # Tamaño de las barras
        elif self.interval == '5m':
            self.barsize = "5 mins"  # Tamaño de las barras
        elif self.interval == '15m':
            self.barsize = "15 mins"  # Tamaño de las barras
        elif self.interval == '10m':
            self.barsize = "10 mins"  # Tamaño de las barras
        elif self.interval == '1h':
            self.barsize = "1 hour"  # Tamaño de las barras
        elif self.interval == '4h':
            self.barsize = "4 hours"  # Tamaño de las barras
        elif self.interval == '30m':
            self.barsize = "30 mins"  # Tamaño de las barras
        elif self.interval == '90m':
            self.barsize = "90 mins"  # Tamaño de las barras
        
        
        if self.interval == '1m':
            duration_str = "60 D"  # Duración de los datos históricos
        elif self.interval in ['2m', '5m','10m', '15m', '30m', '1h']:
            duration_str = "60 D"  # Duración de los datos históricos
    
        elif self.interval in ['4h']:
            duration_str = "1 Y"  # Duración de los datos históricos

        self.historical_market_data[req_id] = self.get_historical_market_data(self.contract, duration_str, self.barsize)
        # print(self.historical_market_data[req_id])
        bar_data_dict_list = [
                                {"Date": data.date, "Open": data.open, "High": data.high, "Low": data.low, "Close": data.close, "Volume": data.volume}
                                for data in self.historical_market_data[req_id]
                            ]
        df = pd.DataFrame(bar_data_dict_list, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        
        df["Date"] = df["Date"].apply(convert_date_time)
        # df["Date"] = pd.to_datetime(df["Date"],  format='%Y-%m-%d %H:%M:%S')
        logger.info(f'FECHA MINIMA DE DESCARGA IBAPI *** {df["Date"].min()}')
        logger.info(f'FECHA MAXIMA DE DESCARGA IBAPI *** {df["Date"].max()}')
        df.set_index("Date", inplace=True)
        df.to_csv(f'data_ib_{self.interval}.csv')
        self.fdata = df
        self.fdata.index = pd.to_datetime(self.fdata.index, format='%Y-%m-%d %H:%M:%S', utc=True)

        
        # Set 'Date' column as the index
        self.fdata = self.fdata[~self.fdata.index.duplicated(keep='last')]
        self.fdata = self.fdata[self.fdata['Close']>0].sort_index()
        
        #print(self.historical_market_data[req_id])

    def estrategy_jemir(self):
        self.fdata = self.fdata[self.fdata['Close']>0]
        ##################################################
        # Estrategia
        #################################################
        # Calcular indicadores necesarios
        self.fdata['EMA_10'] = ta.trend.ema_indicator(self.fdata['Close'], window=10)
        self.fdata['EMA_55'] = ta.trend.ema_indicator(self.fdata['Close'], window=55)
        
        # Definir umbrales y condiciones
    
        # Creamos una nueva columna
        # data1['Entry'] = 0.0
        self.fdata['Long_Signal'] = np.where(self.fdata['EMA_10'] >= self.fdata['EMA_55'], 1.0, 0.0)
        self.fdata['Short_Signal'] = np.where(self.fdata['EMA_55'] >= self.fdata['EMA_10'], 1.0, 0.0)
        self.fdata['Posicion'] = self.fdata['Long_Signal'].diff()
        
    def strategy_metrics_jemir(self, price, req_id):
        
        # Iterar a través de los datos para simular la estrategia
        last_row = self.fdata[-1:]
        logger.info(f' * * * * * * Hora de actualización: {last_row.index[0]}')

        self.current_price = price
        if self.accept_trade == "short":
            print('****************** ESTAS EN SHORT')
            print(f'SHORT *************** {self.fdata["Short_Signal"][last_row.index[0]]}')
            # Señal de venta
            if not self.open_position and (self.fdata['Short_Signal'][last_row.index[0]] == 1): 
                self.open_trade_price = self.current_price
                self.sell(self.open_trade_price, float(self.quantity), "LIMIT")
                self.update_positions()

                if self.cant_cont != self.cant_cont_init:
                    self.request_firts_position()
                    self.open_position = True
                    self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                    self.trades_info['action'].append('Sell')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.open_trade_price)
                else:
                    tiempo_inicio = time.time()

                    for i in tqdm(range(120), desc="Espera, ejecución orden de entrada"):
                        self.request_firts_position()
                        # Actualizar las posiciones y otras variables necesarias
                        self.reqPositions()
                        # Esperar 1 segundo
                        time.sleep(1)
                        print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                        self.cant_cont = self.positions1[self.symbol]["position"]

                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont != self.cant_cont_init:
                            self.request_firts_position()
                            self.open_position = True
                            self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                            self.trades_info['action'].append('Sell')
                            self.trades_info['time'].append(last_row.index[0])
                            self.trades_info['price'].append(self.open_trade_price)
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= 120:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.reqGlobalCancel()
                            #self.get_data(req_id)
                            break

            # Toma de ganancias
            elif self.open_position and (self.fdata['Long_Signal'][last_row.index[0]] == 1):  
                self.buy(self.current_price, float(self.quantity), "LIMIT")
                    
                self.update_positions()
                
                if self.cant_cont > self.cant_cont_init:
                    
                    tiempo_inicio = time.time()

                    for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                        self.request_firts_position()
                        self.update_positions()

                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont == self.cant_cont_init:
                            self.open_position = False
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= 120:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.reqGlobalCancel()
                            #self.get_data(req_id)
                            self.open_position = True
                            break 

                if self.cant_cont == self.cant_cont_init:
                    self.request_firts_position()
                    self.open_position = False
                    self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                    self.trades_info['action'].append('Buy')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.current_price)

        elif self.accept_trade == "short100":
            print('****************** ESTAS EN SHORT')
            print(f'SHORT *************** {self.fdata["Short_Signal"][last_row.index[0]]}')

            if (self.fdata['Short_Signal'][last_row.index[0]] == 0 or
                self.current_price < self.fdata['EMA_10'][last_row.index[0]]):
                self.stop_activate = False

            if self.open_position:
                ###########################################################################
                # Cierre de primera posicion por stop Loss                                #
                ###########################################################################
                if (
                    self.current_price > self.open_trade_price and
                    self.current_price >= self.fdata['EMA_55'][last_row.index[0]]+0.25 and #self.open_trade_price+self.tics_price and 
                    (not self.f_pos or self.s_pos)
                    ):  # Señal de Compra
                    self.buy(self.current_price, float(self.quantity), "LIMIT")
                    
                    self.update_positions()

                    
                    if self.cant_cont != self.cant_cont_init:
                        
                        tiempo_inicio = time.time()

                        for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                            self.request_firts_position()
                            self.update_positions()
                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont == self.cant_cont_init:
                                self.open_position = False
                                self.f_pos = False
                                self.s_pos = False
                                self.stop_activate = True
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= 120:
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                #self.get_data(req_id)
                                self.open_position = True
                                break 
                   

                    if self.cant_cont == self.cant_cont_init:
                        self.request_firts_position()
                        self.open_position = False
                        self.f_pos = False
                        self.s_pos = False
                        self.stop_activate= True
                        self.open_trade_price = 0
                        self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                        self.trades_info['action'].append('Buy')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.current_price)

                ############################################################
                # Compra (Cierra) la primera posición                      #
                ############################################################
                elif (
                    self.fdata['Short_Signal'][last_row.index[0]] == 1 and 
                    not self.f_pos and 
                    self.current_price < self.open_trade_price 
                    ):
                    self.buy(self.current_price, float(self.quantity), "LIMIT")
                    
                    self.update_positions()
                    
                    if self.cant_cont != self.cant_cont_init:
                        
                        tiempo_inicio = time.time()

                        for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                            self.request_firts_position()
                            self.update_positions()
                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont == self.cant_cont_init:
                                self.f_pos = True
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= 120:
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                #self.get_data(req_id)
                                self.open_position = True
                                break 

                    if self.cant_cont == self.cant_cont_init:
                        self.request_firts_position()
                        self.f_pos = True
                        self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                        self.trades_info['action'].append('Buy')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.current_price)
                    time.sleep(30)
                
                    
                ############################################################
                # Venta, toma de las segundas posiciones                   #
                ############################################################
                elif (
                    self.fdata['Short_Signal'][last_row.index[0]] == 1 and 
                    self.current_price < self.fdata['EMA_10'][last_row.index[0]] and
                    self.f_pos and 
                    not self.s_pos and 
                    self.current_price <= self.open_trade_price 
                    ):
                    self.open_trade_price = self.current_price
                    self.sell(self.open_trade_price, float(self.quantity), "LIMIT")
                    self.wait_to_execute_sell_short(req_id)
                    if self.cant_cont != self.cant_cont_init:
                        self.request_firts_position()
                        self.s_pos = True
                        self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                        self.trades_info['action'].append('Sell')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.open_trade_price)
                
                ############################################################
                # Compra (cierre) de las segundas posiciones               #
                ############################################################
                elif (
                    self.fdata['Short_Signal'][last_row.index[0]] == 1 and 
                    self.f_pos and 
                    self.s_pos and 
                    self.current_price < self.open_trade_price
                    ):
                    self.buy(self.current_price, float(self.quantity), "LIMIT")
                    
                    self.update_positions()
                    
                    if self.cant_cont != self.cant_cont_init:
                        
                        tiempo_inicio = time.time()

                        for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                            self.request_firts_position()
                            self.update_positions()
                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont == self.cant_cont_init:
                                self.s_pos = False
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= 120:
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                #self.get_data(req_id)
                                break 

                    if self.cant_cont == self.cant_cont_init:
                        self.request_firts_position()
                        self.s_pos = False
                        self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                        self.trades_info['action'].append('Buy')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.current_price)
                    time.sleep(30)

             # Señal de venta, Toma de posicion
            elif (
                self.fdata['Short_Signal'][last_row.index[0]] == 1 and
                self.candle_force == 'bajista' and
                self.current_price < self.fdata['EMA_10'][last_row.index[0]]
                ):
                if not self.stop_activate:
                    self.open_trade_price = self.current_price
                    self.sell(self.open_trade_price, float(self.quantity), "LIMIT")
                    self.wait_to_execute_sell_short(req_id)
                    if self.cant_cont != self.cant_cont_init:
                        self.request_firts_position()
                        self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                        self.trades_info['action'].append('Sell')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.open_trade_price)
                    
            
        
        elif self.accept_trade == "long100":
            logger.info('****************** ESTAS EN LONG')
            logger.info(f'LONG *************** {self.fdata["Long_Signal"][last_row.index[0]]}')

            if (self.fdata["Long_Signal"][last_row.index[0]] == 0 or
                self.current_price > self.fdata['EMA_10'][last_row.index[0]]):
                self.stop_activate=False

            if self.open_position:
                if (
                    self.current_price < self.open_trade_price and
                    self.current_price <= self.fdata['EMA_55'][last_row.index[0]]+0.25 and #self.open_trade_price-self.tics_price and
                    (not self.f_pos or self.s_pos)
                    ):  # Señal de venta
                    self.sell(self.current_price, float(self.quantity), "LIMIT")
                    self.update_positions()
                    
                    
                    if self.cant_cont != self.cant_cont_init:
                        
                        tiempo_inicio = time.time()

                        for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                            self.request_firts_position()
                            self.update_positions()
                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont == self.cant_cont_init:
                                self.open_position = False
                                self.f_pos = False
                                self.s_pos = False
                                self.stop_activate = True
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= 120:
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                #self.get_data(req_id)
                                self.open_position = True
                                break 
                                            
                    if self.cant_cont == self.cant_cont_init:
                        self.request_firts_position()
                        self.open_position = False
                        self.f_pos = False
                        self.s_pos = False
                        self.stop_activate = True
                        self.open_trade_price = 0
                        self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                        self.trades_info['action'].append('Sell')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.current_price)


                ############################################################
                # Venta (Cierra) la primera posición                       #
                ############################################################

                elif (self.fdata['Long_Signal'][last_row.index[0]] == 1 and 
                      not self.f_pos and 
                      self.current_price > self.open_trade_price
                      ):
                    self.sell(self.current_price, float(self.quantity), "LIMIT")
                    
                    self.update_positions()
                    
                    if self.cant_cont != self.cant_cont_init:
                        
                        tiempo_inicio = time.time()

                        for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                            self.request_firts_position()
                            self.update_positions()
                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont == 0:
                                self.f_pos = True
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= 120:
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                #self.get_data(req_id)
                                self.open_position = True
                                break 
                        
                    if self.cant_cont == self.cant_cont_init:
                        self.request_firts_position()
                        self.f_pos = True
                        self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                        self.trades_info['action'].append('Sell')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.current_price)
                    time.sleep(30)
                

                                    
                ############################################################
                # Compra, toma de la segunda posicion                      #
                ############################################################ 

                elif (
                    self.fdata['Long_Signal'][last_row.index[0]] == 1 and 
                    self.current_price > self.fdata['EMA_10'][last_row.index[0]] and
                    self.f_pos and 
                    not self.s_pos and 
                    self.current_price >= self.open_trade_price 
                    ) :

                    self.open_trade_price = self.current_price
                    self.buy(self.open_trade_price, float(self.quantity), "LIMIT")
                    self.wait_to_execute_buy_long(req_id)
                    if self.cant_cont > self.cant_cont_init:
                        self.request_firts_position()
                        self.s_pos = True
                        self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                        self.trades_info['action'].append('Buy')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.current_price)
                
                ############################################################
                # Venta (cierre) las demas posiciones                      #
                ############################################################
                 
                elif (
                    self.fdata['Long_Signal'][last_row.index[0]] == 1 and 
                    self.f_pos and 
                    self.s_pos and 
                    self.current_price > self.open_trade_price
                    ):
                    self.sell(self.current_price, float(self.quantity), "LIMIT")
                    
                    self.update_positions()

                    if self.cant_cont != self.cant_cont_init:
                        
                        tiempo_inicio = time.time()

                        for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                            self.request_firts_position()
                            self.update_positions()
                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont == self.cant_cont_init:
                                self.s_pos = False
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= 120:
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                #self.get_data(req_id)
                                break 
                    # except:
                    #     self.positions1[self.symbol]["position"] = 0
                    #     print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                    #     # Aumentamos la cantidad de contratos adquiridos
                    #     self.cant_cont = self.positions1[self.symbol]["position"]

                    if self.cant_cont == self.cant_cont_init:
                        self.request_firts_position()
                        self.s_pos = False
                        self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                        self.trades_info['action'].append('Sell')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.current_price)
                    time.sleep(30)
                    
             # Señal de compra, Toma de posicion
            elif (
                self.fdata['Long_Signal'][last_row.index[0]] == 1  and
                self.candle_force == 'alcista' and
                self.current_price > self.fdata['EMA_10'][last_row.index[0]]
                ):
                if not self.stop_activate:
                    print('**************** POSICION DE ENTRADA')
                    self.open_trade_price = self.current_price
                    self.buy(self.open_trade_price, float(self.quantity), "LIMIT")
                    self.wait_to_execute_buy_long(req_id)
                    if self.cant_cont > self.cant_cont_init:
                        self.request_firts_position()
                        self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                        self.trades_info['action'].append('Buy')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.open_trade_price)
                

        elif self.accept_trade == "long":
            print('****************** ESTAS EN LONG')
            print(f'LONG *************** {self.fdata["Long_Signal"][last_row.index[0]]}')
            # Señal de compra
            if not self.open_position and (self.fdata['Long_Signal'][last_row.index[0]] == 1): 
                self.open_trade_price = self.current_price
                self.buy(self.open_trade_price, float(self.quantity), "LIMIT")
                
                self.update_positions()

                if self.cant_cont > self.cant_cont_init:
                    self.request_firts_position()
                    self.open_position = True
                    self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                    self.trades_info['action'].append('Buy')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.open_trade_price)
                else:
                    tiempo_inicio = time.time()

                    for i in tqdm(range(120), desc="Espera, ejecución orden de entrada"):
                        self.request_firts_position()
                        # Actualizar las posiciones y otras variables necesarias
                        self.update_positions()

                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont > self.cant_cont_init:
                            self.request_firts_position()
                            self.open_position = True
                            self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                            self.trades_info['action'].append('Buy')
                            self.trades_info['time'].append(last_row.index[0])
                            self.trades_info['price'].append(self.open_trade_price)
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= 120:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.reqGlobalCancel()
                            #self.get_data(req_id)
                            break


            # Señal de venta    
            elif self.open_position and (self.fdata['Short_Signal'][last_row.index[0]] == 1):  
                self.sell(self.current_price, float(self.quantity), "LIMIT")
                    
                self.update_positions()
                
                
                if self.cant_cont != self.cant_cont_init:
                    
                    tiempo_inicio = time.time()

                    for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                        self.request_firts_position()
                        self.update_positions()
                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont == 0:
                            self.open_position = False
                            self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= 120:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.reqGlobalCancel()
                            #self.get_data(req_id)
                            self.open_position = True
                            break 
                 
                if self.cant_cont == self.cant_cont_init:
                    self.request_firts_position()
                    self.open_position = False
                    self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                    self.trades_info['action'].append('Sell')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.current_price)

                    
        elif self.accept_trade == "short-long":
            # Señal de compra
            if not self.open_position and ((self.fdata['Short_Signal'][last_row.index[0]] == 1) or (self.fdata['Long_Signal'][last_row.index[0]] == 1)): 
                if self.fdata['Short_Signal'][last_row.index[0]] == 1:
                    print('****************** ESTAS EN SHORT')
                    print(f'SHORT *************** {self.fdata["Short_Signal"][last_row.index[0]]}')
                    self.is_short = True
                elif self.fdata['Long_Signal'][last_row.index[0]] == 1:
                    print('****************** ESTAS EN LONG')
                    print(f'LONG *************** {self.fdata["Long_Signal"][last_row.index[0]]}')
                    self.is_long = True 

                self.open_trade_price = self.current_price

                if self.is_short:
                    self.sell(self.open_trade_price, float(self.quantity), "LIMIT")
                else:
                    self.buy(self.open_trade_price, float(self.quantity), "LIMIT")

                self.reqPositions()
                time.sleep(5)
                try:
                    print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                except:
                    self.positions1[self.symbol]["position"] = 0
                    print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                # Aumentamos la cantidad de contratos adquiridos
                self.cant_cont = self.positions1[self.symbol]["position"]

                if self.cant_cont != 0:
                    self.open_position = True
                    if self.is_short:
                        self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                        self.trades_info['action'].append('Sell')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.open_trade_price)
                    else:
                        self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                        self.trades_info['action'].append('Buy')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.open_trade_price)
                else:
                    tiempo_inicio = time.time()

                    for i in tqdm(range(120), desc="Espera, ejecución orden de entrada"):
                        
                        # Actualizar las posiciones y otras variables necesarias
                        self.reqPositions()
                        # Esperar 1 segundo
                        time.sleep(1)
                        print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                        self.cant_cont = self.positions1[self.symbol]["position"]

                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont != 0:
                            self.open_position = True
                            if self.is_short:
                                self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                                self.trades_info['action'].append('Sell')
                                self.trades_info['time'].append(last_row.index[0])
                                self.trades_info['price'].append(self.open_trade_price)
                            else:
                                self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                                self.trades_info['action'].append('Buy')
                                self.trades_info['time'].append(last_row.index[0])
                                self.trades_info['price'].append(self.open_trade_price)
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= 120:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.reqGlobalCancel()
                            #self.get_data(req_id)
                            break


            # Señal de venta    
            if self.open_position and self.is_short and (self.fdata['Long_Signal'][last_row.index[0]] == 1):  
                self.buy(self.current_price, float(self.quantity), "LIMIT")
                self.is_short = False 
                self.reqPositions()   
                time.sleep(5)
                # Consulta de posiciones en cuenta
                
                try:
                    self.cant_cont = self.positions1[self.symbol]["position"]
                    print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                    if self.cant_cont != 0:
                        
                        tiempo_inicio = time.time()

                        for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                            
                            # Actualizar las posiciones y otras variables necesarias
                            self.reqPositions()
                            # Esperar 1 segundo
                            time.sleep(1)
                            try:
                                print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                            except:
                                self.positions1[self.symbol]["position"] = 0
                                print(f'**** POSITION {self.positions1[self.symbol]["position"]}')

                            self.cant_cont = self.positions1[self.symbol]["position"]
                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont == 0:
                                self.open_position = False
                                self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= 120:
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                #self.get_data(req_id)
                                self.open_position = True
                                break 
                except:
                    self.positions1[self.symbol]["position"] = 0
                    print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                    # Aumentamos la cantidad de contratos adquiridos
                    self.cant_cont = self.positions1[self.symbol]["position"]
                
                if self.cant_cont == 0:
                    self.open_position = False
                    self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                    self.trades_info['action'].append('Buy')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.current_price)
                

            elif self.open_position and self.is_long and (self.fdata['Short_Signal'][last_row.index[0]] == 1):  
                self.sell(self.current_price, float(self.quantity), "LIMIT")
                self.is_long = False   
                self.reqPositions() 
                time.sleep(5)
                # Consulta de posiciones en cuenta
                
                try:
                    self.cant_cont = self.positions1[self.symbol]["position"]
                    print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                    if self.cant_cont != 0:
                        
                        tiempo_inicio = time.time()

                        for i in tqdm(range(120), desc="Espera, ejecución orden de venta"):
                            
                            # Actualizar las posiciones y otras variables necesarias
                            self.reqPositions()
                            # Esperar 1 segundo
                            time.sleep(1)
                            try:
                                print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                            except:
                                self.positions1[self.symbol]["position"] = 0
                                print(f'**** POSITION {self.positions1[self.symbol]["position"]}')

                            self.cant_cont = self.positions1[self.symbol]["position"]
                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont == 0:
                                self.open_position = False
                                self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= 120:
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                #self.get_data(req_id)
                                self.open_position = True
                                break 
                except:
                    self.positions1[self.symbol]["position"] = 0
                    print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
                    # Aumentamos la cantidad de contratos adquiridos
                    self.cant_cont = self.positions1[self.symbol]["position"]
                
                if self.cant_cont == 0:
                    self.open_position = False
                    self.fdata.loc[last_row.index[0], 'Open_position'] = -1
                    self.trades_info['action'].append('Sell')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.current_price)

    def buy(self, price, cant, action):
        logger.info("=========================================================== Placing buy order: {}"
              .format(round(price, 5)))

        order_id = self.get_order_id()

        if action == "MARKET":
            order = self.market_order(self.action1, float(cant))
        elif action == "LIMIT":
            order = self.limit_order(self.action1, float(cant), self.min_price_increment(price), self.account)
        else:
            order = self.stop_limit_order(self.action1, float(cant), self.min_price_increment(price), self.account)
        
        self.placeOrder(order_id, self.contract, order)
        

    def sell(self, price, cant, action):
        logger.info("=========================================================== Placing sell order: {}"
              .format(round(price, 5)))

        order_id = self.get_order_id()
        if action == "MARKET":
            order = self.market_order(self.action2, float(cant))
        elif action == "LIMIT":
            order = self.limit_order(self.action2, float(cant), self.min_price_increment(price), self.account)
        else:
            order = self.stop_limit_order(self.action2, float(cant), self.min_price_increment(price), self.account)
 
        self.placeOrder(order_id, self.contract, order)

    def candle_force_determine(self):
        """
        Evalúa la fuerza positiva o negativa de una vela en un gráfico financiero.

        Parámetros:
            - entrada (float): Precio de entrada.
            - salida (float): Precio de salida.
        """
        apertura = float(self.fdata.iloc[-1]['Open'])
        cierre = float(self.fdata.iloc[-1]['Close'])
        alto = float(self.fdata.iloc[-1]['High'])
        bajo = float(self.fdata.iloc[-1]['Low'])

        if (cierre - apertura) > 0:
            if (
                ((cierre == alto) and (apertura - bajo)> 0) or
                ((cierre == alto) and (apertura == bajo)) or
                ((alto - cierre) > 0 and (alto - cierre) < (apertura - bajo)) or
                ((alto - cierre) > 0 and (apertura == bajo))
            ):

                self.candle_force = 'alcista'
            else:
                self.candle_force = 'neutro'
        
        elif (cierre - apertura) < 0:
            if (
                ((cierre == bajo) and (alto - apertura) > 0) or
                ((apertura == alto) and (cierre == bajo)) or
                ((alto - apertura) > 0 and (alto - apertura) > (cierre - bajo)) or
                ((cierre - bajo) > 0 and (apertura == alto))
            ):

                self.candle_force = 'bajista'
            else:
                self.candle_force = 'neutro'
        else:
            self.candle_force = 'neutro'

    def wait_to_execute_sell_short(self, req_id):
        self.update_positions()

        if self.cant_cont != self.cant_cont_init:
            self.open_position = True
        else:
            tiempo_inicio = time.time()

            for i in tqdm(range(120), desc="Espera, ejecución orden de entrada"):
                self.request_firts_position()
                self.update_positions()

                # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                if self.cant_cont != self.cant_cont_init:
                    self.open_position = True
                    break

                # Verificar si ha transcurrido el tiempo límite
                tiempo_transcurrido = time.time() - tiempo_inicio
                if tiempo_transcurrido >= 120:
                    # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                    self.reqGlobalCancel()
                    #self.get_data(req_id)
                    break
    
    def wait_to_execute_buy_long(self, req_id):
        self.update_positions()

        if self.cant_cont > self.cant_cont_init:
            self.open_position = True
        else:
            tiempo_inicio = time.time()

            for i in tqdm(range(120), desc="Espera, ejecución orden de entrada"):
                self.request_firts_position()
                self.update_positions()

                # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                if self.cant_cont > self.cant_cont_init:
                    self.open_position = True
                    break

                # Verificar si ha transcurrido el tiempo límite
                tiempo_transcurrido = time.time() - tiempo_inicio
                if tiempo_transcurrido >= 120:
                    # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                    self.reqGlobalCancel()
                    #self.get_data(req_id)
                    break
                
    def wait_to_execute_sell_long(self, req_id):
        self.update_positions()
        
        if self.cant_cont != 0:
            
            tiempo_inicio = time.time()

            for i in tqdm(range(60), desc="Espera, ejecución orden de venta"):
                
                self.update_positions()
                # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                if self.cant_cont == 0:
                    self.s_pos = False
                    break

                # Verificar si ha transcurrido el tiempo límite
                tiempo_transcurrido = time.time() - tiempo_inicio
                if tiempo_transcurrido >= 60:
                    # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                    self.reqGlobalCancel()
                    #self.get_data(req_id)
                    break 
       
        if self.cant_cont == 0:
            self.s_pos = False
    def plot_strategy_jemir(self):
        self.figure=go.Figure(go.Candlestick(
                        x = self.fdata.index,
                        open = self.fdata['Open'],
                        high = self.fdata['High'],
                        low = self.fdata['Low'],
                        close = self.fdata['Close'],
                        name=f'Precio de {self.symbol}' 
                        ))
        # fig.add_trace( go.Bar(x=[1, 2, 3, 4], y=[7, 4, 5, 6], name='bar', orientation = 'h',opacity = 0.5), secondary_y=True)
        # Agregar el gráfico de barras de volumen en la segunda columna (encima del gráfico de velas)
        
                
        self.figure.add_trace(go.Scatter(
                x=self.fdata[self.fdata['Open_position'] == 1].index,
                y=self.fdata['Close'][self.fdata['Open_position'] == 1],
                mode= 'markers',
                name = 'Compra',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='star-triangle-up'
                ) ) )
        
               
        # Ploteando Señales de VENTA
        self.figure.add_trace(go.Scatter(
            x=self.fdata[self.fdata['Open_position'] == -1].index,
            y=self.fdata['Close'][self.fdata['Open_position'] == -1],
            mode= 'markers',
            name = 'Venta',
            marker=dict(
                size=15,
                color='cyan',
                symbol='star-triangle-down'
            )
                                ))
        
        self.figure.add_trace(
                            go.Scatter(
                            x= self.fdata.index, 
                            y=self.fdata['EMA_55'],
                            line = dict(color='orange',width=2),
                            name='EMA 55'
                            ))
        self.figure.add_trace(
                            go.Scatter(
                            x= self.fdata.index, 
                            y=self.fdata['EMA_10'],
                            line = dict(color='blue',width=2),
                            name='EMA 10'
                            ))
        
        
        #self.figure.data[1].update(xaxis='x2')
        self.figure.update_layout(xaxis_rangeslider_visible=False)
        self.figure.update_layout(width=1500, height=1000)
        self.figure.update_layout(title=f"Estrategia aplicada a {self.symbol} en el intervalo {self.interval}")

    def html_generate(self):
        logger.info('GENERANDO EL HTML ********')
        self.plot_div = pyo.plot(self.figure, output_type='div', include_plotlyjs='cdn', image_width= 1200)
        
        style = '''
                body * {
                    box-sizing: border-box;
                }
                header {
                    display: block;
                }
                #main-header{
                            background-color: #373a36ff;
                            }
                #main-header .inwrap {
                            width: 100%;
                            max-width: 80em;
                            margin: 0 auto;
                            padding: 1.5em 0;
                            display: -webkit-box;
                            display: -ms-flexbox;
                            display: flex;
                            -webkit-box-pack: justify;
                            -ms-flex-pack: justify;
                            justify-content: space-between;
                            -webkit-box-align: center;
                            -ms-flex-align: center;
                            align-items: center;
                            }
        
        '''
        activity = ''
        for i in range(len(self.trades_info['action'])):
            activity += '<li>Operación: '+ self.trades_info['action'][i] + '; Precio: ' + str(self.trades_info['price'][i])+'; Fecha: ' + str(self.trades_info['time'][i]) + '</li>'
        # Crear el archivo HTML y escribir el código de la gráfica en él
        with open(f"bot_activity/{self.accept_trade}Bot_{self.interval}_{self.symbol}_{self.hora_ejecucion}.html", "w") as html_file:
            html_file.write(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Gráfica Plotly</title>
                <!-- Incluir la biblioteca Plotly de CDN -->
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    {style}
                </style>
            </head>
            <body>
                <header id='main-header'>
                    <div class='inwrap'>
                        <img src='zenit_logo_dor.png' width='10%'>
                    </div>
                </header>
                <!-- Div donde se mostrará la gráfica Plotly -->
                <div id="plotly-div" style="width:100%" align="center">{self.plot_div}</div>
                <div>
                    <center>
                        <h2> Operaciones realizadas por el Bot </h2>
                    
                        <ul>
                            {activity}
                        </ul>
                    </center>
                </div>
            </body>
            </html>
            """)


if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address')
    parser.add_argument('--port', type=int, default=7497, help='Port number')
    parser.add_argument('--client', type=int, default=6, help='Client ID')
    parser.add_argument('--symbol', type=str, default='ESU4', help='symbol example AAPL')
    parser.add_argument('--secType', type=str, default='FUT', help='The security type')
    parser.add_argument('--currency', type=str, default='USD', help='currency')
    parser.add_argument('--exchange', type=str, default='CME', help='exchange')
    parser.add_argument('--quantity', type=str, default='1', help='quantity')
    
    parser.add_argument('--account', type=str, default='DU7774793', help='Account')

    parser.add_argument('--interval', type=str, default='1m', help='Data Time Frame')
    parser.add_argument('--accept_trade', type=str, default='short-long', help='Type of trades for trading')
    parser.add_argument('--multiplier', type=str, default="50", help='The multiplier for futures')
    parser.add_argument('--trading_class', type=str, default="ES", help='The trading_class for futures')
    parser.add_argument('--lastTradeDateOrContractMonth', type=str, default="20240920", help='The expire date for futures')
    parser.add_argument('--order_type', type=str, default="LIMIT", help='The type of the order: LIMIT OR MARKET')
    parser.add_argument('--order_validity', type=str, default="DAY", help='The expiration time of the order: DAY or GTC')
    parser.add_argument('--is_paper', type=str_to_bool, default=True, help='Paper or live trading')
           
    
    
    args = parser.parse_args()
    logger.info(f"args {args}")

    bot = BotZenitEMAS1055(args.ip, 
              args.port, 
              args.client, 
              args.symbol, 
              args.secType, 
              args.currency, 
              args.exchange, 
              args.quantity, 
 
              args.account,

              args.interval,
              args.accept_trade,
              args.multiplier,
              args.trading_class,
              args.lastTradeDateOrContractMonth,
              args.is_paper,
              args.order_type, 
              args.order_validity
              )
    try:
        bot.main()
    except KeyboardInterrupt:
        bot.disconnect()

