from api_interface import (Main, 
                           BarData, 
                           PriceInformation, 
                           str_to_bool, 
                           convert_date_time, 
                           initialize_db, 
                           store_action,
                           generate_random_id)
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
import base64
import math
from ta.trend import ADXIndicator


futures_symbols_info = pd.read_csv("data/futures_symbols_v1.csv")

logger = logging.getLogger()
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO)

class BotZenitCRUCEEMAS(Main):

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
                order_validity="DAY",
                hora_ejecucion=None,
                with_trend_study = False,
                with_macd_signal = False,
                smart_interval = 'auto'
                 ):
        Main.__init__(self, ip, port, client)
        self.strategy_name = 'TA'
        self.action1 = "BUY"
        self.action2 = "SELL"
        self.with_trend_study = with_trend_study
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
        self.out_position = False
        if secType == 'FUT':
            self.contract.multiplier = multiplier
            self.contract.tradingClass = trading_class
            self.contract.lastTradeDateOrContractMonth = lastTradeDateOrContractMonth
        
        self.mid_point = None
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
        self.positions1 = {self.account:{}}
        self.pnl_symbol = {}
        self.is_short = False
        self.is_long = False
        self.fdata = None
        self.time_to_wait = 0
        self.open_trade_price = 0
        self.f_pos = False # Close Firts position
        self.s_pos = False # Close Second position
        self.volatilidad = 0 
        if self.contract.secType == 'FUT':
            self.symbol = self.contract.tradingClass
        else:
            self.symbol = self.contract.symbol
        
                
        self.barsize = None

        self.trades_info = {'action':[], 
                            'time':[], 
                            'price':[], 
                            'contracts':[],
                            'Open_position':[], 
                            'Short_Exit':[]}
        if hora_ejecucion is None:
            self.hora_ejecucion = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.hora_ejecucion = hora_ejecucion
        self.traling_stop_activate = False
        self.cant_cont_init = None
        self.cont_1 = None
        self.last_two_adx_values = None
        self.adx_slope = None
        self.daily_returns = None
        self.volatility_sd = None
        self.volatility = None
        self.trailing_stop_activate_di = False
        self.trailing_stop_activate_price = False
        self.trailing_stop_activate_di_40 = False
        self.trailing_stop_DTC = False
        self.di_plus = None
        self.di_minus = None
        self.di_40 = None
        self.with_macd_signal = with_macd_signal
        self.short_entry_signal = None
        self.long_entry_signal = None

        self.wait = 300 # Tiempo de espera de ejecución de orden 5min

        self.df_activity = None        
        self.di_minus_trend = None
        self.di_plus_trend = None
        self.DTC = None
        self.macd_min = None
        self.macd_max = None
        
        self.tick_usd_value = float(futures_symbols_info[futures_symbols_info['Symbol'] == self.symbol]['Value'].iloc[0])
        self.tick_value = futures_symbols_info[futures_symbols_info['Symbol'] == self.symbol]['Tick'].iloc[0]
        self.name = futures_symbols_info[futures_symbols_info['Symbol'] == self.symbol]['Name'].iloc[0]
        
        if self.account == 'DU7186452':
            if 'Micro' in self.name:
                self.USD_1m = 100 / (float(self.tick_usd_value)*10)
                self.USD_5m = 176 / (float(self.tick_usd_value)*10)
                self.USD_max = 300 / (float(self.tick_usd_value)*10)
            else:
                self.USD_1m = 100 / float(self.tick_usd_value)
                self.USD_5m = 176 / float(self.tick_usd_value)
                self.USD_max = 300 / float(self.tick_usd_value)
        else:
            if 'Micro' in self.name:
                self.USD_1m = 50 / (float(self.tick_usd_value)*10)
                self.USD_5m = 88 / (float(self.tick_usd_value)*10)
                self.USD_max = 150 / (float(self.tick_usd_value)*10)
            else:
                self.USD_1m = 50 / float(self.tick_usd_value)
                self.USD_5m = 88 / float(self.tick_usd_value)
                self.USD_max = 150 / float(self.tick_usd_value)
        
        
        self.tick_price = round(self.tick_value * self.USD_max, 2)

        if self.interval == '1m':
            self.DTC = round(self.tick_value * self.USD_1m, 2)
            self.call_down_wait = 3600
        elif self.interval == '5m':
            self.DTC = round(self.tick_value * self.USD_5m, 2)
            self.call_down_wait = 7200
        elif self.interval == '15m':
            self.DTC = round((self.tick_value * self.USD_5m * 2), 2)
            self.call_down_wait = 14400
        elif self.interval == '1h':
            self.DTC = round((self.tick_value * self.USD_5m * 4), 2)
            self.call_down_wait = 21600
        elif self.interval == '4h':
            self.DTC = round((self.tick_value * self.USD_5m * 6), 2)
            self.call_down_wait = 43200
        else:
            self.DTC = round((self.tick_value * self.USD_5m * 2), 2)
            self.call_down_wait = 14400
        
        self.trade_id = generate_random_id()
        self.new_barsize = None
        
        ## Definir el intervalo de descarga de datos para la capa smart
        self.smart_interval = smart_interval
        self.new_barsize = None
        self.smart_duration_str = None
        self.order_id = None

    def main(self):
        unique_id = self.get_unique_id()
        initialize_db(db_name='zenit_oms.db')
        
        if self.with_trend_study:
            self.trend_styudy(unique_id)
            
        self.initial_balance = self.get_account_balance()
        self.highest_balance = self.initial_balance
        logger.info(f"Initical balance {self.initial_balance}")

        seconds_to_wait = (self.start_time() - datetime.datetime.now() + datetime.timedelta(days=1)).total_seconds()
        Timer(seconds_to_wait, self.main).start()

        seconds_to_wait = (self.pnl_time() - datetime.datetime.now()).total_seconds()
        Timer(seconds_to_wait, self.get_pnl).start()


        if not self.isConnected():
            self.reconnect()
        
        logger.info('unique_id',unique_id)
        self.market_data[unique_id] = PriceInformation(self.contract)
        
        if self.is_paper:
            self.reqMarketDataType(3)  # Delayed data
        else:
            self.reqMarketDataType(1)  # Live data
        
        
        self.reqMktData(unique_id, self.contract, "", False, False, [])
        self.loop(unique_id)
        self.cancelMktData(unique_id)
    
    def update_smart_interval(self):
        
        if self.smart_interval == 'auto':
            if self.interval in ['1m','10m', '5m','15m']:
                self.smart_duration_str = "4 D"  
                self.new_barsize = "1 hour"
            else:
                self.smart_duration_str = "40 D"  # Duración de los datos históricos
                self.new_barsize = "1 day"
        
        elif self.smart_interval == '5m':
            if self.interval in ['1m']:
                self.smart_duration_str = "1 D"  
                self.new_barsize = "5 mins"
         
        elif self.smart_interval == '10m':
            if self.interval in ['1m','5m']:
                self.smart_duration_str = "2 D"  
                self.new_barsize = "10 mins"
                
        elif self.smart_interval == '15m':
            if self.interval in ['1m', '5m', '10m']:
                self.smart_duration_str = "3 D"  
                self.new_barsize = "15 mins"
    
    def trend_styudy(self, req_id):
        data = self.get_study_data(req_id)
        # Calcular indicadores necesarios
        data['EMA_11'] = ta.trend.ema_indicator(data['Close'], window=11)
        data['EMA_55'] = ta.trend.ema_indicator(data['Close'], window=55)

        # Calcular el MACD y la señal
        data['MACD'] =  data['EMA_11'] -  data['EMA_55']
        data['Signal'] =  data['MACD'].ewm(span=11, adjust=False).mean()
        
        adx = ta.trend.ADXIndicator(data['High'],data['Close'],data['Low'], window=14, fillna=True)
        data['ADX'] = adx.adx()
        
        if ((data['EMA_11'][-1] > data['EMA_55'][-1]) and 
            (data['MACD'][-1] > data['Signal'][-1]) and 
            (data['ADX'][-1] > 23)):
            self.accept_trade = 'long'
        elif ((data['EMA_11'][-1] < data['EMA_55'][-1]) and 
              (data['MACD'][-1] < data['Signal'][-1]) and 
              (data['ADX'][-1] > 23)):
            self.accept_trade = 'short'
        else:
            self.accept_trade = 'Range'
                
    def request_firts_position(self):
        self.positions1[self.account][self.symbol] = {
                        "position": 0,
                        "averageCost": 0
                    }
        self.reqPositions()
        time.sleep(1)

        logger.info(f'**** POSITION {self.positions1[self.account][self.symbol]["position"]}')
        self.cant_cont = self.positions1[self.account][self.symbol]["position"]


        if self.cant_cont_init is None:
            # Aumentamos la cantidad de contratos adquiridos
            self.cant_cont_init = self.cant_cont
        else:
            if (self.cant_cont > 0) and (self.cont_1 is not None):
                self.cant_cont_init = self.cant_cont - self.cont_1
            elif (self.cant_cont < 0) and (self.cont_1 is not None):
                self.cant_cont_init = self.cant_cont + self.cont_1
            else:
                self.cant_cont_init = self.cant_cont
        
        logger.info(f'#-----------------------> POSICION INICIAL {self.cant_cont_init}')
    
    def loop(self, req_id):

        if self.fdata is None:
            self.get_data(req_id)
            logger.info(f'Se descargó data historica, tamaño {self.fdata.shape}')
        
        if self.redondear_marca_de_tiempo(str(datetime.datetime.now())) in list(self.fdata.index):    
            self.fdata = self.fdata[~self.fdata.index.duplicated(keep=False)]
        
        if 'Short_Exit' not in self.fdata.columns:
            self.fdata['Short_Exit'] = 0
            self.fdata['Open_position'] = 0
            self.fdata['Close_real_price'] = 0
            logger.info('Se agregaron indicadores Short_Exit y Open_position')
        # self.fdata['Retornos'] = self.fdata['Close'].pct_change()

        # # Calcular la volatilidad histórica (en este caso, se asume 252 días laborables en un año)
        # self.volatilidad = self.fdata['Retornos'].std() * (252**0.5)
        # self.volatilidad = self.volatilidad/2

        if 'Open_position' not in self.fdata.columns:
            self.fdata['Open_position'] = 0
            logger.info('Se agregao indicador Open_position')

        self.request_firts_position() 
        #self.position_gestion()
        
        while True:
            
            while not self.isConnected():
                logger.info("ESPERANDO CONEXION")
                self.reconnect()

            self.estrategy_jemir()
            self.plot_strategy_jemir()
            self.html_generate() 
            logger.info(f'-*-**-*-*-*-*-*-*- DATA {self.fdata.tail(5)}')
            
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
                logger.info(f'Esperando {tiempo_de_espera} segundos para agregar precios')
                for i in tqdm(range(tiempo_de_espera)):
                    time.sleep(1)
                    try:
                        if self.is_paper:
                            price = (self.market_data[req_id].DelayedBid + self.market_data[req_id].DelayedAsk) / 2
                            market_price = self.market_data[req_id].DelayedAsk
                            vol = self.market_data[req_id].DelayedVolume
                        else:
                            logger.info('LIVE DATA')
                            price = (self.market_data[req_id].Bid + self.market_data[req_id].Ask) / 2
                            market_price = self.market_data[req_id].Ask 
                            vol = self.market_data[req_id].NotDefined
                        logger.info(f'PRECIO ------------> ${price}')
                        logger.info(f'PRECIO DE MERCADO--> ${market_price}')
                    except TypeError:
                        logger.info(f"Error TypeError, the price is None")
                    
                    if not self.open_position:
                        if (price > 0):
                                break

                    if self.open_position:
                        if (
                            price < self.open_trade_price
                            ):
                            break

                        elif (price > self.open_trade_price):
                            break
                                
                new_price_info = self.get_data_today(req_id)
                new_price_info['Short_Exit'] = 0
                new_price_info['Open_position'] = 0
                new_price_info['Close_real_price'] = 0
                if len(new_price_info) > 0:
                    
                    self.fdata = pd.concat([self.fdata,new_price_info])
                    
                    self.fdata = self.fdata[~self.fdata.index.duplicated(keep='last')]
                    #new_price_info.to_csv("prueba_de_data.csv")
                    
                    
                    self.estrategy_jemir()

                    logger.info('CALCULO DE MÉTRICAS ')
                    logger.info(f'Ultima Vela {self.fdata[-1:].T}')
                    # if self.time_to_wait >= self.bar_size * 2:
                    if price > 0:  
                        self.strategy_metrics_jemir(price, req_id)
                        
                    self.request_firts_position()
                    logger.info('ANALIZANDO ESTRATEGIA')            
                        
            except Exception as e:
                logger.info(f'{e}')

    def time_to_call_down(self, req_id):
        self.estrategy_jemir()
        self.plot_strategy_jemir()
        self.html_generate() 
                
        time_ = 0
        while time_ < self.call_down_wait:
            hora_actual = time.localtime()
            time_part_space = 3600/self.bar_size
            segundos_actuales = (hora_actual.tm_min * 60) + hora_actual.tm_sec
            parte_decimal, _ = math.modf((3600 - segundos_actuales) / self.bar_size)
            segundos_hasta_siguiente_tiempo = int(parte_decimal * self.bar_size)     
            
            if segundos_hasta_siguiente_tiempo > 0:
                tiempo_de_espera = min(self.bar_size, segundos_hasta_siguiente_tiempo)
            else:
                tiempo_de_espera = self.bar_size
            for i in tqdm(range(tiempo_de_espera), desc=f"CONSTRUYENDO VELAS EN EL CALL DOWN"):
                time.sleep(1)
                try:
                    if self.is_paper:
                        price = (self.market_data[req_id].DelayedBid + self.market_data[req_id].DelayedAsk) / 2
                        market_price = self.market_data[req_id].DelayedAsk
                        vol = self.market_data[req_id].DelayedVolume
                    else:
                        logger.info('LIVE DATA')
                        price = (self.market_data[req_id].Bid + self.market_data[req_id].Ask) / 2
                        market_price = self.market_data[req_id].Ask 
                        vol = self.market_data[req_id].NotDefined
                    logger.info(f'PRECIO ------------> ${price}')
                    logger.info(f'PRECIO DE MERCADO--> ${market_price}')
                except TypeError:
                    logger.info(f"Error TypeError, the price is None")
            time_ += tiempo_de_espera
            logger.info(f'CALL DOWN ACTIVADO ----> {time_}/{self.call_down_wait}')
           
            new_price_info = self.get_data_today(req_id)
            new_price_info['Short_Exit'] = 0
            new_price_info['Open_position'] = 0
            new_price_info['Close_real_price'] = 0
            if len(new_price_info) > 0:
                
                self.fdata = pd.concat([self.fdata,new_price_info])
                    
                self.fdata = self.fdata[~self.fdata.index.duplicated(keep='last')]
               
                self.estrategy_jemir()
                self.plot_strategy_jemir()
                self.html_generate()

        self.out_position = False
    
    def round_to_tick(self, price):
        return round(price / self.tick_value) * self.tick_value

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
        self.positions1[self.account][self.symbol] = {
                        "position": 0,
                        "averageCost": 0
                    }
        self.reqPositions()
        time.sleep(3)

        logger.info(f'**** POSITION {self.positions1[self.account][self.symbol]["position"]}')
        self.cant_cont = self.positions1[self.account][self.symbol]["position"]
    
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
            duration_str = "3 D"  # Duración de los datos históricos
        elif self.interval in ['2m', '5m','10m', '15m', '30m', '1h']:
            duration_str = "7 D"  # Duración de los datos históricos  # Duración de los datos históricos
    
        elif self.interval in ['4h']:
            duration_str = "10 D"  # Duración de los datos históricos

        self.historical_market_data[req_id] = self.get_historical_market_data(self.contract, duration_str, self.barsize)
        # logger.info(self.historical_market_data[req_id])
        bar_data_dict_list = [
                                {"Date": data.date, "Open": data.open, "High": data.high, "Low": data.low, "Close": data.close, "Volume": data.volume}
                                for data in self.historical_market_data[req_id]
                            ]
        df = pd.DataFrame(bar_data_dict_list, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        
        df["Date"] = df["Date"].apply(convert_date_time)
        logger.info(f'FECHA MINIMA DE DESCARGA IBAPI *** {df["Date"].min()}')
        logger.info(f'FECHA MAXIMA DE DESCARGA IBAPI *** {df["Date"].max()}')
        df.set_index("Date", inplace=True)

        self.fdata = df
        self.fdata.index = pd.to_datetime(self.fdata.index, format='%Y-%m-%d %H:%M:%S', utc=True)

        
        # Set 'Date' column as the index
        self.fdata = self.fdata[~self.fdata.index.duplicated(keep='last')]
        self.fdata = self.fdata[self.fdata['Close']>0].sort_index()
        
        #logger.info(self.historical_market_data[req_id])
    
    def get_study_data(self, req_id):
        self.update_smart_interval()
        historical_market_data = self.get_historical_market_data(self.contract, self.smart_duration_str, self.new_barsize)
        # logger.info(self.historical_market_data[req_id])
        bar_data_dict_list = [
                                {"Date": data.date, "Open": data.open, "High": data.high, "Low": data.low, "Close": data.close, "Volume": data.volume}
                                for data in historical_market_data
                            ]
        df = pd.DataFrame(bar_data_dict_list, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        #print(df)
        #df.to_csv('data_ib.csv')
        if self.new_barsize in ['5 mins','10 mins','15 mins','1 hour']:
            df["Date"] = df["Date"].apply(convert_date_time)
        else:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
            
        # df["Date"] = pd.to_datetime(df["Date"],  format='%Y-%m-%d %H:%M:%S')
        logger.info(f'FECHA MINIMA DE DESCARGA IBAPI *** {df["Date"].min()}')
        logger.info(f'FECHA MAXIMA DE DESCARGA IBAPI *** {df["Date"].max()}')
        df.set_index("Date", inplace=True)
        try:
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)
        except:
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d', utc=True)
        
        return df
    
    
    def get_data_today(self, req_id):
                    
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
        
        
        duration_str = "1 D"  # Duración de los datos históricos
        
        self.historical_market_data[req_id] = self.get_historical_market_data(self.contract, duration_str, self.barsize)
        # logger.info(self.historical_market_data[req_id])
        bar_data_dict_list = [
                                {"Date": data.date, "Open": data.open, "High": data.high, "Low": data.low, "Close": data.close, "Volume": data.volume}
                                for data in self.historical_market_data[req_id]
                            ]
        df = pd.DataFrame(bar_data_dict_list, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        #df.to_csv('data_ib.csv')
        df["Date"] = df["Date"].apply(convert_date_time)
        # df["Date"] = pd.to_datetime(df["Date"],  format='%Y-%m-%d %H:%M:%S')
        logger.info(f'FECHA MINIMA DE DESCARGA IBAPI *** {df["Date"].min()}')
        logger.info(f'FECHA MAXIMA DE DESCARGA IBAPI *** {df["Date"].max()}')
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)
        
        return df
        
        #logger.info(self.historical_market_data[req_id])

    def check_trailing_stop_activate(self,last_row):
        
        if ( (self.accept_trade == 'long') and
            (self.fdata['+DI'][last_row.index[0]] >= 23) and
            (self.fdata['+DI'][last_row.index[0]] < 40) 
            ): 
            self.trailing_stop_activate_di = True
            logger.info('TRAILING STOP ACTIVADO +DI')

        if ( (self.accept_trade == 'long') and
            (self.current_price >= (self.open_trade_price + self.tick_price + self.tick_value))
            ): 
            self.trailing_stop_activate_price = True
            logger.info('TRAILING STOP ACTIVADO PRICE')

        if ( (self.accept_trade == 'long') and
            (self.fdata['+DI'][last_row.index[0]] > 40) and
            (self.current_price > self.open_trade_price)
            ): 
            self.trailing_stop_activate_di_40 = True
            logger.info('TRAILING STOP ACTIVADO +DI 40')
        
        if ( (self.accept_trade == 'long') and
            (self.current_price > (self.open_trade_price + (self.DTC * 1.5)))
            ): 
            self.trailing_stop_DTC = True
            logger.info('TRAILING STOP ACTIVADO DTC')

        if ( (self.accept_trade == 'short') and
            (self.fdata['-DI'][last_row.index[0]] >= 23) and
            (self.fdata['-DI'][last_row.index[0]] < 40) 
            ): 
            self.trailing_stop_activate_di = True
            logger.info('TRAILING STOP ACTIVADO -DI')

        if ( (self.accept_trade == 'short') and
            (self.current_price <= (self.open_trade_price - self.tick_price - self.tick_value))
            ): 
            self.trailing_stop_activate_price = True
            logger.info(f'TRAILING STOP ACTIVADO PRICE EN {self.current_price}')
        
        if ( (self.accept_trade == 'short') and
            (self.fdata['-DI'][last_row.index[0]] > 40) and
            (self.current_price < self.open_trade_price)
            ): 
            self.trailing_stop_activate_di_40 = True
            logger.info('TRAILING STOP ACTIVADO -DI 40')
        
        if ( (self.accept_trade == 'short') and
            (self.current_price < (self.open_trade_price - (self.DTC * 1.5)))
            ): 
            self.trailing_stop_DTC = True
            logger.info('TRAILING STOP ACTIVADO DTC')

    def close_position_method(self, last_row, req_id):
        if self.accept_trade == 'long':
            self.sell(self.current_price, float(self.quantity), "LIMIT")
                    
            self.update_positions()
            
                
            if self.cant_cont != self.cant_cont_init:
                
                tiempo_inicio = time.time()

                for i in tqdm(range(self.wait), desc="Espera, ejecución orden de venta"):
                    
                    self.request_firts_position()
                    self.update_positions()

                    # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                    if self.cant_cont == self.cant_cont_init:
                        self.open_position = False
                        break

                    # Verificar si ha transcurrido el tiempo límite
                    tiempo_transcurrido = time.time() - tiempo_inicio
                    if tiempo_transcurrido >= self.wait:
                        # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                        self.cancelOrder(self.order_id, manualCancelOrderTime="")
                        self.open_position = True
                        self.trailing_stop_activate_di = False
                        self.trailing_stop_activate_price = False
                        self.trailing_stop_activate_di_40 = False
                        self.trailing_stop_DTC = False
                        break 
            
            
            if self.cant_cont == self.cant_cont_init:
                self.cant_cont_init = None
                self.cont_1 = None
                self.request_firts_position()
                self.open_position = False
                self.fdata.loc[last_row.index[0], 'Short_Exit'] = 1
                self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                self.trades_info['action'].append('Sell')
                self.trades_info['time'].append(last_row.index[0])
                self.trades_info['price'].append(self.round_to_tick(self.current_price))
                self.trades_info['contracts'].append(float(self.quantity))
                self.trades_info['Short_Exit'].append(1)
                self.trades_info['Open_position'].append(0)
                store_action(self.account,
                            self.strategy_name,
                            self.interval,
                            self.symbol,
                            self.accept_trade,
                            self.trade_id, 
                            "Sell", 
                            str(last_row.index[0]), 
                            self.round_to_tick(self.current_price), 
                            float(self.quantity),
                            short_exit=1, 
                            open_position=0)
                self.out_position = True
                self.trailing_stop_activate_di = False
                self.trailing_stop_activate_price = False
                self.trailing_stop_activate_di_40 = False
                self.trailing_stop_DTC = False
                self.trade_id = generate_random_id()
                self.time_to_call_down(req_id)
                #time.sleep(self.wait)
        else:
            self.buy(self.current_price, float(self.quantity), "LIMIT")

            self.update_positions()

            if self.cant_cont != self.cant_cont_init:
                    
                tiempo_inicio = time.time()

                for i in tqdm(range(self.wait), desc="Espera, ejecución orden de venta"):
                    
                    self.request_firts_position()
                    self.update_positions()

                    # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                    if self.cant_cont == self.cant_cont_init:
                        self.open_position = False
                        break

                    # Verificar si ha transcurrido el tiempo límite
                    tiempo_transcurrido = time.time() - tiempo_inicio
                    if tiempo_transcurrido >= self.wait:
                        # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                        self.cancelOrder(self.order_id, manualCancelOrderTime="")
                        self.open_position = True
                        self.trailing_stop_activate_di = False
                        self.trailing_stop_activate_price = False
                        self.trailing_stop_activate_di_40 = False
                        self.trailing_stop_DTC = False
                        break 

            if self.cant_cont == self.cant_cont_init:
                self.cant_cont_init = None
                self.cont_1 = None
                self.request_firts_position()
                self.open_position = False
                self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                self.trades_info['action'].append('Buy')
                self.trades_info['time'].append(last_row.index[0])
                self.trades_info['price'].append(self.round_to_tick(self.current_price))
                self.trades_info['contracts'].append(float(self.quantity))
                self.trades_info['Short_Exit'].append(0)
                self.trades_info['Open_position'].append(1)
                store_action(self.account,
                            self.strategy_name,
                            self.interval,
                            self.symbol,
                            self.accept_trade,
                            self.trade_id, 
                            "Buy", 
                            str(last_row.index[0]), 
                            self.round_to_tick(self.current_price), 
                            float(self.quantity),
                            short_exit=0, 
                            open_position=1)
                self.out_position = True
                self.trailing_stop_activate_di = False
                self.trailing_stop_activate_price = False
                self.trailing_stop_activate_di_40 = False
                self.trailing_stop_DTC = False
                self.trade_id = generate_random_id()
                self.time_to_call_down(req_id)

    def execute_trailing_stop(self, last_row, req_id):
        if  (self.accept_trade == 'long'): 
            if ((self.trailing_stop_activate_di) and
                ((self.fdata['+DI'][last_row.index[0]] <= self.di_plus)) #and 
                #(self.adx_slope < 0)
                ):
                logger.info(f'*---------------- ACTIVADO TRALING STOP DE +DI EN {self.di_plus}')
                self.close_position_method(last_row, req_id)

            if ((self.trailing_stop_activate_price) and
                (self.current_price <= (self.open_trade_price + self.tick_price))
                ):
                logger.info(f'*---------------- ACTIVADO TRALING STOP PRECIO {(self.open_trade_price * (1+self.volatility))+0.25}')
                self.close_position_method(last_row, req_id)
            
            if ((self.trailing_stop_activate_di_40) and
                (self.fdata['+DI'][last_row.index[0]] <= 40) and 
                (self.current_price > self.open_trade_price) 
                ):
                logger.info('*---------------- ACTIVADO TRALING STOP DE +DI EN 40')
                self.close_position_method(last_row, req_id)
            
            if ((self.trailing_stop_DTC) and
                (self.current_price <= self.open_trade_price + self.DTC) 
                ):
                logger.info('*---------------- ACTIVADO TRALING STOP DTC')
                self.close_position_method(last_row, req_id)

        elif (self.accept_trade == 'short') :
            if ((self.trailing_stop_activate_di) and 
                ((self.fdata['-DI'][last_row.index[0]] <= self.di_minus)) #and 
                #(self.adx_slope < 0)
                ):
                logger.info(f'*---------------- ACTIVADO TRALING STOP DE -DI EN {self.di_minus}')
                self.close_position_method(last_row, req_id)

            if ((self.trailing_stop_activate_price) and 
                (self.current_price >= (self.open_trade_price - self.tick_price ))
                ):
                logger.info(f'*---------------- ACTIVADO TRALING STOP PRECIO {(self.open_trade_price * (1-self.volatility))-0.25}')
                self.close_position_method(last_row, req_id)

            if ((self.trailing_stop_activate_di_40) and 
                ((self.fdata['-DI'][last_row.index[0]] <= 40)) and 
                (self.current_price < self.open_trade_price)
                ):
                logger.info('*---------------- ACTIVADO TRALING STOP DE -DI EN 40')
                self.close_position_method(last_row, req_id)
            
            if ((self.trailing_stop_DTC) and
                (self.current_price >= self.open_trade_price - self.DTC) 
                ):
                logger.info('*---------------- ACTIVADO TRALING STOP DTC')
                self.close_position_method(last_row, req_id)

            




    def estrategy_jemir(self):
        self.fdata = self.fdata[self.fdata['Close']>0]
        ##################################################
        # Estrategia
        #################################################
        self.daily_returns = self.fdata['Close'].pct_change().dropna()  # Calcular los retornos diarios
        self.volatility_sd = self.daily_returns.std() 
        self.volatility = round(self.volatility_sd * 6, 4)

        # Calcular indicadores necesarios
        self.fdata['EMA_9'] = ta.trend.ema_indicator(self.fdata['Close'], window=9)
        self.fdata['EMA_21'] = ta.trend.ema_indicator(self.fdata['Close'], window=21)
        self.fdata['EMA_11'] = ta.trend.ema_indicator(self.fdata['Close'], window=11)
        self.fdata['EMA_55'] = ta.trend.ema_indicator(self.fdata['Close'], window=55)
        
        self.fdata['High-Low'] = self.fdata['High'] - self.fdata['Low']
        self.fdata['High-PreviousClose'] = abs(self.fdata['High'] - self.fdata['Close'].shift(1))
        self.fdata['Low-PreviousClose'] = abs(self.fdata['Low'] - self.fdata['Close'].shift(1))
        self.fdata['TR'] = self.fdata[['High-Low', 'High-PreviousClose', 'Low-PreviousClose']].max(axis=1)

        # Calcular el Positive Directional Movement (+DM) y Negative Directional Movement (-DM)
        self.fdata['UpMove'] = self.fdata['High'] - self.fdata['High'].shift(1)
        self.fdata['DownMove'] = self.fdata['Low'].shift(1) - self.fdata['Low']
        self.fdata['+DM'] = self.fdata['UpMove'].where((self.fdata['UpMove'] > self.fdata['DownMove']) & (self.fdata['UpMove'] > 0), 0)
        self.fdata['-DM'] = self.fdata['DownMove'].where((self.fdata['DownMove'] > self.fdata['UpMove']) & (self.fdata['DownMove'] > 0), 0)

        # Calcular el True Positive (+DI) y True Negative (-DI)
        window = 14
        self.fdata['TR14'] = self.fdata['TR'].rolling(window).sum()
        self.fdata['+DM14'] = self.fdata['+DM'].rolling(window).sum()
        self.fdata['-DM14'] = self.fdata['-DM'].rolling(window).sum()
        # self.fdata['+DI'] = (self.fdata['+DM14'] / self.fdata['TR14']) * 100
        # self.fdata['-DI'] = (self.fdata['-DM14'] / self.fdata['TR14']) * 100

        # Inicializar el indicador ADX
        adx_indicator = ADXIndicator(high=self.fdata['High'], low=self.fdata['Low'], close=self.fdata['Close'], window=14)

        # Calcular +DI y -DI
        self.fdata['+DI'] = adx_indicator.adx_pos()
        self.fdata['-DI'] = adx_indicator.adx_neg()
        
        # Calcular el ADX
        adx = ta.trend.ADXIndicator(self.fdata['High'],self.fdata['Close'],self.fdata['Low'], window=14, fillna=True)
        self.fdata[f'ADX'] = adx.adx()

        # Calcular el MACD y la señal
        self.fdata['MACD'] =  self.fdata['EMA_11'] -  self.fdata['EMA_55']
        self.fdata['Signal'] =  self.fdata['MACD'].ewm(span=34, adjust=False).mean()

        today = datetime.datetime.now().date()
        
        self.macd_min = self.fdata[self.fdata.index >= str(today)]["MACD"].min()
        self.macd_max = self.fdata[self.fdata.index >= str(today)]["MACD"].max()
        
        # Calcular Bollinger Bands
        self.fdata['Bollinger_Upper'] = ta.volatility.bollinger_hband(self.fdata['Close'], window=20)
        self.fdata['Bollinger_Lower'] = ta.volatility.bollinger_lband(self.fdata['Close'], window=20)
        
        # Calcular Keltner Channels (usando ATR)
        self.fdata['ATR'] = ta.volatility.average_true_range(self.fdata['High'], self.fdata['Low'], self.fdata['Close'], window=20)
        self.fdata['Keltner_Upper'] = self.fdata['Bollinger_Upper'] + 1.5 * self.fdata['ATR']
        self.fdata['Keltner_Lower'] = self.fdata['Bollinger_Lower'] - 1.5 * self.fdata['ATR']
        
        # Calcular el Squeeze Momentum Indicator (SMI)
        self.fdata['SMI'] = 100 * (self.fdata['Bollinger_Upper'] - self.fdata['Bollinger_Lower']) / self.fdata['Keltner_Upper']
        
        # Calcular estadísticas del SMI para definir umbrales
        smi_mean = self.fdata['SMI'].mean()
        smi_min = self.fdata['SMI'].min()
        smi_max = self.fdata['SMI'].max()
        
        # Definir umbrales en función de las estadísticas del SMI
        threshold_force_bearish = smi_mean - (smi_max - smi_mean) * 0.25
        threshold_momentum_bullish = smi_mean + (smi_max - smi_mean) * 0.1
        threshold_force_bullish = smi_mean + (smi_max - smi_mean) * 0.25
        threshold_momentum_bearish = smi_mean - (smi_max - smi_mean) * 0.1
        
        # Determinar fases en función de los umbrales
        self.fdata['Squeez_Momentum_Phase'] = 'No Phase'
        self.fdata.loc[self.fdata['SMI'] < threshold_force_bearish, 'Squeez_Momentum_Phase'] = 'Impulso Bajista'
        self.fdata.loc[(self.fdata['SMI'] >= threshold_force_bearish) & (self.fdata['SMI'] < threshold_momentum_bearish), 'Squeez_Momentum_Phase'] = 'Fuerza Bajista'
        self.fdata.loc[(self.fdata['SMI'] >= threshold_force_bullish) & (self.fdata['SMI'] < threshold_momentum_bullish), 'Squeez_Momentum_Phase'] = 'Fuerza Alcista'
        self.fdata.loc[(self.fdata['SMI'] >= threshold_momentum_bearish) & (self.fdata['SMI'] <= threshold_momentum_bullish), 'Squeez_Momentum_Phase'] = 'Impulso Alcista'


        # Obtén los dos últimos valores del ADX
        self.last_two_adx_values = self.fdata['ADX'][-2:]

        # Calcula la diferencia entre los dos últimos valores del ADX
        self.adx_slope = self.last_two_adx_values[1] - self.last_two_adx_values[0]
        
        

        self.di_plus = round(self.fdata['+DI'].describe()['75%'], 2)
        self.di_minus = round(self.fdata['-DI'].describe()['75%'], 2)

        if (self.fdata['-DI'][-2:][1] > self.di_minus) and (self.fdata['-DI'][-2:][0] > self.di_minus):
            self.di_minus_trend = self.fdata['-DI'][-2:][1] - self.fdata['-DI'][-2:][0]
        else:
            self.di_minus_trend = -1
        
        if (self.fdata['+DI'][-2:][1] > self.di_plus) and (self.fdata['+DI'][-2:][0] > self.di_plus):            
            self.di_plus_trend = self.fdata['+DI'][-2:][1] - self.fdata['+DI'][-2:][0]
        else:
            self.di_plus_trend = -1
        
        
        # Definir umbrales y condiciones
        if self.accept_trade == 'long':
            self.mid_point = round(self.fdata[(self.fdata['+DI'] >= self.di_plus) & (self.fdata['+DI'] <= 40)]['+DI'].describe()['75%'], 3)
        else:
            self.mid_point = round(self.fdata[(self.fdata['-DI'] >= self.di_minus) & (self.fdata['-DI'] <= 40)]['-DI'].describe()['75%'], 3)

        # Creamos una nueva columna
        # data1['Entry'] = 0.0
        self.fdata['Long_Signal'] = np.where((self.fdata['EMA_9'] >= self.fdata['EMA_21']) &
                                              (self.fdata['ADX'] >= self.di_plus) &
                                              (self.fdata['ADX'] < 40) &
                                              (self.fdata['+DI'] <= self.mid_point) &
                                              (self.fdata['+DI'] >= self.di_plus) &
                                              (self.fdata['+DI'] > self.fdata['-DI']) &
                                              (self.adx_slope > 0) &
                                              (self.di_plus_trend > 0)
                                              , 1.0, 0.0)
        self.fdata['Short_Signal'] = np.where((self.fdata['EMA_21'] >= self.fdata['EMA_9']) &
                                               (self.fdata['ADX'] >= self.di_minus) &
                                               (self.fdata['ADX'] < 40) &
                                               (self.fdata['-DI'] <= self.mid_point) &
                                               (self.fdata['-DI'] >= self.di_minus) &
                                               (self.fdata['-DI'] > self.fdata['+DI']) &
                                              (self.adx_slope > 0) &
                                              (self.di_minus_trend > 0)
                                               , 1.0, 0.0)
        self.fdata['Posicion'] = self.fdata['Long_Signal'].diff()
        
    def cal_1(self):
        self.cont_1 = int(self.quantity)

    def strategy_metrics_jemir(self, price, req_id):
        
        # Iterar a través de los datos para simular la estrategia
        last_row = self.fdata[-1:]
        logger.info(f' * * * * * * Hora de actualización: {last_row.index[0]}')

        self.current_price = price
        
        if (self.with_trend_study) and (not self.open_position):
            self.trend_styudy(req_id)
        
        if self.accept_trade == "short":
            logger.info('****************** ESTAS EN SHORT')
            logger.info(f'SHORT *************** {self.fdata["Short_Signal"][last_row.index[0]]}')

            if (self.out_position) and (self.fdata['-DI'][last_row.index[0]] <= self.di_minus):
                self.out_position = False

            if self.with_macd_signal:
                self.short_entry_signal = ((not self.open_position) and 
                                           (not self.out_position) and 
                                           (self.fdata['Short_Signal'][last_row.index[0]] == 1) and 
                                           self.fdata['MACD'][last_row.index[0]] > self.macd_min)
            else:
                self.short_entry_signal = ((not self.open_position) and 
                                           (not self.out_position) and 
                                           (self.fdata['Short_Signal'][last_row.index[0]] == 1))
            
            # Señal de venta
            if self.short_entry_signal: 
                
                self.open_trade_price = self.current_price
                self.sell(self.open_trade_price, float(self.quantity), "LIMIT")
                self.update_positions()
                self.cal_1()

                if self.cant_cont <= self.cant_cont_init +((-1)* self.cont_1):
                    self.request_firts_position()
                    self.open_position = True
                    self.fdata.loc[last_row.index[0], 'Short_Exit'] = 1
                    self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                    self.trades_info['action'].append('Sell')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.round_to_tick(self.open_trade_price))
                    self.trades_info['contracts'].append(float(self.quantity))
                    self.trades_info['Short_Exit'].append(1)
                    self.trades_info['Open_position'].append(0)
                    store_action(self.account,
                                self.strategy_name,
                                self.interval,
                                self.symbol,
                                self.accept_trade,
                                self.trade_id, 
                                "Sell", 
                                str(last_row.index[0]), 
                                self.round_to_tick(self.open_trade_price), 
                                float(self.quantity),
                                short_exit=1, 
                                open_position=0)
                else:
                    tiempo_inicio = time.time()

                    for i in tqdm(range(self.wait), desc="Espera, ejecución orden de entrada"):
                        #self.request_firts_position()
                        self.update_positions()

                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont <= self.cant_cont_init +((-1)* self.cont_1):
                            self.request_firts_position()
                            self.open_position = True
                            self.fdata.loc[last_row.index[0], 'Short_Exit'] = 1
                            self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                            self.trades_info['action'].append('Sell')
                            self.trades_info['time'].append(last_row.index[0])
                            self.trades_info['price'].append(self.round_to_tick(self.open_trade_price))
                            self.trades_info['contracts'].append(float(self.quantity))
                            self.trades_info['Short_Exit'].append(1)
                            self.trades_info['Open_position'].append(0)
                            store_action(self.account,
                                self.strategy_name,
                                self.interval,
                                self.symbol,
                                self.accept_trade,
                                self.trade_id, 
                                "Sell", 
                                str(last_row.index[0]), 
                                self.round_to_tick(self.open_trade_price), 
                                float(self.quantity),
                                short_exit=1, 
                                open_position=0)
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= self.wait:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.cancelOrder(self.order_id, manualCancelOrderTime="")
                            #self.get_data(req_id)
                            break
                
            if (self.open_position) and (not self.out_position):
                self.check_trailing_stop_activate(last_row)
                self.execute_trailing_stop(last_row, req_id)


            # Toma de ganancias
            if self.open_position and (self.fdata['EMA_9'][last_row.index[0]] > self.fdata['EMA_21'][last_row.index[0]]) and (not self.out_position):  

                self.buy(self.current_price, float(self.quantity), "LIMIT")
                    
                self.update_positions()
                
                if self.cant_cont != self.cant_cont_init:
                    
                    tiempo_inicio = time.time()

                    for i in tqdm(range(self.wait), desc="Espera, ejecución orden de venta"):
                        self.request_firts_position()
                        self.update_positions()

                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont == self.cant_cont_init:
                            self.open_position = False
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= self.wait:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.cancelOrder(self.order_id, manualCancelOrderTime="")
                            #self.get_data(req_id)
                            self.open_position = True
                            break 

                if self.cant_cont == self.cant_cont_init:
                    self.cant_cont_init = None
                    self.cont_1 = None
                    self.request_firts_position()
                    self.open_position = False
                    self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                    self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                    self.trades_info['action'].append('Buy')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.round_to_tick(self.current_price))
                    self.trades_info['contracts'].append(float(self.quantity))
                    self.trades_info['Short_Exit'].append(0)
                    self.trades_info['Open_position'].append(1)
                    store_action(self.account,
                                self.strategy_name,
                                self.interval,
                                self.symbol,
                                self.accept_trade,
                                self.trade_id, 
                                "Buy", 
                                str(last_row.index[0]), 
                                self.round_to_tick(self.current_price), 
                                float(self.quantity),
                                short_exit=0, 
                                open_position=1)
                    self.trailing_stop_activate_di = False
                    self.trailing_stop_activate_price = False
                    self.trailing_stop_activate_di_40 = False
                    self.trailing_stop_DTC = False
                    self.trade_id= generate_random_id()
                    self.time_to_call_down(req_id)
                    #time.sleep(self.wait)
        
        elif self.accept_trade == "long":
            logger.info('****************** ESTAS EN LONG')
            logger.info(f'LONG *************** {self.fdata["Long_Signal"][last_row.index[0]]}')

            if (self.out_position) and (self.fdata['+DI'][last_row.index[0]] <= self.di_plus):
                self.out_position = False

            if self.with_macd_signal:
                self.long_entry_signal = ((not self.open_position) and 
                                           (not self.out_position) and 
                                           (self.fdata['Long_Signal'][last_row.index[0]] == 1) and 
                                           self.fdata['MACD'][last_row.index[0]] < self.macd_max)
            else:
                self.long_entry_signal = ((not self.open_position) and 
                                           (not self.out_position) and 
                                           (self.fdata['Long_Signal'][last_row.index[0]] == 1))
            
            # Señal de compra
            if (not self.open_position) and (not self.out_position) and (self.fdata['Long_Signal'][last_row.index[0]] == 1): 
                self.open_trade_price = self.current_price
                self.buy(self.open_trade_price, float(self.quantity), "LIMIT")
                
                self.update_positions()
                self.cal_1()

                if self.cant_cont >= self.cant_cont_init + self.cont_1:
                    self.request_firts_position()
                    self.open_position = True
                    self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                    self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                    self.trades_info['action'].append('Buy')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.round_to_tick(self.open_trade_price))
                    self.trades_info['contracts'].append(float(self.quantity))
                    self.trades_info['Short_Exit'].append(0)
                    self.trades_info['Open_position'].append(1)
                    store_action(self.account,
                                self.strategy_name,
                                self.interval,
                                self.symbol,
                                self.accept_trade,
                                self.trade_id, 
                                "Buy", 
                                str(last_row.index[0]), 
                                self.round_to_tick(self.open_trade_price), 
                                float(self.quantity),
                                short_exit=0, 
                                open_position=1)
                else:
                    tiempo_inicio = time.time()

                    for i in tqdm(range(self.wait), desc="Espera, ejecución orden de entrada"):
                        #self.request_firts_position()
                        # Actualizar las posiciones y otras variables necesarias
                        self.update_positions()

                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont >= self.cant_cont_init + self.cont_1:
                            self.request_firts_position()
                            self.open_position = True
                            self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                            self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                            self.trades_info['action'].append('Buy')
                            self.trades_info['time'].append(last_row.index[0])
                            self.trades_info['price'].append(self.round_to_tick(self.open_trade_price))
                            self.trades_info['contracts'].append(float(self.quantity))
                            self.trades_info['Short_Exit'].append(0)
                            self.trades_info['Open_position'].append(1)
                            store_action(self.account,
                                self.strategy_name,
                                self.interval,
                                self.symbol,
                                self.accept_trade,
                                self.trade_id, 
                                "Buy", 
                                str(last_row.index[0]), 
                                self.round_to_tick(self.open_trade_price), 
                                float(self.quantity),
                                short_exit=0, 
                                open_position=1)
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= self.wait:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.cancelOrder(self.order_id, manualCancelOrderTime="")
                            #self.get_data(req_id)
                            break

            if self.open_position and (not self.out_position):
                self.check_trailing_stop_activate(last_row)
                self.execute_trailing_stop(last_row, req_id)


            # Señal de venta    
            if self.open_position and (not self.out_position) and (self.fdata['EMA_9'][last_row.index[0]] < self.fdata['EMA_21'][last_row.index[0]]):  
                self.sell(self.current_price, float(self.quantity), "LIMIT")
                    
                self.update_positions()
                
                
                if self.cant_cont != self.cant_cont_init:
                    
                    tiempo_inicio = time.time()

                    for i in tqdm(range(self.wait), desc="Espera, ejecución orden de venta"):
                        self.request_firts_position()
                        self.update_positions()
                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont == self.cant_cont_init:
                            self.open_position = False
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= self.wait:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.cancelOrder(self.order_id, manualCancelOrderTime="")
                            #self.get_data(req_id)
                            self.open_position = True
                            break 
                 
                if self.cant_cont == self.cant_cont_init:
                    self.cant_cont_init = None
                    self.cont_1 = None
                    self.request_firts_position()
                    self.open_position = False
                    self.fdata.loc[last_row.index[0], 'Short_Exit'] = 1
                    self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                    self.trades_info['action'].append('Sell')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.round_to_tick(self.current_price))
                    self.trades_info['contracts'].append(float(self.quantity))
                    self.trades_info['Short_Exit'].append(1)
                    self.trades_info['Open_position'].append(0)
                    store_action(self.account,
                                self.strategy_name,
                                self.interval,
                                self.symbol,
                                self.accept_trade,
                                self.trade_id, 
                                "Sell", 
                                str(last_row.index[0]), 
                                self.round_to_tick(self.current_price), 
                                float(self.quantity),
                                short_exit=1, 
                                open_position=0)
                    self.trailing_stop_activate_di = False
                    self.trailing_stop_activate_price = False
                    self.trailing_stop_activate_di_40 = False
                    self.trailing_stop_DTC = False
                    self.trade_id = generate_random_id()
                    self.time_to_call_down(req_id)
                    #time.sleep(self.wait)
    
    def buy(self, price, cant, action):
        logger.info("=========================================================== Placing buy order: {}"
              .format(round(price, 5)))

        self.order_id = self.get_order_id()

        if action == "MARKET":
            order = self.market_order(self.action1, float(cant))
        elif action == "LIMIT":
            order = self.limit_order(self.action1, float(cant), self.min_price_increment(price), self.account)
        else:
            order = self.stop_limit_order(self.action1, float(cant), self.min_price_increment(price), self.account)
        
        self.placeOrder(self.order_id, self.contract, order)
        

    def sell(self, price, cant, action):
        logger.info("=========================================================== Placing sell order: {}"
              .format(round(price, 5)))

        self.order_id = self.get_order_id()
        if action == "MARKET":
            order = self.market_order(self.action2, float(cant))
        elif action == "LIMIT":
            order = self.limit_order(self.action2, float(cant), self.min_price_increment(price), self.account)
        else:
            order = self.stop_limit_order(self.action2, float(cant), self.min_price_increment(price), self.account)
 
        self.placeOrder(self.order_id, self.contract, order)
    
    def plot_strategy_jemir(self):
        
        self.df_activity = pd.DataFrame(self.trades_info)
        
        if self.with_macd_signal:
            self.figure=make_subplots( 
                            rows = 3,
                            cols=1,
                            shared_xaxes = True,
                            row_heights=[0.7, 0.3, 0.3],
                            vertical_spacing = 0.06,
            #specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"colspan": 2}, None]]
            )
        else:
            self.figure=make_subplots( 
                            rows = 2,
                            cols=1,
                            shared_xaxes = True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing = 0.06,
            #specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"colspan": 2}, None]]
            )
        
        #self.figure.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'}, xaxis_domain=[0, 0.94])

        self.figure.add_trace(go.Candlestick(
                        x = self.fdata.index,
                        open = self.fdata['Open'],
                        high = self.fdata['High'],
                        low = self.fdata['Low'],
                        close = self.fdata['Close'],
                        name=f'Precio de {self.symbol}' 
                        ),
                    col=1,
                    row=1
                    )
        
        # Agregar el gráfico de barras de volumen en la segunda columna (encima del gráfico de velas)
                       
        self.figure.add_trace(go.Scatter(
                x=self.df_activity[self.df_activity['Open_position'] == 1]['time'],
                y=self.df_activity[self.df_activity['Open_position'] == 1]['price'],
                mode= 'markers',
                name = 'Compra',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='star-triangle-up'
                ) ),
                col=1,
                row=1 )
        
               
        # Ploteando Señales de VENTA
        self.figure.add_trace(go.Scatter(
            x=self.df_activity[self.df_activity['Short_Exit'] == 1]['time'],
            y=self.df_activity[self.df_activity['Short_Exit'] == 1]['price'],
            mode= 'markers',
            name = 'Venta',
            marker=dict(
                size=15,
                color='cyan',
                symbol='star-triangle-down'
            )
                                ),
                col=1,
                row=1)
        
        self.figure.add_trace(
                            go.Scatter(
                            x= self.fdata.index, 
                            y=self.fdata['EMA_21'],
                            line = dict(color='orange',width=2),
                            name='EMA 21'
                            ),
                col=1,
                row=1)
        self.figure.add_trace(
                            go.Scatter(
                            x= self.fdata.index, 
                            y=self.fdata['EMA_9'],
                            line = dict(color='blue',width=2),
                            name='EMA 9'
                            ),
                col=1,
                row=1)
        
        self.figure.add_trace(go.Scatter(x=self.fdata.index, 
                                y=self.fdata[f'ADX'], 
                                mode='lines', 
                                name='ADX', 
                                line=dict(color='blue', dash='dash')
                                    ),
                                col=1,
                                row=2
                                )
        self.figure.add_trace(go.Scatter(x=self.fdata.index, 
                                    y=self.fdata['+DI'], 
                                    mode='lines', 
                                    name='+DI', 
                                    line=dict(color='green')
                                        ),
                                    col=1,
                                    row=2)
        self.figure.add_trace(go.Scatter(x=self.fdata.index, 
                                    y=self.fdata['-DI'], 
                                    mode='lines', 
                                    name='-DI', 
                                    line=dict(color='red')
                                        ),
                                    col=1,
                                    row=2)
        if self.with_macd_signal:
            self.figure.add_trace(go.Scatter(x=self.fdata.index, 
                                        y=self.fdata['MACD'], 
                                        mode='lines', 
                                        name='MACD', 
                                        line=dict(color='red')
                                            ),
                                        col=1,
                                        row=3)
            
            # Añadir líneas horizontales en 23 y 40
            self.figure.add_shape(type="line", 
                                x0=self.fdata.index[0], 
                                y0=self.macd_min, 
                                x1=self.fdata.index[-1], 
                                y1=self.macd_min, 
                                name='MACD MIN DAY',
                                line=dict(color="gray", dash="dash"),
                                col=1,
                                row=3)
            self.figure.add_shape(type="line", 
                                x0=self.fdata.index[0], 
                                y0=self.macd_max, 
                                x1=self.fdata.index[-1], 
                                y1=self.macd_max, 
                                name='MACD MAX DAY',
                                line=dict(color="gray", dash="dash"),
                                col=1,
                                row=3)
        
        # Añadir líneas horizontales en 23 y 40
        self.figure.add_shape(type="line", 
                            x0=self.fdata.index[0], 
                            y0=self.mid_point, 
                            x1=self.fdata.index[-1], 
                            y1=self.mid_point, 
                            line=dict(color="orange", dash="dash"),
                            col=1,
                            row=2)
        if self.accept_trade == 'long':
            self.figure.add_shape(type="line", 
                                x0=self.fdata.index[0], 
                                y0=self.di_plus, 
                                x1=self.fdata.index[-1], 
                                y1=self.di_plus, 
                                line=dict(color="gray", dash="dash"),
                                col=1,
                                row=2)
        else:
            self.figure.add_shape(type="line", 
                                x0=self.fdata.index[0], 
                                y0=self.di_minus, 
                                x1=self.fdata.index[-1], 
                                y1=self.di_minus, 
                                line=dict(color="gray", dash="dash"),
                                col=1,
                                row=2)
            
        self.figure.add_shape(type="line", 
                            x0=self.fdata.index[0], 
                            y0=40, 
                            x1=self.fdata.index[-1], 
                            y1=40, 
                            line=dict(color="gray", dash="dash"),
                            col=1,
                            row=2)

        #self.figure.data[1].update(xaxis='x2')
        self.figure.update_layout(xaxis_rangeslider_visible=False, hovermode='x unified')
        self.figure.update_layout(width=1500, height=1000)
        self.figure.update_layout(title=f"Estrategia aplicada a {self.symbol} en el intervalo {self.interval}")

    def html_generate(self):
        logger.info('GENERANDO EL HTML ********')
        self.plot_div = pyo.plot(self.figure, output_type='div', include_plotlyjs='cdn', image_width= 1200)
        with open('zenit_logo_dor.png', 'rb') as img_file:
            imagen_base64 = base64.b64encode(img_file.read()).decode('utf-8')
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
        if self.with_trend_study:
            file_name = f"bot_activity/TA_smart_{self.interval}_{self.symbol}_{self.hora_ejecucion}.html"
        else:
            file_name = f"bot_activity/TA_{self.accept_trade}_{self.interval}_{self.symbol}_{self.hora_ejecucion}.html"
            
        with open(file_name, "w") as html_file:
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
                        <img src="data:image/png;base64,{imagen_base64}" alt="Imagen" width='10%'>
                    </div>
                </header>
                <!-- Div donde se mostrará la gráfica Plotly -->
                <div id="mensaje"></div>
                <div id="plotly-div" style="width:100%" align="center">{self.plot_div}</div>
                <div>
                    <center>
                        <h1>Parametros de evaluación: <h1>
                        <h2> Trade Type: {self.accept_trade}</h2>
                        <h3> Smart layer interval: {self.new_barsize} </h3>
                        <h3>Trailing stop precio: {self.trailing_stop_activate_price}</h3>
                        <h3>Trailing stop DI: {self.trailing_stop_activate_di}</h3>
                        <h3>Trailing stop DI en 40: {self.trailing_stop_activate_di_40}</h3>
                        <h3>Trailing stop por DTC: {self.trailing_stop_DTC}</h3>
                        <h2>DTC USD: {self.DTC}</h2>
                        <h2> Operaciones realizadas por el Bot </h2>
                    
                        <ul>
                            {activity}
                        </ul>
                    </center>
                </div>
            </body>
            <script>
                // Función para actualizar la página cada segundo
                function actualizarPagina() {{
                    location.reload();
                }}

                // Función para mostrar el contador de tiempo
                function mostrarContador(tiempoRestante) {{
                    var mensaje = document.getElementById('mensaje');
                    mensaje.innerHTML = 'La página se actualizará en ' + tiempoRestante + ' segundos.';
                    
                    // Actualizar el contador cada segundo
                    setTimeout(function() {{
                        if (tiempoRestante > 0) {{
                            mostrarContador(tiempoRestante - 1);
                        }} else {{
                            // Cuando el contador llega a cero, actualizar la página
                            actualizarPagina();
                        }}
                    }}, 1000);
                }}

                // Obtener el tiempo total en segundos (60 segundos por defecto)
                var tiempoTotal = 120; // Tiempo en segundos

                // Iniciar el contador al cargar la página
                mostrarContador(tiempoTotal);
            </script>
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
    parser.add_argument('--accept_trade', type=str, default='long', help='Type of trades for trading')
    parser.add_argument('--multiplier', type=str, default="50", help='The multiplier for futures')
    parser.add_argument('--trading_class', type=str, default="ES", help='The trading_class for futures')
    parser.add_argument('--lastTradeDateOrContractMonth', type=str, default="20240920", help='The expire date for futures')
    parser.add_argument('--order_type', type=str, default="LIMIT", help='The type of the order: LIMIT OR MARKET')
    parser.add_argument('--order_validity', type=str, default="DAY", help='The expiration time of the order: DAY or GTC')
    parser.add_argument('--is_paper', type=str_to_bool, default=False, help='Paper or live trading')
    parser.add_argument('--hora_ejecucion', type=str, default=None, help='Time of execution bot')
    parser.add_argument('--with_trend_study', type=str_to_bool, default=False, help='Trend study for day trading')
    parser.add_argument('--with_macd_signal', type=str_to_bool, default=False, help='Trade area up the macd day minimum')
    parser.add_argument('--smart_interval', type=str, default='auto', help='Smart layer interval study')
    
    args = parser.parse_args()
    logger.info(f"args {args}")

    bot = BotZenitCRUCEEMAS(args.ip, 
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
              args.order_validity,
              args.hora_ejecucion,
              args.with_trend_study,
              args.with_macd_signal,
              args.smart_interval
              )
    try:
        bot.main()
    except KeyboardInterrupt:
        bot.disconnect()

