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
import math
import base64

futures_symbols_info = pd.read_csv("data/futures_symbols_v1.csv")

logger = logging.getLogger()
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO)

class BotZenitTrendMaster(Main):

    def __init__(self, 
                ip, 
                port, 
                client, 
                symbol, 
                secType, 
                currency, 
                exchange, 
                quantity,  
                stop_limit_ratio,    
                max_safety_orders, 
                safety_order_size, 
                volume_scale,
                safety_order_deviation,  
                account,
                take_profit_percent,
                interval,
                accept_trade,
                threshold_adx,
                multiplier,
                trading_class,
                lastTradeDateOrContractMonth,
                order_type="LIMIT", 
                order_validity="DAY",
                is_paper=True,
                quantity_type='fixed',
                hora_ejecucion=None,
                with_trend_study = False
                 ):
        Main.__init__(self, ip, port, client)
        self.action1 = "BUY"
        self.action2 = "SELL"
        self.ip = ip
        self.port = port
        self.with_trend_study = with_trend_study
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
        # self.take_profit_percent = take_profit_percent
        self.take_profit_percent = 0
        self.volatility = 0
        self.required_take_profit = 0

        #risk indicators
        self.trades_today = 0 # 
        self.today = datetime.datetime.now().date() # 
        self.open_positions = 0 # 

        self.order_type = order_type
        self.quantity_type = quantity_type
        #para el punto 9
        self.stop_limit_ratio = stop_limit_ratio
        self.trailing_stop = None
        self.break_even_triggered = False
        self.order_id = None
        self.strategy_name = 'TREND_MASTER'

        self.order_id_tp = None

        #para el punto 11
        self.max_safety_orders = max_safety_orders
        #self.max_safety_orders = 4
        self.safety_order_size = safety_order_size
        self.volume_scale = volume_scale
        self.safety_order_deviation = safety_order_deviation
        self.active_safety_orders = 0
        self.safety_order_1 = False
        self.safety_order_2 = False
        self.total_volume = 0
        self.average_purchase_price = 0
        if hora_ejecucion is None:
            self.hora_ejecucion = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.hora_ejecucion = hora_ejecucion
        if self.contract.secType == 'FUT':
            self.symbol = self.contract.tradingClass
        else:
            self.symbol = self.contract.symbol

        self.cont_1 = None
        self.cant_cont_init = None
        # Datos para la estrategia
        self.accept_trade = accept_trade
        self.orders_info = {
            "safety_orders": {
                    "so_1": self.volatility,
                    "so_2": self.volatility * 2,
                    "so_3": self.volatility * 3,
                    "so_4": self.volatility * 4,
            }
        }
        
        self.stop_loss_percent = self.volatility 
        # Finance data
        self.fdata = None
        self.volume_df = None
        self.poc_price = None
        self.dict_metrics = None
        self.profit_dict = {'price':[], 'profit':[], 'time':[]}
        self.loss_dict = {'price':[], 'loss':[], 'time':[]}
        self.trades_info = {'action':[], 
                            'time':[], 
                            'price':[], 
                            'contracts':[],
                            'Open_position':[], 
                            'Short_Exit':[]}
        # self.trades_info = {'action':[], 
        #                     'time':[], 
        #                     'price':[],
        #                     'contracts':[],
        #                     'trade_id':[],
        #                     'symbol':[],
        #                     'strategy':[]
        #                     }
        # Definir umbrales y condiciones
        self.threshold_adx = 23

        if self.quantity_type == 'fixed':
            self.quantity = quantity
            self.orders_info["contracts"] = {
                                            "so_1": 3,
                                            "so_2": 1,
                                            "so_3": 0,
                                            "so_4": 0,
                                            }
        else:
            self.quantity = str(round(int(quantity) * 0.4))
            self.orders_info["contracts"] = {
                                            "so_1": round(int(quantity)*0.6),
                                            "so_2": round(int(quantity)*0.6),
                                            "so_3": 0,
                                            "so_4": 0,
                                            }
            print(f'QUANTITY -------------> {self.quantity}')
            print(f'SAFETY ORDER 1 QUANTITY -> {self.orders_info["contracts"]["so_1"]}')
            print(f'SAFETY ORDER 2 QUANTITY -> {self.orders_info["contracts"]["so_2"]}')
        self.total_quantity = quantity
        self.order_n1 = self.orders_info['contracts']['so_1']
        self.order_n2 = self.orders_info['contracts']['so_2']
        self.order_n3 = self.orders_info['contracts']['so_3']
        self.order_n4 = self.orders_info['contracts']['so_4']
        self.orderSum = int(self.quantity) + self.order_n1 + self.order_n2 + self.order_n3 + self.order_n4
        # Position
        self.open_position = False

        # Capital inicial y variables de posición abierta
        self.open_trade_price = 0
        self.open_trade_price1 = 0
        self.open_trade_price2 = 0
        self.open_trade_price3 = 0
        self.open_trade_price4 = 0
        self.operations = 0
        self.successful_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.risk_free_rate = 0.02
        self.cant_cont = 0 # Cantidad de contratos adquiridos
        self.cont_ven = 0
        self.stoploss_activate = False

        # Candle graph
        self.colors = ['#00FF00', '#FF0000', '#FFFF00', '#0000FF', '#FFA500','#FF0000']
        self.figure = None

        self.order_validity = order_validity
        self.current_price = 0
        self.positions1 = {}
        self.plot_div = None
        self.trailing_stop_activate_di = False
        self.trailing_stop_activate_price = False
        self.trailing_stop_activate_di_40 = False
        self.di_plus = None
        self.di_minus = None
        
        self.mid_point = None
        self.out_position = False
        self.last_two_adx_values = None
        self.mid_point = None
        self.adx_slope = None
        self.wait_to_trade = 300
        
        self.df_activity = None
        # Tick information
        self.tick_usd_value = float(futures_symbols_info[futures_symbols_info['Symbol'] == self.symbol]['Value'].iloc[0])
        self.tick_value = futures_symbols_info[futures_symbols_info['Symbol'] == self.symbol]['Tick'].iloc[0]
        self.name = futures_symbols_info[futures_symbols_info['Symbol'] == self.symbol]['Name'].iloc[0]
        
        if 'Micro' in self.name:
            self.USD_max = 150 / (float(self.tick_usd_value)*10)
        else:
            self.USD_max = 150 / float(self.tick_usd_value)
        
        if self.interval == '1m':
            self.call_down_wait = 3600
        elif self.interval == '5m':
            self.call_down_wait = 7200
        elif self.interval == '15m':
            self.call_down_wait = 14400
        elif self.interval == '1h':
            self.call_down_wait = 21600
        elif self.interval == '4h':
            self.call_down_wait = 43200
        else:
            self.call_down_wait = 4000
        
        self.tick_price = round(self.tick_value * self.USD_max, 2)
        self.trade_id = generate_random_id()
        self.current_contracts = None

    def main(self):
        unique_id = self.get_unique_id()
        initialize_db(db_name='zenit_oms.db')
        
        if self.with_trend_study:
            if self.interval in ['1m','10m', '5m','15m']:
                self.trend_styudy(unique_id, '1h')
            else:
                self.trend_styudy(unique_id, '1d')
            
        self.initial_balance = self.get_account_balance()
        self.highest_balance = self.initial_balance
        logger.info(f"Initical balance {self.initial_balance}")
        # if datetime.datetime.now() < self.start_time():
        #     seconds_to_wait = (self.start_time() - datetime.datetime.now()).total_seconds()
        #     Timer(seconds_to_wait, self.main).start()
        #     logger.info("Starting later today at: {}".format(self.start_time().time()))
        #     return None

        seconds_to_wait = (self.start_time() - datetime.datetime.now() + datetime.timedelta(days=1)).total_seconds()
        Timer(seconds_to_wait, self.main).start()

        # if datetime.datetime.now().weekday() in [5, 6]:
        #     logging.info("It's the weekend, no trading today")
        #     return None

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
    
    def trend_styudy(self, req_id, interval):
        data = self.get_study_data(req_id, interval)
        # Calcular indicadores necesarios
        data['EMA_11'] = ta.trend.ema_indicator(data['Close'], window=11)
        data['EMA_55'] = ta.trend.ema_indicator(data['Close'], window=55)

        # Calcular el MACD y la señal
        data['MACD'] =  data['EMA_11'] -  data['EMA_55']
        data['Signal'] =  data['MACD'].ewm(span=34, adjust=False).mean()
        
        if ((data['EMA_11'][-1] > data['EMA_55'][-1])): # and (self.fdata['MACD'] > self.fdata['Signal'])):
            self.accept_trade = 'ab'
            logger.info('************************ LONG')
        elif ((data['EMA_11'][-1] < data['EMA_55'][-1])): # and (self.fdata['MACD'] < self.fdata['Signal'])):
            self.accept_trade = 'Short'
            logger.info('************************ SHORT')
        
    
    def update_positions(self):
        self.positions1[self.symbol] = {
                        "position": 0,
                        "averageCost": 0
                    }
        self.reqPositions()
        time.sleep(3)

        print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
        self.cant_cont = self.positions1[self.symbol]["position"]
    

    def cal_cont(self, cal):
        if cal == 1:
            self.cont_1 = int(self.quantity) 
        elif cal == 2:
            self.cont_1 = int(self.quantity) + self.order_n1
        elif cal == 3:
            self.cont_1 = int(self.quantity) + self.order_n1 + self.order_n2 
        elif cal == 4:
            self.cont_1 = int(self.quantity) + self.order_n1 + self.order_n2 + self.order_n3

    def request_firts_position(self):
        self.positions1[self.symbol] = {
                        "position": 0,
                        "averageCost": 0
                    }
        self.reqPositions()
        time.sleep(1)

        print(f'**** POSITION {self.positions1[self.symbol]["position"]}')
        self.cant_cont = self.positions1[self.symbol]["position"]

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
        
        if 'Short_Exit' not in self.fdata.columns:
            self.fdata['Short_Exit'] = 0
            self.fdata['Open_position'] = 0
            self.fdata['Close_real_price'] = 0
            logger.info('Se agregaron indicadores Short_Exit y Open_position')
        
        #self.position_gestion()
        self.request_firts_position()

        while True:

            self.estrategy_jemir()

            self.plot_strategy_jemir()
            self.html_generate() 

            # time.sleep(self.bar_size)
            logger.info(f'Esperando {self.bar_size} segundos para agregar precios')
            #self.buy(self.open_trade_price, float(self.quantity), "MARKET")
            
            # Bid Price y Ask Price durante un minuto
            try:
                # Obtén la hora actual
                 # Obtén la hora actual
                hora_actual = time.localtime()
                segundos_actuales = (hora_actual.tm_min * 60) + hora_actual.tm_sec
                parte_decimal, _ = math.modf((3600 - segundos_actuales) / self.bar_size)
                segundos_hasta_siguiente_tiempo = int(parte_decimal * self.bar_size)     
                
                if segundos_hasta_siguiente_tiempo > 0:
                    tiempo_de_espera = min(self.bar_size, segundos_hasta_siguiente_tiempo)
                else:
                    tiempo_de_espera = self.bar_size
                datos_prices = []
                for i in tqdm(range(tiempo_de_espera)):
                    time.sleep(1)
                    try:
                        if self.is_paper:
                            price = (self.market_data[req_id].DelayedBid + self.market_data[req_id].DelayedAsk) / 2
                            market_price = self.market_data[req_id].DelayedAsk
                            sell_market_price = self.market_data[req_id].DelayedBid
                            vol = self.market_data[req_id].DelayedVolume
                        else:
                            print('LIVE DATA')
                            price = (self.market_data[req_id].Bid + self.market_data[req_id].Ask) / 2
                            market_price = self.market_data[req_id].Ask 
                            sell_market_price = self.market_data[req_id].Bid
                            vol = self.market_data[req_id].NotDefined
                        logger.info(f'PRECIO ------------> ${price}')
                        logger.info(f'PRECIO DE MERCADO--> ${market_price}')
                        datos_prices.append(price)

                        # Para cerrar el proceso en caso de existir algunas de estas condiciones
                        if not self.open_position:
                            if (price > 0):
                                break

                        if self.open_position:
                            if (self.average_purchase_price > 0) and ((price > self.average_purchase_price) or (price < self.average_purchase_price)):
                                logger.info('Actualización de graficas')
                                break

                    except TypeError:
                        logger.info(f"Error TypeError, the price is None")

                new_price_info = self.get_data_today(req_id)
                new_price_info['Short_Exit'] = 0
                new_price_info['Open_position'] = 0
                new_price_info['Close_real_price'] = 0
                if len(new_price_info) > 0:
                                        
                    self.fdata = pd.concat([self.fdata,new_price_info])
                    
                    self.fdata = self.fdata[~self.fdata.index.duplicated(keep='last')]

                    self.estrategy_jemir()
                    
                    logger.info('CALCULO DE MÉTRICAS ')
                    logger.info(f'Ultima Vela {self.fdata[-1:].T}')
                    
                    self.strategy_metrics_jemir(price, market_price, sell_market_price, req_id)
                    logger.info('ANALIZANDO ESTRATEGIA')
                    logger.info(f'Métricas de la estrategia {self.dict_metrics}')
                    self.request_firts_position()                    
                    # cont += 1                
                        
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
            datos_prices = []
            logger.info(f'CALL DOWN ACTIVADO ----> {time_}/{self.call_down_wait}')
            for i in tqdm(range(tiempo_de_espera), desc=f"CONSTRUYENDO VELAS EN EL CALL DOWN"):
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
        
        formatos_validos = ['1m','2m', '5m','10m','15m', '30m', '60m', '90m', '1h', '4h','1d', '1wk', '1mo', '3mo']
        
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
            bar_size = "1 min"  # Tamaño de las barras
        elif self.interval == '5m':
            bar_size = "5 mins"  # Tamaño de las barras
        elif self.interval == '15m':
            bar_size = "15 mins"  # Tamaño de las barras
        elif self.interval == '10m':
            bar_size = "10 mins"  # Tamaño de las barras
        elif self.interval == '1h':
            bar_size = "1 hour"  # Tamaño de las barras
        elif self.interval == '4h':
            bar_size = "4 hours"  # Tamaño de las barras
        elif self.interval == '30m':
            bar_size = "30 mins"  # Tamaño de las barras
        elif self.interval == '90m':
            bar_size = "90 mins"  # Tamaño de las barras
        
        if self.interval == '1m':
            duration_str = "3 D"  # Duración de los datos históricos
        elif self.interval in ['2m', '5m','10m', '15m', '30m', '1h']:
            duration_str = "7 D"  # Duración de los datos históricos
    
        elif self.interval in ['4h']:
            duration_str = "10 D"  # Duración de los datos históricos

        self.historical_market_data[req_id] = self.get_historical_market_data(self.contract, duration_str, bar_size)
        # print(self.historical_market_data[req_id])
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
        
        #print(self.historical_market_data[req_id])

    def get_study_data(self, req_id, interval):
                    
        if interval in ['1h']:
            duration_str = "7 D"  # Duración de los datos históricos  # Duración de los datos históricos
            self.new_barsize = "1 hour"
            print(duration_str)
        elif interval in ['1d']:
            duration_str = "60 D"  # Duración de los datos históricos
            self.new_barsize = "1 day"
            print(duration_str)

        historical_market_data = self.get_historical_market_data(self.contract, duration_str, self.new_barsize)
        # logger.info(self.historical_market_data[req_id])
        bar_data_dict_list = [
                                {"Date": data.date, "Open": data.open, "High": data.high, "Low": data.low, "Close": data.close, "Volume": data.volume}
                                for data in historical_market_data
                            ]
        df = pd.DataFrame(bar_data_dict_list, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        print(df)
        #df.to_csv('data_ib.csv')
        if interval in ['1h']:
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
        # print(self.historical_market_data[req_id])
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
    
    
    def estrategy_jemir(self):
        self.fdata = self.fdata[self.fdata['Close']>0]
        # Calcular el ADX 
        adx = ta.trend.ADXIndicator(self.fdata['High'],self.fdata['Close'],self.fdata['Low'], window=14, fillna=True)
        self.fdata[f'{self.symbol}_ADX'] = adx.adx()
        
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

        # Utilizar los precios de cierre para calcular el Volume Profile
        prices = self.fdata['Close'].to_numpy()
        
        # Calcular el Volume Profile
        hist, bins = np.histogram(prices, bins=20)
        
        # Encontrar el índice del bin con la frecuencia máxima (POC)
        poc_index = np.argmax(hist)
        
        # DataFrame de Volumen con frecuencia
        self.volume_df = pd.DataFrame({'Close':bins[:-1], 'Frecuency': hist})
        
        # Calcular el precio del POC
        self.poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2

        self.fdata['High-Low'] = self.fdata['High'] - self.fdata['Low']
        self.fdata['High-PreviousClose'] = abs(self.fdata['High'] - self.fdata['Close'].shift(1))
        self.fdata['Low-PreviousClose'] = abs(self.fdata['Low'] - self.fdata['Close'].shift(1))
        self.fdata['TR'] = self.fdata[['High-Low', 'High-PreviousClose', 'Low-PreviousClose']].max(axis=1)

        # Calcular el True Positive (+DI) y True Negative (-DI)
        window = 14
        
        adx_indicator = ta.trend.ADXIndicator(high=self.fdata['High'], low=self.fdata['Low'], close=self.fdata['Close'], window=14)

        # Calcular +DI y -DI
        self.fdata['+DI'] = adx_indicator.adx_pos()
        self.fdata['-DI'] = adx_indicator.adx_neg()

        # Calcular el ADX
        # self.fdata['DX'] = (abs(self.fdata['+DI'] - self.fdata['-DI']) / (self.fdata['+DI'] + self.fdata['-DI'])) * 100
        # self.fdata[f'{self.symbol}_ADX'] = self.fdata['DX'].rolling(window=14).mean()

        # Obtén los dos últimos valores del ADX
        self.last_two_adx_values = self.fdata[f'{self.symbol}_ADX'][-2:]

        # Calcula la diferencia entre los dos últimos valores del ADX
        self.adx_slope = self.last_two_adx_values[1] - self.last_two_adx_values[0]

        self.di_plus = round(self.fdata['+DI'].describe()['50%'], 2)
        self.di_minus = round(self.fdata['-DI'].describe()['50%'], 2)

        # Definir umbrales y condiciones
        if self.accept_trade == 'ab':
            self.mid_point = round(self.fdata[(self.fdata['+DI'] >= self.di_plus) & (self.fdata['+DI'] <= 40)]['+DI'].describe()['75%'], 3)
        else:
            self.mid_point = round(self.fdata[(self.fdata['-DI'] >= self.di_minus) & (self.fdata['-DI'] <= 40)]['-DI'].describe()['75%'], 3)
        
        ##################################################
        # Estrategia                                     #
        ##################################################
        # Calcular indicadores necesarios
        self.fdata['EMA_34'] = ta.trend.ema_indicator(self.fdata['Close'], window=34)
        self.fdata['EMA_55'] = ta.trend.ema_indicator(self.fdata['Close'], window=55)
        self.fdata['Visible_Range_POC'] = self.poc_price # Calcula el POC del Visible Range Volume Profile según tu método
        
        if self.accept_trade == 'a':
        
            # Generar señales de entrada
            self.fdata['Long_EntryA'] = np.where(
                (self.fdata[f'{self.symbol}_ADX'] > self.di_plus) &
                (self.fdata['Squeez_Momentum_Phase'] == 'Impulso Alcista') &
                (self.fdata['Close'] >= self.fdata['EMA_55']) &
                (self.fdata['EMA_34'] > self.fdata['EMA_55']) &
                (self.fdata['+DI'] >= self.di_plus) &
                (self.fdata['+DI'] > self.fdata['-DI']) &
                (self.adx_slope > 0) &
                (self.fdata['Close'] > self.fdata['Visible_Range_POC']),
                1, 0
            )
            self.fdata['Long_EntryB'] = 0
        elif self.accept_trade == 'b':
            self.fdata['Long_EntryB'] = np.where(
                (self.fdata[f'{self.symbol}_ADX'] > self.di_plus) &
                (self.fdata['Squeez_Momentum_Phase'] == 'Impulso Alcista') &
                (self.fdata['Close'] >= self.fdata['EMA_55']) &
                (self.fdata['EMA_34'] > self.fdata['EMA_55']) &
                (self.fdata['+DI'] >= self.di_plus) &
                (self.fdata['+DI'] > self.fdata['-DI']) &
                (self.adx_slope > 0) &
                (self.fdata['Close'] <= self.fdata['Visible_Range_POC']),
                1, 0
            )
            self.fdata['Long_EntryA'] = 0
        elif self.accept_trade == 'ab':
            # Generar señales de entrada
            self.fdata['Long_EntryA'] = np.where(
                (self.fdata[f'{self.symbol}_ADX'] > self.di_plus) &
                (self.fdata['Squeez_Momentum_Phase'] == 'Impulso Alcista') &
                (self.fdata['Close'] >= self.fdata['EMA_55']) &
                (self.fdata['EMA_34'] > self.fdata['EMA_55']) &
                # (self.fdata['+DI'] >= self.di_plus) &
                # (self.fdata['+DI'] > self.fdata['-DI']) &
                # (self.adx_slope > 0) &
                (self.fdata['Close'] > self.fdata['Visible_Range_POC']),
                1, 0
            )
            self.fdata['Long_EntryB'] = np.where(
                (self.fdata[f'{self.symbol}_ADX'] > self.di_plus) &
                (self.fdata['Squeez_Momentum_Phase'] == 'Impulso Alcista') &
                (self.fdata['Close'] >= self.fdata['EMA_55']) &
                (self.fdata['EMA_34'] > self.fdata['EMA_55']) &
                # (self.fdata['+DI'] >= self.di_plus) &
                # (self.fdata['+DI'] > self.fdata['-DI']) &
                # (self.adx_slope > 0) &
                (self.fdata['Close'] <= self.fdata['Visible_Range_POC']),
                1, 0
            )
        
            # Clasificar los trades según las condiciones
            self.fdata['Trade_Classification'] = 'C'
            self.fdata.loc[self.fdata['Long_EntryA'] == 1, 'Trade_Classification'] = 'A'
            self.fdata.loc[(self.fdata['Long_EntryB'] == 1) & (self.fdata['Close'] <= self.fdata['Visible_Range_POC']), 'Trade_Classification'] = 'B'

        self.daily_returns = self.fdata['Close'].pct_change().dropna()  # Calcular los retornos diarios
        self.volatility = self.daily_returns.std() # Calcular la volatilidad del periodo estudiado
        self.volatility = self.volatility * 3
        self.update_strategies_values()
        print(f'VOLATILIDAD ******** --------> {self.take_profit_percent}')
        print(f'SAFETY ORDER 1 ******** -------->{self.orders_info["safety_orders"]["so_1"]}')
        print(f'SAFETY ORDER 2 ******** -------->{self.orders_info["safety_orders"]["so_2"]}')
        print(f'STOP LOSS ******** -------->{self.stop_loss_percent}')
    

    def round_to_tick(self, price):
        return round(price / self.tick_value) * self.tick_value
    
    def strategy_metrics_jemir(self, price, market_price, sell_market_price, req_id):
        
        # Iterar a través de los datos para simular la estrategia
        last_row = self.fdata[-1:]
        logger.info(f' * * * * * * Hora de actualización: {last_row.index[0]}')

        self.current_price = price
        
        if (self.with_trend_study) and (not self.open_position):
            if self.interval in ['1m','5m','15m', '10m']:
                self.trend_styudy(req_id, '1h')
            else:
                self.trend_styudy(req_id, '1d')
        
        
        # if self.open_position and self.safety_order_1:
        #     self.stoploss_activate = True
        # else:
        #     self.stoploss_activate = False
        
        if (self.out_position) and (self.fdata['+DI'][last_row.index[0]] <= self.di_plus):
            self.out_position = False
            
        if self.accept_trade == 'ab':
            
            if self.fdata['Long_EntryA'][last_row.index[0]] == 1:
                logger.info("* * * * Existe Trade tipo A")
            else: 
                logger.info("* * * * NO existe Trade tipo A")
            
            if self.fdata['Long_EntryB'][last_row.index[0]] == 1:
                logger.info("* * * * Existe Trade tipo B")
            else: 
                logger.info("* * * * NO existe Trade tipo B")
                
            # Abrir una posición si hay señal de entrada y no hay una posición abierta
            if ((self.fdata['Long_EntryA'][last_row.index[0]] == 1 or self.fdata['Long_EntryB'][last_row.index[0]] == 1)  and 
                (not self.open_position) and
                (self.current_price < self.fdata['EMA_34'][last_row.index[0]]) and
                (self.current_price >= self.fdata['EMA_55'][last_row.index[0]])
                ):
                            
                # Fijamos el precio de entrada
                if self.active_safety_orders > 0:
                        self.active_safety_orders = 0

                self.open_trade_price = self.current_price
                self.buy(self.open_trade_price, float(self.quantity), "LIMIT")
                self.update_positions()
                self.cal_cont(1)

                if self.cant_cont > self.cant_cont_init:
                    self.open_position = True
                else:
                    tiempo_inicio = time.time()

                    for i in tqdm(range(self.wait_to_trade), desc="Espera, ejecución orden de entrada"):
                        self.request_firts_position()
                        # Actualizar las posiciones y otras variables necesarias
                        self.update_positions()

                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont > self.cant_cont_init:
                            self.open_position = True
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= self.wait_to_trade:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.reqGlobalCancel()
                            break


                if self.open_position:
                    # Promedio ponderado de precios con peso igual a los contratos adquiridos
                    self.average_purchase_price = self.open_trade_price
                    self.trades_info['action'].append('Buy')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.open_trade_price)
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
                    self.fdata.loc[last_row.index[0], 'Open_position'] = 1
                    self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                    logger.info(f"Opened long position at price {self.open_trade_price}.")
                

            elif self.open_position and (self.cant_cont > 0): #and (self.active_safety_orders <= self.max_safety_orders):
                
                if (not self.safety_order_1) and (self.current_price < self.average_purchase_price * (1-self.orders_info['safety_orders']['so_1'])):
                                    
                    # Fijamos el precio de entrada
                    self.open_trade_price1 = self.current_price
                    self.buy(self.open_trade_price1, self.order_n1, "LIMIT")
                    self.update_positions()
                    self.cal_cont(2)

                    if self.cant_cont == self.cant_cont_init + int(self.quantity):
                        tiempo_inicio = time.time()

                        for i in tqdm(range(self.wait_to_trade), desc="Espera, ejecución safety order 1"):
                            self.request_firts_position()
                            self.update_positions()
                            

                            # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                            if self.cant_cont >= self.cant_cont_init + self.cont_1:
                                break

                            # Verificar si ha transcurrido el tiempo límite
                            tiempo_transcurrido = time.time() - tiempo_inicio
                            if tiempo_transcurrido >= self.wait_to_trade:
                                self.cal_cont(1)
                                # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                                self.reqGlobalCancel()
                                break

                    if self.cant_cont >= self.cant_cont_init + self.cont_1:
                        self.request_firts_position()
                        # Promedio ponderado de precios con peso igual a los contratos adquiridos
                        self.average_purchase_price = ((int(self.quantity) * self.open_trade_price) + 
                                                        (self.order_n1 * self.open_trade_price1)
                                                        ) / (int(self.quantity) + self.order_n1)
                        self.fdata.loc[last_row.index[0], 'Open_position'] = 2
                        self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                        
                        logger.info(f"Opened 1 safety position at price {self.open_trade_price1}.")
                        self.active_safety_orders += 1
                        self.trades_info['action'].append('Buy')
                        self.trades_info['time'].append(last_row.index[0])
                        self.trades_info['price'].append(self.open_trade_price1)
                        self.trades_info['contracts'].append(float(self.order_n1))
                        self.trades_info['Short_Exit'].append(0)
                        self.trades_info['Open_position'].append(2)
                        store_action(self.account,
                                    self.strategy_name,
                                    self.interval,
                                    self.symbol,
                                    self.accept_trade,
                                    self.trade_id, 
                                    "Buy", 
                                    str(last_row.index[0]), 
                                    self.round_to_tick(self.open_trade_price1), 
                                    float(self.order_n1),
                                    short_exit=0, 
                                    open_position=2)
                        self.safety_order_1 = True
                    
                    
                # elif (self.safety_order_1) and (not self.safety_order_2) and (self.current_price < self.average_purchase_price * (1-self.orders_info['safety_orders']['so_2'])):
                    
                #     self.open_trade_price2 = self.current_price
                #     # Fijamos el precio de entrada
                #     self.buy(self.open_trade_price2, self.order_n2, "LIMIT")
                #     self.update_positions()

                #     if self.cant_cont == self.cant_cont_init + self.cont_1:
                #         tiempo_inicio = time.time()

                #         for i in tqdm(range(self.wait_to_trade), desc="Espera, ejecución safety order 2"):
                #             self.request_firts_position()
                #             # Actualizar las posiciones y otras variables necesarias
                #             self.update_positions()

                #             # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                #             if self.cant_cont >= self.cant_cont_init + self.cont_1:
                #                 self.cal_cont(3)
                #                 break

                #             # Verificar si ha transcurrido el tiempo límite
                #             tiempo_transcurrido = time.time() - tiempo_inicio
                #             if tiempo_transcurrido >= self.wait_to_trade:
                #                 # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                #                 self.reqGlobalCancel()
                #                 break



                #     if self.cant_cont >= self.cant_cont_init + (int(self.quantity) + self.order_n1 + self.order_n2):
                #         self.request_firts_position()
                #         self.cal_cont(3)
                #         # Promedio ponderado de precios con peso igual a los contratos adquiridos
                #         self.average_purchase_price = ((int(self.quantity) * self.open_trade_price) + 
                #                                         (self.order_n1 * self.open_trade_price1) + 
                #                                         (self.order_n2 * self.open_trade_price2)
                #                                 ) / (int(self.quantity) + self.order_n1 + self.order_n2)
                        
                #         self.fdata.loc[last_row.index[0], 'Open_position'] = 3
                        
                #         logger.info(f"Opened 2 safety position at price {self.open_trade_price2}.")
                #         self.active_safety_orders += 1
                #         self.trades_info['action'].append('Buy')
                #         self.trades_info['time'].append(last_row.index[0])
                #         self.trades_info['price'].append(self.open_trade_price2)
                #         self.safety_order_2 = True
                    

            if (self.open_position) and (not self.out_position):
                self.required_take_profit = (1 + self.take_profit_percent) * self.average_purchase_price

                self.check_trailing_stop_activate(last_row)
                self.execute_trailing_stop(last_row, req_id)
                
            # Verificar la regla de cierre basada stop loss
            if (self.open_position and 
                self.safety_order_1 and 
                (self.current_price < ((1 - self.stop_loss_percent) * self.average_purchase_price))):
                
                self.sell(self.current_price, self.cont_1, "LIMIT")
                self.current_contracts = self.cant_cont
                self.update_positions()
                
                if self.cant_cont != self.cant_cont_init:
                    
                    tiempo_inicio = time.time()

                    for i in tqdm(range(self.wait_to_trade), desc="Espera, ejecución orden de venta de stop loss"):
                        self.request_firts_position()
                        self.update_positions()
                        # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                        if self.cant_cont == self.cant_cont_init:
                            break

                        # Verificar si ha transcurrido el tiempo límite
                        tiempo_transcurrido = time.time() - tiempo_inicio
                        if tiempo_transcurrido >= self.wait_to_trade:
                            # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                            self.reqGlobalCancel()
                            self.open_position = True
                            break 

                if self.cant_cont == self.cant_cont_init:
                    self.cal_cont(1)
                    self.stoploss_activate = False
                    # cap_final +=  cant_cont * self.current_price
                    self.fdata['Short_Exit'][last_row.index[0]] = 1
                    self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
                    logger.info(f"Closed long position at price {self.current_price} based on stop loss.")
                    # self.loss_dict['price'].append(self.average_purchase_price) 
                    # self.loss_dict['loss'].append(self.current_price)
                    # self.loss_dict['time'].append(last_row.index[0])
                    
                    # Calcular ganancias o pérdidas de la operación
                    trade_profit = (self.current_price - self.average_purchase_price) / self.average_purchase_price
                    logger.info(f'trade profit {trade_profit}')
                    self.total_profit += trade_profit
                    logger.info(f'Venta por stop loss, quedan {self.cant_cont} contratos disponibles')
                    if trade_profit > 0:
                        self.successful_trades += 1
                        if trade_profit > self.take_profit_percent:
                            self.profitable_trades += 1
                    self.operations += 1
                
                    self.trades_info['action'].append('Sell')
                    self.trades_info['time'].append(last_row.index[0])
                    self.trades_info['price'].append(self.current_price)
                    self.trades_info['contracts'].append(float(self.current_contracts))
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
                                float(self.current_contracts),
                                short_exit=1, 
                                open_position=0)
                    self.open_position = True
                    self.trailing_stop_activate_di = False
                    self.trailing_stop_activate_price = False
                    self.trailing_stop_activate_di_40 = False
                    self.safety_order_1 = False
                    self.safety_order_2 = False
                    self.out_position = False
                    self.trade_id = generate_random_id()
                    self.time_to_call_down(req_id)
            
            # Calcular métricas generales
            total_trades = self.successful_trades + (self.operations - self.successful_trades)
            win_rate = self.successful_trades / total_trades if total_trades > 0 else 0
            profit_factor = self.total_profit if self.total_profit > 0 else 0
            
            # Calcular el apalancamiento en términos de porcentaje
            # leverage_percentage = (self.total_profit / initial_capital) * 100
            # Calcula el drawdown y el drawdown máximo
            cumulative_max = self.fdata['Close'].cummax()
            drawdown = (self.fdata['Close'] - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()
            average_trade_duration = total_trades / len(self.fdata)

            # Calcula los rendimientos diarios
            self.fdata['Daily_Return'] = self.fdata['Close'].pct_change()
            
            # Calcula el rendimiento promedio y la desviación estándar de los rendimientos diarios
            # average_daily_return = self.fdata['Daily_Return'].mean()
            # std_daily_return = self.fdata['Daily_Return'].std()
            
            # Calcula el Sharpe Ratio
            # sharpe_ratio = (average_daily_return - self.risk_free_rate) / std_daily_return


            average_win_duration = self.successful_trades / total_trades * average_trade_duration
            average_loss_duration = (total_trades - self.successful_trades) / total_trades * average_trade_duration
            profitable_percentage = self.profitable_trades / total_trades * 100

            cont_com = self.cant_cont + self.cont_ven
            self.dict_metrics = {
                'total_trades':total_trades,
                'win_rate':win_rate,
                'profit_factor':profit_factor,
                # 'leverage_percentage':leverage_percentage,
                'successful_trades':self.successful_trades,
                'drawdown':drawdown,
                'max_drawdown':max_drawdown,
                'average_trade_duration':average_trade_duration,
                # 'sharpe_ratio':sharpe_ratio,
                'average_win_duration':average_win_duration,
                'average_loss_duration':average_loss_duration,
                'profitable_percentage':profitable_percentage,
                'contratos_adquiridos':cont_com,
                'contratos_vendidos': self.cont_ven,
                'contratos_posesion': self.cant_cont
            }


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
        
        
    def check_trailing_stop_activate(self,last_row):
        
        if ((self.current_price > self.required_take_profit) ): 
            self.trailing_stop_activate_di = True
            print('TRAILING STOP ACTIVADO REQUIRED TAKE PROFIT')

        if ((self.current_price >= (self.average_purchase_price + self.tick_price + self.tick_value))
            ): 
            self.trailing_stop_activate_price = True
            print('TRAILING STOP ACTIVADO PRICE 40 TICKS')

        if ((self.fdata['+DI'][last_row.index[0]] > 40) and
              (self.current_price > self.average_purchase_price)
            ): 
            self.trailing_stop_activate_di_40 = True
            print('TRAILING STOP ACTIVADO +DI 40')
    

    def close_position_method(self, last_row, req_id):
            
        self.sell(self.required_take_profit, self.cont_1, "LIMIT")
        contracts_to_sell = self.cont_1
        self.cont_ven += contracts_to_sell
        self.update_positions()
        
        if self.cant_cont != self.cant_cont_init:
            
            tiempo_inicio = time.time()

            for i in tqdm(range(self.wait_to_trade), desc="Espera, ejecución orden de venta take profit"):
                self.request_firts_position()
                self.update_positions()
                # Verificar si la posición es mayor que 0 para establecer open_position en True y salir del bucle
                if self.cant_cont == self.cant_cont_init:
                    break

                # Verificar si ha transcurrido el tiempo límite
                tiempo_transcurrido = time.time() - tiempo_inicio
                if tiempo_transcurrido >= self.wait_to_trade:
                    # Si ha transcurrido el tiempo límite, cancelar la orden y salir del bucle
                    self.reqGlobalCancel()
                    self.open_position = True
                    break 
        
        if self.cant_cont == self.cant_cont_init:
            self.operations += 1
            self.cal_cont(1)
            logger.info(f'se vendieron {contracts_to_sell} quedan {self.cant_cont}')
            # cap_final += self.current_price
            self.fdata.loc[last_row.index[0], 'Short_Exit'] = 1
            self.fdata.loc[last_row.index[0], 'Close_real_price'] = self.round_to_tick(self.current_price)
            logger.info(f"Closed long position at price {self.current_price} based on Safety Order 2 take profit.")
            # self.profit_dict['price'].append(self.average_purchase_price)
            # self.profit_dict['profit'].append(self.current_price)
            trade_profit = (self.current_price - self.average_purchase_price) / self.average_purchase_price
            logger.info(f'trade profit {trade_profit}')
            self.total_profit += trade_profit
            self.active_safety_orders = 0
            self.trades_info['action'].append('Sell')
            self.trades_info['time'].append(last_row.index[0])
            self.trades_info['price'].append(self.current_price)
            self.trades_info['contracts'].append(float(self.cont_1))
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
                        self.cont_1,
                        short_exit=1, 
                        open_position=0)
            self.open_position = False
            self.trailing_stop_activate_di = False
            self.trailing_stop_activate_price = False
            self.trailing_stop_activate_di_40 = False
            self.safety_order_1 = False
            self.safety_order_2 = False
            self.out_position = True
            if trade_profit > 0:
                self.successful_trades += 1
            self.trade_id = generate_random_id()
            self.time_to_call_down(req_id)
        
    def execute_trailing_stop(self, last_row, req_id):
        if ((self.trailing_stop_activate_di) and
            (self.current_price <= self.required_take_profit) 
            ):
            print(f'*---------------- ACTIVADO TRALING STOP DE +DI EN {self.di_plus}')
            self.close_position_method(last_row, req_id)

        if ((self.trailing_stop_activate_price) and
            (self.current_price <= (self.average_purchase_price + self.tick_price))
            ):
            print(f'*---------------- ACTIVADO TRALING STOP PRECIO {self.average_purchase_price + self.tick_price}')
            self.close_position_method(last_row, req_id)
        
        if ((self.trailing_stop_activate_di_40) and
            (self.fdata['+DI'][last_row.index[0]] <= 40) and 
            (self.current_price > self.average_purchase_price) 
            ):
            print('*---------------- ACTIVADO TRALING STOP DE +DI EN 40')
            self.close_position_method(last_row, req_id)



    def update_strategies_values(self):
        self.orders_info["safety_orders"] ={
                    "so_1": round(self.volatility, 4),
                    "so_2": round(self.volatility, 4) * 2,
                    "so_3": round(self.volatility, 4) * 3,
                    "so_4": round(self.volatility, 4) * 4,
                }
        self.stop_loss_percent = round(self.volatility, 4)
        self.take_profit_percent = round((self.volatility*2.3) , 4)

    def plot_strategy_jemir(self):
        
        self.df_activity = pd.DataFrame(self.trades_info)
        
        self.figure=make_subplots( 
                        rows = 2,
                        cols=2,
                        shared_xaxes = True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing = 0.06,
        specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"colspan": 2}, None]]
        )
        
        self.figure.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'}, xaxis_domain=[0, 0.94])
        
        self.figure.add_trace(go.Candlestick(
                        x = self.fdata.index,
                        open = self.fdata['Open'],
                        high = self.fdata['High'],
                        low = self.fdata['Low'],
                        close = self.fdata['Close'],
                        name=f'Precio de {self.contract.symbol}' 
                        ),
                        col=1,
                        row=1,
                        secondary_y = False,
                        )
        # fig.add_trace( go.Bar(x=[1, 2, 3, 4], y=[7, 4, 5, 6], name='bar', orientation = 'h',opacity = 0.5), secondary_y=True)
        # Agregar el gráfico de barras de volumen en la segunda columna (encima del gráfico de velas)
        volume_bars_trace = go.Bar(
            y=self.volume_df['Close'],
            x=self.volume_df['Frecuency'],
            orientation='h',
            name='Volumen',
            opacity = 0.2
        )
        self.figure.add_trace(volume_bars_trace, secondary_y=True, col=1,row=1)
        
        self.figure.add_trace(
            go.Scatter(
            x= self.fdata.index, 
            y=self.fdata[f'{self.symbol}_ADX'],
            line = dict(color='blue', dash='dash'),
            name=f'{self.symbol}_ADX'
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
        
        self.figure.add_trace(go.Scatter(
                x=self.df_activity[self.df_activity['Open_position'] == 1]['time'],
                y=self.df_activity[self.df_activity['Open_position'] == 1]['price'],
                mode= 'markers',
                name = 'Compra',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='star-triangle-up'
                ) ),col=1,
                    row=1
            )
        self.figure.add_shape(type="line", 
                            x0=self.fdata.index[0], 
                            y0=40, 
                            x1=self.fdata.index[-1], 
                            y1=40, 
                            line=dict(color="gray", dash="dash"),
                            col=1,
                            row=2)
        for i in range(2,6):
            try:
                self.figure.add_trace(go.Scatter(
                            x=self.df_activity[self.df_activity['Open_position'] == i]['time'],
                            y=self.df_activity[self.df_activity['Open_position'] == i]['price'],
                            mode= 'markers',
                            name = f'Safety Order {i-1}',
                            marker=dict(
                                size=15,
                                color= self.colors[i],
                                symbol='star-triangle-up'
                            )
                                                ),
                                
                                col=1,
                                row=1
                        )
            except:
                pass
        
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
                    col = 1,
                    row = 1
        )
        
        self.figure.add_trace(
        go.Scatter(
        x= self.fdata.index, 
        y=self.fdata['EMA_34'],
        line = dict(color='blue',width=2),
        name='EMA 34'
        ),
        col=1,
        row=1
        )

        self.figure.add_trace(
        go.Scatter(
        x= self.fdata.index, 
        y=self.fdata['EMA_55'],
        line = dict(color='orange',width=2),
        name='EMA 55'
        ),
        col=1,
        row=1
        )
        
        # Add a horizontal line with title to exit_ind plot
        self.figure.add_shape(
            type="line",
            x0=self.fdata.index[0],
            x1=self.fdata.index[-1],
            y0=self.threshold_adx,
            y1=self.threshold_adx,
            line=dict(color="red", width=2, dash="dash"),
            col=1,
            row=2
        )

        self.figure.add_shape(type="line", 
                                x0=self.fdata.index[0], 
                                y0=self.di_plus, 
                                x1=self.fdata.index[-1], 
                                y1=self.di_plus, 
                                line=dict(color="gray", dash="dash"),
                                col=1,
                                row=2)
       
        self.figure.add_shape(
            type="line",
            x0=self.fdata.index[0],
            x1=self.fdata.index[-1],
            y0=self.poc_price,
            y1=self.poc_price,
            line=dict(color="green", width=3, dash="dash"),
            col=1,
            row=1
        )

        self.figure.data[1].update(xaxis='x2')
        #self.figure.update_yaxes(range=[min(self.fdata['Close']), max(self.fdata['Close'])])
        self.figure.update_layout(xaxis_rangeslider_visible=False)
        self.figure.update_layout(width=1500, height=1000)
        self.figure.update_layout(title=f"Estrategia aplicada a {self.symbol} en el intervalo {self.interval}")

    def html_generate(self):
        logger.info('GENERANDO EL HTML ********')
        self.plot_div = pyo.plot(self.figure, output_type='div', include_plotlyjs='cdn', image_width= 1500)
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
        with open(f"bot_activity/{self.strategy_name}_ab_{self.interval}_{self.symbol}_{self.hora_ejecucion}.html", "w") as html_file:
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
                        <h2> Trade Type: {self.accept_trade} </h2>
                        <h2> Squeeze Phase: {self.fdata['Squeez_Momentum_Phase'][-1]} </h2>
                        <h2>Take profit percent : {round(self.volatility* 100, 3) }%<h2>
                        <h2>Stop Loss percent : {round(self.stop_loss_percent* 100, 3)}%<h2>
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
                var tiempoTotal = 60; // Tiempo en segundos

                // Iniciar el contador al cargar la página
                mostrarContador(tiempoTotal);
            </script>
            </html>
            """)

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address')
    parser.add_argument('--port', type=int, default=7496, help='Port number')
    parser.add_argument('--client', type=int, default=2, help='Client ID')
    parser.add_argument('--symbol', type=str, default='SPY', help='symbol example AAPL')
    parser.add_argument('--secType', type=str, default='STK', help='The security type')
    parser.add_argument('--currency', type=str, default='USD', help='currency')
    parser.add_argument('--exchange', type=str, default='SMART', help='exchange')
    parser.add_argument('--quantity', type=str, default='6', help='quantity')
    
    parser.add_argument('--stop_limit_ratio', type=int, default=3, help='stop limit ratio default 3')
    
    parser.add_argument('--max_safety_orders', type=int, default=2, help='max safety orders')
    parser.add_argument('--safety_order_size', type=int, default=2, help='safety order size')
    parser.add_argument('--volume_scale', type=int, default=2, help='volume scale')
    parser.add_argument('--safety_order_deviation', type=float, default=0.05, help='% safety_order_deviation')
    parser.add_argument('--account', type=str, default='DU7774793', help='Account')

    parser.add_argument('--take_profit_percent', type=float, default=0.0006, help='Take profit percentage')
    parser.add_argument('--interval', type=str, default='1m', help='Data Time Frame')
    parser.add_argument('--accept_trade', type=str, default='ab', help='Type of trades for trading')
    parser.add_argument('--threshold_adx', type=float, default=23, help='Limit for ADX indicator')
    parser.add_argument('--multiplier', type=str, default="5", help='The multiplier for futures')
    parser.add_argument('--trading_class', type=str, default="MES", help='The trading_class for futures')
    parser.add_argument('--lastTradeDateOrContractMonth', type=str, default="20231215", help='The expire date for futures')
    parser.add_argument('--order_type', type=str, default="LIMIT", help='The type of the order: LIMIT OR MARKET')
    parser.add_argument('--order_validity', type=str, default="DAY", help='The expiration time of the order: DAY or GTC')
    parser.add_argument('--is_paper', type=str_to_bool, default=True, help='Paper or live trading')
    parser.add_argument('--quantity_type', type=str, default='nofixed', help='Quantity type distribution')       
    parser.add_argument('--hora_ejecucion', type=str, default=None, help='Time of execution bot') 
    parser.add_argument('--with_trend_study', type=str_to_bool, default=False, help='Trend study for day trading')
    
    args = parser.parse_args()
    logger.info(f"args {args}")

    bot = BotZenitTrendMaster(args.ip, 
              args.port, 
              args.client, 
              args.symbol, 
              args.secType, 
              args.currency, 
              args.exchange, 
              args.quantity, 
 
              args.stop_limit_ratio,  

              args.max_safety_orders, 
              args.safety_order_size, 
              args.volume_scale, 
              args.safety_order_deviation,
              args.account,

              args.take_profit_percent,
              args.interval,
              args.accept_trade,
              args.threshold_adx,
              args.multiplier,
              args.trading_class,
              args.lastTradeDateOrContractMonth,
              args.order_type, 
              args.order_validity,
              args.is_paper,
              args.quantity_type,
              args.hora_ejecucion,
              args.with_trend_study
              )
    try:
        bot.main()
    except KeyboardInterrupt:
        bot.disconnect()

