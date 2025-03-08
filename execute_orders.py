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
import random

futures_symbols_info = pd.read_csv("data/futures_symbols_v1.csv")

logger = logging.getLogger()
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO)

dict_str = {'EOE': {'description': 'Netherlands  equity index AEX', 'ewma_ticker': 'EOE', 'instrument': 'AEX', 'region': 'EMEA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250117', 'last_trade_day_after': '20250221', 'multiplier': 200}, 'exchange': 'FTA', 'currency': 'EUR', 'trading_class': 'EOE', 'point_size': 200}, 'M6A': {'description': 'AUDUSD micro', 'ewma_ticker': 'M6A', 'instrument': 'AUD_micro', 'region': 'US', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250317', 'last_trade_day_after': '20250616', 'multiplier': 10000}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'M6A', 'point_size': 10000}, 'AIGCI': {'description': 'Bloomberg  Commodity Index', 'ewma_ticker': 'AIGCI', 'instrument': 'BBCOMM', 'region': 'US', 'asset_class': 'Ags', 'carry_symbols': {'last_trade_day_now': '20250319', 'last_trade_day_after': '20250618', 'multiplier': 100}, 'exchange': 'CBOT', 'currency': 'USD', 'trading_class': 'AW', 'point_size': 100}, 'MCD': {'description': 'CADUSD_micro', 'ewma_ticker': 'MCD', 'instrument': 'CAD_micro', 'region': 'US', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250318', 'last_trade_day_after': '20250617', 'multiplier': 10000}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'MCD', 'point_size': 10000}, 'RS': {'description': 'Rapeseed  (Canola) in CAD', 'ewma_ticker': 'RS', 'instrument': 'CANOLA', 'region': 'US', 'asset_class': 'Ags', 'carry_symbols': {'last_trade_day_now': '20250114', 'last_trade_day_after': '20250314', 'multiplier': 20}, 'exchange': 'NYBOT', 'currency': 'CAD', 'trading_class': 'RS', 'point_size': 20}, 'MSF': {'description': 'CHFUSD micro', 'ewma_ticker': 'MSF', 'instrument': 'CHF_micro', 'region': 'US', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250317', 'last_trade_day_after': '20250616', 'multiplier': 12500}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'MSF', 'point_size': 12500}, 'MHG': {'description': 'Copper micro', 'ewma_ticker': 'MHG', 'instrument': 'COPPER-micro', 'region': 'US', 'asset_class': 'Metals', 'carry_symbols': {'last_trade_day_now': '20250129', 'last_trade_day_after': '20250226', 'multiplier': 2500}, 'exchange': 'COMEX', 'currency': 'USD', 'trading_class': 'MHG', 'point_size': 2500}, 'YC': {'description': 'Corn mini', 'ewma_ticker': 'YC', 'instrument': 'CORN_mini', 'region': 'US', 'asset_class': 'Ags', 'carry_symbols': {'last_trade_day_now': '20250314', 'last_trade_day_after': '20250514', 'multiplier': 1000}, 'exchange': 'CBOT', 'currency': 'USD', 'trading_class': 'XC', 'point_size': 10}, 'MCL': {'description': 'mic ro WTI crude oil', 'ewma_ticker': 'MCL', 'instrument': 'CRUDE_W_micro', 'region': 'US', 'asset_class': 'OilGas', 'carry_symbols': {'last_trade_day_now': '20250117', 'last_trade_day_after': '20250219', 'multiplier': 100}, 'exchange': 'NYMEX', 'currency': 'USD', 'trading_class': 'MCL', 'point_size': 100}, 'DAX': {'description': 'DAX 30 Inde x (Deutsche Ak tien Xchange 30)', 'ewma_ticker': 'DAX', 'instrument': 'DAX', 'region': 'EMEA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 1.0}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FDAX', 'point_size': 1.0}, 'DJ200S': {'description': 'D ow Jones STOXX Small 200 Index', 'ewma_ticker': 'DJ200S', 'instrument': 'DJSTX-SMALL', 'region': 'EMEA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 50}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FSCP', 'point_size': 50}, 'MYM': {'description': 'Micro E-M ini Dow  Jones Industri al Average Index', 'ewma_ticker': 'MYM', 'instrument': 'DOW', 'region': 'US', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 0.0}, 'exchange': 'CBOT', 'currency': 'USD', 'trading_class': 'MYM', 'point_size': 0.5}, 'MET': {'description': 'Ethereum micro', 'ewma_ticker': 'MET', 'instrument': 'ETHER-micro', 'region': 'US', 'asset_class': 'Metals', 'carry_symbols': {'last_trade_day_now': '20250131', 'last_trade_day_after': '20250228', 'multiplier': 0.0}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'MET', 'point_size': 0.1}, 'SX7E': {'description': 'Do w Jones Euro S TOXX Banks Index', 'ewma_ticker': 'SX7E', 'instrument': 'EU-BANKS', 'region': 'EMEA', 'asset_class': 'Sector', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 50}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FESB', 'point_size': 50}, 'DJSD': {'description': 'Dow Jon es Euro  STOXX Select D ividend 30 Index', 'ewma_ticker': 'DJSD', 'instrument': 'EU-DIV30', 'region': 'EMEA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 10}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FEDV', 'point_size': 10}, 'SXKE': {'description': 'D ow Jones  Euro STOXX Te lecommunications', 'ewma_ticker': 'SXKE', 'instrument': 'EU-DJ-TELECOM', 'region': 'EMEA', 'asset_class': 'Sector', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 50}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FEST', 'point_size': 50}, 'SXIP': {'description': 'Dow Jones STO XX 600 Insurance', 'ewma_ticker': 'SXIP', 'instrument': 'EU-INSURE', 'region': 'EMEA', 'asset_class': 'Sector', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 50}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FSTI', 'point_size': 50}, 'SX86P': {'description': 'D ow Jones STOXX 600 Real Estate', 'ewma_ticker': 'SX86P', 'instrument': 'EU-REALESTATE', 'region': 'EMEA', 'asset_class': 'Sector', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 50}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FSTL', 'point_size': 50}, 'M6E': {'description': 'EURUSD micro', 'ewma_ticker': 'M6E', 'instrument': 'EUR_micro', 'region': 'US', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250317', 'last_trade_day_after': '20250616', 'multiplier': 12500}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'M6E', 'point_size': 12500}, 'EU3': {'description': 'Thr ee Month EURIBOR', 'ewma_ticker': 'EU3', 'instrument': 'EURIBOR', 'region': 'EMEA', 'asset_class': 'Bond', 'carry_symbols': {'last_trade_day_now': '20250113', 'last_trade_day_after': '20250217', 'multiplier': 2500}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FEU3', 'point_size': 2500}, 'DJESS': {'description': 'Euro Stoxx Small', 'ewma_ticker': 'DJESS', 'instrument': 'EUROSTX-SMALL', 'region': 'EMEA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 50}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FSCE', 'point_size': 50}, 'XINA50': {'description': 'FTSE/ Xinhua China A50', 'ewma_ticker': 'XINA50', 'instrument': 'FTSECHINAA', 'region': 'ASIA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250124', 'last_trade_day_after': '20250227', 'multiplier': 1.0}, 'exchange': 'SGX', 'currency': 'USD', 'trading_class': 'XINA50', 'point_size': 1.0}, 'WIIDN': {'description': 'FTSE  Indonesia Index', 'ewma_ticker': 'WIIDN', 'instrument': 'FTSEINDO', 'region': 'ASIA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250130', 'last_trade_day_after': '20250227', 'multiplier': 5}, 'exchange': 'SGX', 'currency': 'USD', 'trading_class': 'WIIDN', 'point_size': 5}, 'FIVNM30': {'description': 'FTSE Vietnam 30 Pr ice Return Index', 'ewma_ticker': 'FIVNM30', 'instrument': 'FTSEVIET', 'region': 'ASIA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250124', 'last_trade_day_after': '20250227', 'multiplier': 5}, 'exchange': 'SGX', 'currency': 'USD', 'trading_class': 'FIVNM30', 'point_size': 5}, 'M6B': {'description': 'GBPUSD_micro', 'ewma_ticker': 'M6B', 'instrument': 'GBP_micro', 'region': 'US', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250317', 'last_trade_day_after': '20250616', 'multiplier': 6250}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'M6B', 'point_size': 6250}, 'MCH.HK': {'description': 'Hang Se ng China Enter prise Index mini', 'ewma_ticker': 'MCH.HK', 'instrument': 'HANGENT_mini', 'region': 'ASIA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250127', 'last_trade_day_after': '20250328', 'multiplier': 10}, 'exchange': 'HKFE', 'currency': 'HKD', 'trading_class': 'MCH.HK', 'point_size': 10}, 'CUS': {'description': 'Housing  Index Composite', 'ewma_ticker': 'CUS', 'instrument': 'HOUSE-US', 'region': 'US', 'asset_class': 'Housing', 'carry_symbols': {'last_trade_day_now': '20250224', 'last_trade_day_after': '20250523', 'multiplier': 250}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'CUS', 'point_size': 250}, 'IBEX': {'description': 'IBEX 35 mini', 'ewma_ticker': 'IBEX', 'instrument': 'IBEX_mini', 'region': 'EMEA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250117', 'last_trade_day_after': '20250221', 'multiplier': 1.0}, 'exchange': 'MEFFRV', 'currency': 'EUR', 'trading_class': 'IBEX', 'point_size': 1.0}, 'SIR': {'description': 'Indian Rupee', 'ewma_ticker': 'SIR', 'instrument': 'INR', 'region': 'US', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250129', 'last_trade_day_after': '20250225', 'multiplier': 5000000}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'SIR', 'point_size': 500}, 'TSEREIT': {'description': 'TSE REIT Index', 'ewma_ticker': 'TSEREIT', 'instrument': 'JP-REALESTATE', 'region': 'ASIA', 'asset_class': 'Sector', 'carry_symbols': {'last_trade_day_now': '20250313', 'last_trade_day_after': '20250612', 'multiplier': 1000}, 'exchange': 'OSE.JPN', 'currency': 'JPY', 'trading_class': 'TSEREIT', 'point_size': 1000}, 'KOSDQ150': {'description': 'KOSDAQ 150 Index', 'ewma_ticker': 'KOSDQ150', 'instrument': 'KOSDAQ', 'region': 'ASIA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250313', 'last_trade_day_after': '20250612', 'multiplier': 10000}, 'exchange': 'KSE', 'currency': 'KRW', 'trading_class': 'KOSDQ150', 'point_size': 10000}, 'K200M': {'description': 'KOSPI mini', 'ewma_ticker': 'K200M', 'instrument': 'KOSPI_mini', 'region': 'ASIA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250109', 'last_trade_day_after': '20250213', 'multiplier': 50000}, 'exchange': 'KSE', 'currency': 'KRW', 'trading_class': 'K200M', 'point_size': 50000}, '3KTB': {'description': 'Ko rean 3 year bond', 'ewma_ticker': '3KTB', 'instrument': 'KR3', 'region': 'ASIA', 'asset_class': 'Bond', 'carry_symbols': {'last_trade_day_now': '20250318', 'last_trade_day_after': '20250617', 'multiplier': 1000000}, 'exchange': 'KSE', 'currency': 'KRW', 'trading_class': '3KTB', 'point_size': 1000000}, 'KU': {'description': 'SGX  Korean W on in US Dolla r Futures (Mini)', 'ewma_ticker': 'KU', 'instrument': 'KRWUSD_mini', 'region': 'ASIA', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250120', 'last_trade_day_after': '20250217', 'multiplier': 25000000}, 'exchange': 'SGX', 'currency': 'USD', 'trading_class': 'KU', 'point_size': 25000}, 'TSEMOTHR': {'description': 'T SE Mothers Index', 'ewma_ticker': 'TSEMOTHR', 'instrument': 'MUMMY', 'region': 'ASIA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250313', 'last_trade_day_after': '20250612', 'multiplier': 1000}, 'exchange': 'OSE.JPN', 'currency': 'JPY', 'trading_class': 'TSEMOTHR', 'point_size': 1000}, 'JPNK400': {'description': 'JPX- Nikkei Index 400', 'ewma_ticker': 'JPNK400', 'instrument': 'NIKKEI400', 'region': 'ASIA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250313', 'last_trade_day_after': '20250612', 'multiplier': 100}, 'exchange': 'OSE.JPN', 'currency': 'JPY', 'trading_class': 'JPNK400', 'point_size': 100}, 'TSR20': {'description': 'TSR 20 Rubber', 'ewma_ticker': 'TSR20', 'instrument': 'RUBBER', 'region': 'ASIA', 'asset_class': 'Other', 'carry_symbols': {'last_trade_day_now': '20250131', 'last_trade_day_after': '20250228', 'multiplier': 50}, 'exchange': 'SGX', 'currency': 'USD', 'trading_class': 'TSR20', 'point_size': 50}, 'M2K': {'description': 'M icro E-Mini Ru ssell 2000 Index', 'ewma_ticker': 'M2K', 'instrument': 'RUSSELL', 'region': 'US', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250321', 'last_trade_day_after': '20250620', 'multiplier': 5}, 'exchange': 'CME', 'currency': 'USD', 'trading_class': 'M2K', 'point_size': 5}, 'US': {'description': 'SGX US Dollar in Singapore Dolla r (Mini) Futures', 'ewma_ticker': 'US', 'instrument': 'SGD_mini', 'region': 'ASIA', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250113', 'last_trade_day_after': '20250217', 'multiplier': 25000}, 'exchange': 'SGX', 'currency': 'SGD', 'trading_class': 'US', 'point_size': 25000}, 'GBS': {'description': 'German 2 year bond Schatz', 'ewma_ticker': 'GBS', 'instrument': 'SHATZ', 'region': 'EMEA', 'asset_class': 'Bond', 'carry_symbols': {'last_trade_day_now': '20250306', 'last_trade_day_after': '20250606', 'multiplier': 1000}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FGBS', 'point_size': 1000}, 'YK': {'description': 'Soybean mini', 'ewma_ticker': 'YK', 'instrument': 'SOYBEAN_mini', 'region': 'US', 'asset_class': 'Ags', 'carry_symbols': {'last_trade_day_now': '20250114', 'last_trade_day_after': '20250314', 'multiplier': 1000}, 'exchange': 'CBOT', 'currency': 'USD', 'trading_class': 'XK', 'point_size': 10}, 'SLI': {'description': 'Sw iss Leader Ind ex (PREIS_INDEX)', 'ewma_ticker': 'SLI', 'instrument': 'SWISSLEAD', 'region': 'EMEA', 'asset_class': 'Equity', 'carry_symbols': {'last_trade_day_now': '20250320', 'last_trade_day_after': '20250620', 'multiplier': 10}, 'exchange': 'EUREX', 'currency': 'CHF', 'trading_class': 'FSLI', 'point_size': 10}, 'TD': {'description': 'SGX Tai wan Doll ar in US Dolla r Futures (Mini)', 'ewma_ticker': 'TD', 'instrument': 'TWD-mini', 'region': 'ASIA', 'asset_class': 'FX', 'carry_symbols': {'last_trade_day_now': '20250113', 'last_trade_day_after': '20250217', 'multiplier': 1000000}, 'exchange': 'SGX', 'currency': 'USD', 'trading_class': 'TD', 'point_size': 100000}, 'V2TX': {'description': 'Vol Eur opean equity V2X', 'ewma_ticker': 'V2TX', 'instrument': 'V2X', 'region': 'EMEA', 'asset_class': 'Vol', 'carry_symbols': {'last_trade_day_now': '20250122', 'last_trade_day_after': '20250219', 'multiplier': 100}, 'exchange': 'EUREX', 'currency': 'EUR', 'trading_class': 'FVS', 'point_size': 100}, 'VXM': {'description': 'Vol US  equity VIX mini', 'ewma_ticker': 'VXM', 'instrument': 'VIX_mini', 'region': 'US', 'asset_class': 'Vol', 'carry_symbols': {'last_trade_day_now': '20250122', 'last_trade_day_after': '20250219', 'multiplier': 100}, 'exchange': 'CFE', 'currency': 'USD', 'trading_class': 'VXM', 'point_size': 100}, 'NKVI': {'description': 'Nikkei 225 Volatility Index', 'ewma_ticker': 'NKVI', 'instrument': 'VNKI', 'region': 'ASIA', 'asset_class': 'Vol', 'carry_symbols': {'last_trade_day_now': '20250114', 'last_trade_day_after': '20250210', 'multiplier': 10000}, 'exchange': 'OSE.JPN', 'currency': 'JPY', 'trading_class': 'NKVI', 'point_size': 10000}}
for sym in dict_str.keys():
    dict_str[sym]["contracts_to_operate"] = random.choice([-100,100])

class BotZenitEXECUTE_ORDERS(Main):

    def __init__(self, 
                ip, 
                port, 
                contracts,
                symbol_list,
                account,
                order_validity="DAY",
                 ):
        client = self.get_unique_id()
        Main.__init__(self, ip, port, client)
        
        self.action1 = "BUY"
        self.action2 = "SELL"
        self.ip = ip
        self.port = port
        self.contracts = eval(contracts)
        self.symbol_list = eval(symbol_list)
        print(f"-----> {type(self.symbol_list)}")
        self.account = account
        
    
        #risk indicators
        self.today = datetime.datetime.now().date() # 
       
        self.average_purchase_price = 0
        
        self.positions1 = {self.account:{}}
        self.is_short = False
        self.is_long = False
        
     
        
        self.trade_id = generate_random_id()
        self.df_activity = None
        self.order_validity = order_validity

    def main(self):
        #self.reqGlobalCancel()
        for symbol in self.symbol_list:
            print(f"SYMBOL ---> {symbol}")
            print(f"CONTRACTS_ SYMBOL {self.contracts[symbol]}")
            contract = self.CONTRACT_CONFIG()
            contract.symbol = symbol
            contract.secType = 'FUT'
            contract.multiplier = self.contracts[symbol]["carry_symbols"]["multiplier"]
            contract.currency = self.contracts[symbol]["currency"]
            contract.exchange = self.contracts[symbol]["exchange"]
            contract.lastTradeDateOrContractMonth = self.contracts[symbol]["carry_symbols"]["last_trade_day_now"]
            contract_details = self.complete_contract(contract)
            print(f"Contract Details: {contract_details}")
            print(contract_details[0].tradingClass)
            contract.tradingClass = contract_details[0].tradingClass
            cant = self.contracts[symbol]["contracts_to_operate"]
            
            if cant < 0:
                self.sell(0, abs(cant), "MARKET", contract)
            else:
                self.buy(0, cant, "MARKET", contract)
                
        self.disconnect()
    
    def buy(self, price, cant, action, contract):
        logger.info("=========================================================== Placing buy order: {} contracts"
              .format(round(cant, 5)))

        self.order_id = self.get_order_id()

        if action == "MARKET":
            order = self.market_order(self.action1, float(cant), self.account)
        elif action == "LIMIT":
            order = self.limit_order(self.action1, float(cant), self.min_price_increment(price), self.account)
        else:
            order = self.stop_limit_order(self.action1, float(cant), self.min_price_increment(price), self.account)
        
        self.placeOrder(self.order_id, contract, order)
        

    def sell(self, price, cant, action, contract):
        logger.info("=========================================================== Placing sell order: {} contracts"
              .format(round(cant, 5)))

        self.order_id = self.get_order_id()
        if action == "MARKET":
            order = self.market_order(self.action2, float(cant), self.account)
        elif action == "LIMIT":
            order = self.limit_order(self.action2, float(cant), self.min_price_increment(price), self.account)
        else:
            order = self.stop_limit_order(self.action2, float(cant), self.min_price_increment(price), self.account)
 
        self.placeOrder(self.order_id, contract, order)
    
    
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address')
    parser.add_argument('--port', type=int, default=7497, help='Port number')
    parser.add_argument("--contracts", type=str, default="{'ES':{'multiplier':50, 'region':'US', 'asset_class':'index'}}", help="Contract information")
    parser.add_argument("--symbol_list", type=str, default="['DAX']", help="Instruments list")
    parser.add_argument('--account', type=str, default='DU7774793', help='Account')
    parser.add_argument('--order_validity', type=str, default="DAY", help='The expiration time of the order: DAY or GTC')
    
    args = parser.parse_args()
    logger.info(f"args {args}")
    contracts = eval(args.contracts)
    symbol_list = eval(args.symbol_list)
    bot = BotZenitEXECUTE_ORDERS(args.ip, 
              args.port, 
              contracts, 
              symbol_list, 
              args.account,
              args.order_validity
              )
    try:
        bot.main()
    except Exception as e:
        print(e)
        bot.disconnect()
    

