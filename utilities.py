import io
from PIL import Image
import base64
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import sqlite3
import subprocess
import multiprocessing
import os
import time
import signal
import psutil
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from zenit_CRUCEEMAS_strategy import BotZenitCRUCEEMAS
from zenit_TRENDEMASCLOUD_strategy import BotZenitTRENDEMASCLOUD 
from zenit_strategy_bot import BotZenitTrendMaster
import platform
from account_info import symbols_info
from datetime import datetime
from streamlit_calendar import calendar
from account_info import *
import random
import string

def generate_numeric_id(length=3):
    return int(''.join(random.choice('0123456789') for _ in range(length)))

os_platform = platform.platform()

def logo_login():
    # Cargar la imagen (asegúrate de que la imagen esté en el directorio correcto)
    image = Image.open("./zenit_logo_dor.png")
    cl, cl1, cl2= st.columns([4,4,4])
    # Mostrar la imagen en Streamlit
    with cl1:
        st.image(image, width=200)

def get_filtered_data_by_symbols(csv_path, symbols):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Filtrar los datos por los símbolos especificados
    filtered_df = df[df['symbol'].isin(symbols)]

    return filtered_df

def logo_up():
    file = open("./zenit_logo_dor.png", "rb")
    contents = file.read()
    img_str = base64.b64encode(contents).decode("utf-8")
    buffer = io.BytesIO()
    file.close()
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize((130, 100))  # x, y
    resized_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"""
        <style>
          .reportview-container {{
              margin-top: -2em;
          }}
          #MainMenu {{visibility: hidden;}}
          .stDeployButton {{display:none;}}
          footer {{visibility: hidden;}}
          #stDecoration {{display:none;}}
        </style>
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{img_b64}');
                background-repeat: no-repeat;
                padding-top: 50px;
                background-position: 100px 30px;
            }}
        </style>
        """

def team_made(args, 
              symbol, 
              trade, 
              quantity, 
              id, 
              df_symbols_editable,
              trading_type = 'day_traiding', 
              a_type = 'paper',
              smart_interval='auto'):
    
                
    df_sym_info = df_symbols_editable[df_symbols_editable['symbol'] == symbol].reset_index(drop=True).to_dict(orient='records')[0]
    df_sym_info['Multiplier'] = str(df_sym_info['Multiplier'])
    df_sym_info['ContractMonth'] = str(df_sym_info['ContractMonth'])
    
    if trading_type == 'day_traiding':
        if trade == 'long':
            list_command = [
                # TA Long
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}2 --smart_interval {smart_interval}",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}3 --smart_interval {smart_interval}",

                # TREND EMAS CLOUD Long
                #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}4",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}5 --smart_interval {smart_interval}",
                
                # TEND MASTER Long
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}6 --smart_interval {smart_interval}",
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7 --smart_interval {smart_interval}"
            ]
        elif trade == 'short':
            list_command = [
                #TA Short
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}8 --smart_interval {smart_interval}",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}9 --smart_interval {smart_interval}",

                #TREND EMAS CLOUD Short
                #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}10",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}11 --smart_interval {smart_interval}",
            ]
        elif (trade == 'smart') and (a_type != 'live'):
            list_command = [
                # TA Long
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}2 --with_trend_study True --smart_interval {smart_interval}",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}3 --with_trend_study True --smart_interval {smart_interval}",

                # TREND EMAS CLOUD Long
                #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}4 --with_trend_study True",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}5 --with_trend_study True --smart_interval {smart_interval}",
                
                # TEND MASTER Long
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}6 --with_trend_study True --smart_interval {smart_interval}",
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 15m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7 --with_trend_study True --smart_interval {smart_interval}"
            ]
        
        elif (trade == 'smart_1'):
            list_command = [
                # TA Long
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}2 --with_trend_study True --with_macd_signal True --smart_interval {smart_interval}",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}3 --with_trend_study True --with_macd_signal True --smart_interval {smart_interval}",

                # TREND EMAS CLOUD Long
                #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}4 --with_trend_study True",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}5 --with_trend_study True --smart_interval {smart_interval}",
                
                # TEND MASTER Long
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}6 --with_trend_study True --smart_interval {smart_interval}",
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 15m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7 --with_trend_study True --smart_interval {smart_interval}"
            ]
            
        elif (trade == 'smart') and (a_type == 'live'):
            list_command = [
                # TA Long
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity 3 --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}2 --with_trend_study True --smart_interval {smart_interval}",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity 3 --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}3 --with_trend_study True --smart_interval {smart_interval}",

                # TREND EMAS CLOUD Long
                #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}4 --with_trend_study True",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}5 --with_trend_study True --smart_interval {smart_interval}",
                
                # TEND MASTER Long
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}6 --with_trend_study True --smart_interval {smart_interval}",
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 15m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7 --with_trend_study True --smart_interval {smart_interval}"
            ]
        elif (trade == 'smart_ta'):
            list_command = [
                # TA Long
                #f'''python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}2 --with_trend_study True --smart_interval {smart_interval}''',
                f'''python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}3 --with_trend_study True --smart_interval {smart_interval}'''
            ]
        elif (trade == 'smart_trend') :
            if a_type == 'live':
                list_command = [
               
                # TREND EMAS CLOUD Long
                #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}4 --with_trend_study True",
                f'''python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}5 --with_trend_study True --smart_interval {smart_interval}''',
                
                # TEND MASTER Long
                # f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}6 --with_trend_study True",
                # f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 15m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7 --with_trend_study True"
            ]
            else:
                list_command = [
                
                    # TREND EMAS CLOUD Long
                    #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}4 --with_trend_study True",
                    f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade {trade} --hora_ejecucion {args['hora_ejecucion']} --client {id}5 --with_trend_study True --smart_interval {smart_interval}",
                    
                    # TEND MASTER Long
                    f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}6 --with_trend_study True --smart_interval {smart_interval}",
                    f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 15m --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7 --with_trend_study True --smart_interval {smart_interval}"
                ]
    else:
        if trade == 'long':
            list_command = [
                # TA Long
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}3 --smart_interval {smart_interval}",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}4 --smart_interval {smart_interval}",

                # TREND EMAS CLOUD Long
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}5 --smart_interval {smart_interval}",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}6 --smart_interval {smart_interval}",
                # f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}6",
                
                # TEND MASTER Long
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1h --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7 --smart_interval {smart_interval}",
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}8 --smart_interval {smart_interval}",
                # f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}9",
            ]
        elif trade == 'short':
            list_command = [
                #TA Short
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}9 --smart_interval {smart_interval}",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}10 --smart_interval {smart_interval}",

                #TREND EMAS CLOUD Short
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}11 --smart_interval {smart_interval}",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}12 --smart_interval {smart_interval}",
            ]
        elif trade == 'smart':
            list_command = [
                # TA Long
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade smart --hora_ejecucion {args['hora_ejecucion']} --client {id}3 --with_trend_study True --smart_interval {smart_interval}",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade smart --hora_ejecucion {args['hora_ejecucion']} --client {id}4 --with_trend_study True --smart_interval {smart_interval}",

                # TREND EMAS CLOUD Long
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade smart --hora_ejecucion {args['hora_ejecucion']} --client {id}5 --with_trend_study True --smart_interval {smart_interval}",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade smart --hora_ejecucion {args['hora_ejecucion']} --client {id}6 --with_trend_study True --smart_interval {smart_interval}",
                # f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}6",
                
                # TEND MASTER Long
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1h --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7 --with_trend_study True --smart_interval {smart_interval}",
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}8 --with_trend_study True --smart_interval {smart_interval}",
                # f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 4h --quantity {quantity} --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}9",
            ]
        
    return list_command

### Nueva funcion para actualizar posteriormente
# def team_made(args, symbol, trade, quantity, id, df_symbols_editable, 
#               trading_type='day_trading', a_type='paper', smart_interval='auto'):
    
#     # Get the symbol information
#     df_sym_info = df_symbols_editable[df_symbols_editable['symbol'] == symbol].iloc[0]
#     df_sym_info['Multiplier'] = str(df_sym_info['Multiplier'])
#     df_sym_info['ContractMonth'] = str(df_sym_info['ContractMonth'])

#     # Define the base command template
#     base_command = (
#         "python {script} --ip {ip} --symbol {symbol_ib} --exchange {exchange} "
#         "--secType {secType} --trading_class {trading_class} --multiplier {Multiplier} "
#         "--lastTradeDateOrContractMonth {ContractMonth} --is_paper False --interval {interval} "
#         "--quantity {quantity} --account {account} --port {port} --accept_trade {trade} "
#         "--hora_ejecucion {hora_ejecucion} --client {client} --smart_interval {smart_interval}"
#     )

#     # Determine which scripts to run based on trade type and strategy
#     scripts = []
#     if trading_type == 'day_trading':
#         if trade in ['long', 'short']:
#             scripts = [
#                 ("zenit_CRUCEEMAS_strategy.py", ['1m', '5m']),
#                 ("zenit_TRENDEMASCLOUD_strategy.py", ['5m']),
#                 ("zenit_strategy_bot.py", ['1m', '5m']),
#             ]
#         elif trade == 'smart' and a_type != 'live':
#             scripts = [
#                 ("zenit_CRUCEEMAS_strategy.py", ['1m', '5m']),
#                 ("zenit_TRENDEMASCLOUD_strategy.py", ['5m']),
#                 ("zenit_strategy_bot.py", ['5m', '15m']),
#             ]
#     # Generate commands
#     list_command = []
#     for script, intervals in scripts:
#         for i, interval in enumerate(intervals, start=2):
#             command = base_command.format(
#                 script=script, ip=args['ip'], symbol_ib=df_sym_info['symbol_ib'], 
#                 exchange=df_sym_info['exchange'], secType=df_sym_info['secType'], 
#                 trading_class=symbol, Multiplier=df_sym_info['Multiplier'], 
#                 ContractMonth=df_sym_info['ContractMonth'], interval=interval, 
#                 quantity=quantity, account=args['account'], port=args['port'], 
#                 trade=trade, hora_ejecucion=args['hora_ejecucion'], 
#                 client=f"{id}{i}", smart_interval=smart_interval
#             )
#             if trade == 'smart':
#                 command += " --with_trend_study True"
#             list_command.append(command)

#     return list_command



#@st.cache_data
def get_chart_49243206():
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x='year', y='pop')

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

# CSV file to store process information
PROCESS_FILE = "data/custom_processes.csv"

DATABASE_FILE = 'processes.db'
TABLE_NAME = 'processes'


# Function to load processes from CSV
# def load_processes():
#     if os.path.exists(DATABASE_FILE):
#         try:
#             conn = sqlite3.connect(DATABASE_FILE)
#             df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
#             conn.close()
#             return df.to_dict(orient='records')
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return []
#     else:
#         return []


# Function to load processes from CSV
def load_processes():
    if os.path.exists(PROCESS_FILE):
        try:
            df = pd.read_csv(PROCESS_FILE).to_dict(orient='records')
            return df
        except:
             return [] 
    else:
        return []

# Function to save processes to CSV
def save_processes(processes):
    df = pd.DataFrame(processes)
    df.to_csv(PROCESS_FILE, index=False)


def run_custom_commands_single(commands):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    new_processes = []
    for i, cmd in enumerate(commands):
        pro = round(((i+1)/ len(commands)), 2)
        status_text.text(f"{round(float(pro)*100)}% Complete")
        progress_bar.progress(pro)
        if 'Windows' in os_platform:
            process = subprocess.Popen(cmd, 
                                    shell=True, 
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, 
                                    #    stdout=subprocess.PIPE,
                                    #    stderr=subprocess.PIPE,
                                    text=True)
        else:
            process = subprocess.Popen(cmd, 
                                    shell=True, 
                                    preexec_fn=os.setsid, 
                                    #    stdout=subprocess.PIPE,
                                    #    stderr=subprocess.PIPE,
                                    text=True)
        time.sleep(3)
       # Verificar si el PID existe
        pid = process.pid
        try:
            os.killpg(os.getpgid(pid), 0)
            #os.kill(pid, 0)  # Señal 0 no mata el proceso
            pid_exists = True
        except:
            pid_exists = False

        if not pid_exists:
            raise RuntimeError(f"Error: el proceso con PID {pid} no se pudo iniciar correctamente.")
        
        args = extract_args(cmd)
        if "zenit_TRENDEMASCLOUD_strategy.py" in cmd:
            args['strategy'] = "TREND_EMAS_CLOUD"
        elif "zenit_CRUCEEMAS_strategy.py" in cmd:
            args['strategy'] = "TA"
        elif "zenit_strategy_bot.py" in cmd:
            args['strategy'] = "TREND_MASTER"
        elif "zenit_MACDTREND_strategy.py" in cmd:
            args['strategy'] = "MACD_TREND"

        new_processes.append({
            "command": cmd,
            "pid": process.pid,
            "account": args["account"],
            "symbol": args["trading_class"],
            "size_contracts":  args["quantity"],
            "client": args["client"],
            "strategy": args["strategy"],
            "trade_type": args["accept_trade"],
            "interval": args["interval"],
            "hora_ejecucion": args["hora_ejecucion"]
        })
    
    processes = load_processes() + new_processes
    save_processes(processes)

def run_command(cmd):
    os_platform = os.uname().sysname
    if 'Windows' in os_platform:
        process = subprocess.Popen(cmd,
                                   shell=True,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                   text=True)
    else:
        process = subprocess.Popen(cmd,
                                   shell=True,
                                   preexec_fn=os.setsid,
                                   text=True)
    time.sleep(3)
    
    pid = process.pid
    try:
        os.killpg(os.getpgid(pid), 0)
        pid_exists = True
    except:
        pid_exists = False

    if not pid_exists:
        raise RuntimeError(f"Error: el proceso con PID {pid} no se pudo iniciar correctamente.")

    args = extract_args(cmd)
    if "zenit_TRENDEMASCLOUD_strategy.py" in cmd:
        args['strategy'] = "TREND_EMAS_CLOUD"
    elif "zenit_CRUCEEMAS_strategy.py" in cmd:
        args['strategy'] = "TA"
    elif "zenit_strategy_bot.py" in cmd:
        args['strategy'] = "TREND_MASTER"
    elif "zenit_MACDTREND_strategy.py" in cmd:
        args['strategy'] = "MACD_TREND"
    
    if args["accept_trade"] in ['smart_ta', 'smart_trend', 'smart_1']:
        args["accept_trade"] = 'smart'

    return {
        "command": cmd,
        "pid": process.pid,
        "account": args["account"],
        "symbol": args["trading_class"],
        "size_contracts": args["quantity"],
        "client": args["client"],
        "strategy": args["strategy"],
        "trade_type": args["accept_trade"],
        "interval": args["interval"],
        "hora_ejecucion": args["hora_ejecucion"]
    }

def run_custom_commands(commands):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    new_processes = []

    with ThreadPoolExecutor() as executor:
        future_to_cmd = {executor.submit(run_command, cmd): cmd for cmd in commands}
        futures = list(future_to_cmd.keys())
        print(futures)
        for i, future in enumerate(as_completed(future_to_cmd)):
            cmd = future_to_cmd[future]
            print(cmd)
            try:
                result = future.result()
                new_processes.append(result)
            except Exception as exc:
                st.error(f"{cmd} generated an exception: {exc}")
            
            pro = round(((i + 1) / len(commands)), 2)
            status_text.text(f"{round(float(pro) * 100)}% Complete")
            progress_bar.progress(pro)
            # Agregar un retraso de 2 segundos
            time.sleep(2)
            
    processes = load_processes() + new_processes
    save_processes(processes)




# args = {
#     'symbol': symbol,
#     'exchange': exchange,
#     'secType': secType,
#     'client': client,
#     'trading_class': trading_class,
#     'multiplier': multiplier,
#     'lastTradeDateOrContractMonth': contract_date,
#     'is_paper': is_paper,
#     'interval': interval,
#     'quantity': quantity,
#     'account': account,
#     'accept_trade': accept_trade,
#     'port': port,
#     'ip':'127.0.0.1',
#     'currency':'USD'
# }


# Function to run custom commands and store process information
# def run_custom_commands(commands, args):
#     new_processes = []
#     for cmd in commands:
#         try:
#         # Capturar la salida estándar y de error
#             process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid, text=True)
#             stdout, stderr = process.communicate()

#             # Verificar si el proceso se ha iniciado correctamente
#             if process.returncode != 0:
#                 raise RuntimeError(f"Error: el comando '{cmd}' no pudo iniciarse. \nError: {stderr}")
            
#             new_processes.append({
#                 "command": cmd,
#                 "pid": process.pid,
#                 "account": args["account"],
#                 "symbol": args["trading_class"],
#                 "size_contracts":  args["quantity"],
#                 "client": args["client"]
#             })

#             # Mostrar la salida del comando
#             st.write(f"Resultado del comando '{cmd}':\n{stdout}")

#         except Exception as e:
#             st.error(f"Error al iniciar el comando '{cmd}': {e}")
#             continue
#     processes = load_processes() + new_processes
#     save_processes(processes)

    # Verificar si hay procesos nuevos añadidos y mostrar un mensaje apropiado
    # if new_processes:
    #     st.success('Comandos ejecutados con éxito.')
    # else:
    #     st.error('No se pudo ejecutar ninguno de los comandos proporcionados.')

# Function to kill a custom process
def kill_custom_process(pid):
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except:
        st.error("Este bot nunca se ejecutó, pudo deberse a una mala configuración")

def kill_custom_process_threads(bot):
    #try:
    bot.disconnect()
    return f"Process terminated successfully."
    # except:
    #     return f"Process {pid} does not exist."

# Function to get system processes information
def get_system_processes_info():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        try:
            process_info = proc.info
            processes.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

# Function to kill a system process
def kill_system_process(pid):
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=3)  # Wait for the process to terminate
        return f"Process {pid} terminated successfully."
    except psutil.NoSuchProcess:
        return f"Process {pid} does not exist."
    except psutil.AccessDenied:
        return f"Access denied to terminate process {pid}."
    except psutil.TimeoutExpired:
        return f"Process {pid} termination timed out."


def run_bot(strategy, kwargs):
    strategy_mapping = {
        'TREND_MASTER': BotZenitTrendMaster,
        'TREND_EMAS_CLOUD': BotZenitTRENDEMASCLOUD,
        'TA': BotZenitCRUCEEMAS
    }
    
    bot_class = strategy_mapping[strategy]
    bot = bot_class(**kwargs)
    bot.main()


@st.experimental_dialog("Bot Activity", width="large")
def bot_activity(html_link):
    try:
        with open(html_link, 'r', encoding='utf-8') as open_html:
            source_code = open_html.read()
            # Añadir un contenedor div con scroll en el HTML
            html_with_scroll = f"""
            <div style="height: 80%; width: 100%; overflow: auto; background-color: white; position: relative;">
                <div id="content" style="transform: scale(0.487); transform-origin: top left; width: 100%; height: 100%;">
                    {source_code}
                    <div style="position: absolute; top: 10px; right: 10px; background: rgba(255, 255, 255, 0.8); border: 1px solid #ccc; padding: 5px;">
                        <button style='height: 50px; width: 50px;' onclick="document.getElementById('content').style.transform = 'scale(0.45)';">45%</button>
                        <button style='height: 50px; width: 50px;' onclick="document.getElementById('content').style.transform = 'scale(0.75)';">75%</button>
                        <button style='height: 50px; width: 50px;' onclick="document.getElementById('content').style.transform = 'scale(1.5)';">150%</button>
                        <button style='height: 50px; width: 50px;' onclick="document.getElementById('content').style.transform = 'scale(2)';">200%</button>
                    </div>
                </div>
                
            </div>
            """
            components.html(html_with_scroll, height=800, width=1500)
    except:
        st.warning("Espera unos minutos que se genere la grafica del bot")
    # st.write(f"Why is {item} your favorite?")
    # reason = st.text_input("Because...")
    # if st.button("Submit"):
    #     st.session_state.vote = {"item": item, "reason": reason}
    #     st.rerun()
        

def extract_args(command_str):
    # Definición del string
    
    # Elimina el comando inicial y los espacios innecesarios
    args_str = command_str.replace('python zenit_TRENDEMASCLOUD_strategy.py ', '').replace('python zenit_CRUCEEMAS_strategy.py ', '').replace('python zenit_strategy_bot.py ', '').replace('python zenit_MACDTREND_strategy.py ', '').strip()

    # Divide la cadena en argumentos
    args_list = args_str.split()

    # Crear un objeto de análisis de argumentos
    parser = argparse.ArgumentParser()

    # Añadir los argumentos al parser
    parser.add_argument('--symbol')
    parser.add_argument('--exchange')
    parser.add_argument('--secType')
    parser.add_argument('--client', type=int)
    parser.add_argument('--trading_class')
    parser.add_argument('--multiplier', type=int)
    parser.add_argument('--lastTradeDateOrContractMonth')
    parser.add_argument('--is_paper', type=bool)
    parser.add_argument('--interval')
    parser.add_argument('--quantity', type=int)
    parser.add_argument('--account')
    parser.add_argument('--port')
    parser.add_argument('--accept_trade')
    parser.add_argument('--hora_ejecucion')
    parser.add_argument('--ip')
    parser.add_argument('--with_trend_study')
    parser.add_argument('--with_macd_signal')
    parser.add_argument('--smart_interval')
    # Analizar los argumentos
    args = parser.parse_args(args_list)

    args_dict = vars(args)

    # Mostrar los argumentos como diccionario
    return args_dict


def teams_commands(args, team, df_symbols_editable):
    
    df_sym_info = df_symbols_editable[df_symbols_editable['symbol'] == 'ES'].reset_index(drop=True).to_dict(orient='records')[0]
    df_sym_info['Multiplier'] = str(df_sym_info['Multiplier'])
    df_sym_info['ContractMonth'] = str(df_sym_info['ContractMonth'])
    
    las_contract = '20240920'
    if team == 'DTC_Beta_Live':
        commands = [
            # TA MES Long
            f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol MESU4 --exchange CME --secType FUT --client 2 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth {las_contract} --is_paper False --interval 1m --quantity 30 --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']}",
            f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol MESU4 --exchange CME --secType FUT --client 3 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth {las_contract} --is_paper False --interval 5m --quantity 30 --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']}",

            # TREND EMAS CLOUD MES Long
            f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol MESU4 --exchange CME --secType FUT --client 4 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth {las_contract} --is_paper False --interval 5m --quantity 40 --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']}",
            
            # TEND MASTER MES Long
            #f"python zenit_strategy_bot.py --accept_trade ab --symbol MESM4 --exchange CME --secType FUT --client 5 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth {las_contract} --is_paper False --interval 1m --quantity 30 --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']}",
            f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol MESM4 --exchange CME --secType FUT --client 6 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth {las_contract} --is_paper False --interval 5m --quantity 40 --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']}",
            #f"python zenit_strategy_bot.py --accept_trade ab --symbol MESM4 --exchange CME --secType FUT --client 7 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth {las_contract} --is_paper False --interval 15m --quantity 30 --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']}",

            #TA ES Short
            f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol ESU4 --exchange CME --secType FUT --client 8 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 1m --quantity 3 --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']}",
            f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol ESU4 --exchange CME --secType FUT --client 9 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 5m --quantity 3 --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']}",

            #TREND EMAS CLOUD ES Short
            f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol ESU4 --exchange CME --secType FUT --client 10 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 5m --quantity 4 --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']}",

        ]
    elif team == 'DTC_Beta_Live_Smart':
        commands = []
        for add in [
            ('ES', 'smart', 4,1),
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3], df_symbols_editable,trading_type = 'day_traiding',a_type='live')
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Live_Long':
        commands = [
            # TA Long
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class ES --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity 8 --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}2",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class ES --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity 8 --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}3",

                # TREND EMAS CLOUD Long
                #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}4",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class ES --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity 12 --account {args['account']} --port {args['port']} --accept_trade long --hora_ejecucion {args['hora_ejecucion']} --client {id}5",
                
                # TEND MASTER Long
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class ES --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity 12 --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}6",
                f"python zenit_strategy_bot.py --ip {args['ip']} --accept_trade ab --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class ES --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 15m --quantity 12 --account {args['account']} --port {args['port']} --hora_ejecucion {args['hora_ejecucion']} --client {id}7"
        ]
    elif team == 'DTC_Beta_Live_Short':
        commands = [
                #TA Short
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class ES --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity 8 --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}8",
                f"python zenit_CRUCEEMAS_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class ES --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity 8 --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}9",

                #TREND EMAS CLOUD Short
                #f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class {symbol} --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 1m --quantity {quantity} --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}10",
                f"python zenit_TRENDEMASCLOUD_strategy.py --ip {args['ip']} --symbol {df_sym_info['symbol_ib']} --exchange {df_sym_info['exchange']} --secType {df_sym_info['secType']} --trading_class ES --multiplier {df_sym_info['Multiplier']} --lastTradeDateOrContractMonth {df_sym_info['ContractMonth']} --is_paper False --interval 5m --quantity 12 --account {args['account']} --port {args['port']} --accept_trade short --hora_ejecucion {args['hora_ejecucion']} --client {id}11",
            ]
    elif team == 'DTC_Beta_Paper':
        commands = []
        for add in [
            ('ES', 'short', 24,1),
            ('MES', 'long', 240,2),
            ('NQ', 'short', 6,3),
            ('MNQ', 'long', 60,4),
            ('M2K', 'long', 140,5),
            ('RTY', 'short', 14,6),
            ('GC', 'short', 5,7),
            ('MGC', 'long', 50,8),
            ('CL', 'short', 3,9),
            ('MCL', 'long', 30,10)
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3], df_symbols_editable)
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Paper_Smart':
        commands = []
        for add in [
            ('ES', 'smart', 24,generate_numeric_id(length=3)),
            ('NQ', 'smart', 6,generate_numeric_id(length=3)),
            ('RTY', 'smart', 14,generate_numeric_id(length=3)),
            ('GC', 'smart', 5,generate_numeric_id(length=3)),
            ('CL', 'smart', 3,generate_numeric_id(length=3))
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable)
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Paper_Smart_1':
        commands = []
        for add in [
            ('ES', 'smart_1', 24,generate_numeric_id(length=3)),
            ('NQ', 'smart_1', 6,generate_numeric_id(length=3)),
            ('RTY', 'smart_1', 14,generate_numeric_id(length=3)),
            ('GC', 'smart_1', 5,generate_numeric_id(length=3)),
            ('CL', 'smart_1', 3,generate_numeric_id(length=3))
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable)
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Paper_Smart_TA':
        commands = []
        for add in [
            ('ES', 'smart_ta', 8,generate_numeric_id(length=3)),
            ('NQ', 'smart_ta', 2,generate_numeric_id(length=3)),
            ('RTY', 'smart_ta', 5,generate_numeric_id(length=3)),
            ('GC', 'smart_ta', 2,generate_numeric_id(length=3)),
            ('CL', 'smart_ta', 1,generate_numeric_id(length=3))
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable)
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Live_Smart_TA':
        commands = []
        for add in [
            ('ES', 'smart_ta', 8,generate_numeric_id(length=3), 'auto'),
            ('RTY', 'smart_ta',5,generate_numeric_id(length=3), 'auto'),
            ('NQ', 'smart_ta', 2,generate_numeric_id(length=3), 'auto')
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable)
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Paper_Smart_Trend':
        commands = []
        for add in [
            ('ES', 'smart_trend', 20,generate_numeric_id(length=3)),
            ('NQ', 'smart_trend', 6 ,generate_numeric_id(length=3)),
            ('RTY', 'smart_trend', 14 ,generate_numeric_id(length=3)),
            ('GC', 'smart_trend', 5 ,generate_numeric_id(length=3)),
            ('CL', 'smart_trend', 3 ,generate_numeric_id(length=3))
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable)
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Live_Smart_Trend':
        commands = []
        for add in [
            ('ES', 'smart_trend',  20 ,generate_numeric_id(length=3), 'auto'),
            ('RTY', 'smart_trend', 14 ,generate_numeric_id(length=3), 'auto'),
            ('NQ', 'smart_trend', 6 ,generate_numeric_id(length=3), 'auto'),
            ('GC', 'smart_trend', 5 ,generate_numeric_id(length=3), 'auto'),
            ('CL', 'smart_trend', 3 ,generate_numeric_id(length=3), 'auto')
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable, a_type='live', smart_interval=add[4])
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Paper_Long':
        commands = []
        for add in [
            ('MES', 'long', 240,generate_numeric_id(length=3)),
            ('MNQ', 'long', 60,generate_numeric_id(length=3)),
            ('M2K', 'long', 140,generate_numeric_id(length=3)),
            ('MGC', 'long', 50,generate_numeric_id(length=3)),
            ('MCL', 'long', 30,generate_numeric_id(length=3))
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable)
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Paper_Short':
        commands = []
        for add in [
            ('ES', 'short', 24,1),
            ('NQ', 'short', 6,3),
            ('RTY', 'short', 14,6),
            ('GC', 'short', 5,7),
            ('CL', 'short', 3,9)
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable)
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Paper_Swing':
        commands = []
        for add in [
            ('ES', 'short', 7,11),
            ('MES', 'long', 70,12),
            ('NQ', 'short', 3,13),
            ('MNQ', 'long', 30,14),
            ('M2K', 'long', 40,15),
            ('RTY', 'short', 14,16),
            ('GC', 'short', 3,17),
            ('MGC', 'long', 30,18),
            ('CL', 'short', 3,19),
            ('MCL', 'long', 30,20)
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable, 'swing_traiding')
            commands.extend(list_cmd)
    elif team == 'DTC_Beta_Paper_Swing_Smart':
        commands = []
        for add in [
            ('ES', 'smart', 7,generate_numeric_id(length=3)),
            ('NQ', 'smart', 3,generate_numeric_id(length=3)),
            ('RTY', 'smart', 4,generate_numeric_id(length=3)),
            ('GC', 'smart', 3,generate_numeric_id(length=3)),
            ('CL', 'smart', 3,generate_numeric_id(length=3))
        ]:
            list_cmd = team_made(args, add[0], add[1], add[2], add[3],df_symbols_editable, 'swing_traiding')
            commands.extend(list_cmd)
           
    
    return commands



def fetch_all_records(db_name='zenit_oms.db'):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query('SELECT * FROM activity', conn)
    conn.close()
    return df

def calculate_trade_metrics(df):
    trade_metrics = []

    for trade_id, group in df.groupby('trade_id'):
        # Determine trade type
        trade_type = group.iloc[0]['trade_type']
        
        if trade_type in ['long', 'ab']:
            entry_rows = group[group['action'] == 'Buy']
            exit_rows = group[group['action'] == 'Sell']
        else:
            entry_rows = group[group['action'] == 'Sell']
            exit_rows = group[group['action'] == 'Buy']
            
        # Calculate weighted average entry and exit prices
        entry_price = (entry_rows['price'] * entry_rows['contracts']).sum() / entry_rows['contracts'].sum()
        exit_price = (exit_rows['price'] * exit_rows['contracts']).sum() / exit_rows['contracts'].sum()
        
        # Calculate PnL
        if trade_type in ['long', 'ab']:
            pnl = (exit_price - entry_price)
        else:  # short
            pnl = (entry_price - exit_price)
        
        try:
            time_trade = exit_rows.iloc[0]['time']
        except:
            time_trade = entry_rows.iloc[0]['time']
        
        if pnl > 0:
            trade_result = 'Win'
        else:
            trade_result = 'Loss'
        
        # Append metrics
        trade_metrics.append({
            'trade_id': trade_id,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'contracts': entry_rows['contracts'].sum(),
            'strategy': group.iloc[0]['strategy'],
            'interval': group.iloc[0]['interval'],
            'symbol': group.iloc[0]['symbol'],
            'trade_type': trade_type,
            'account': group.iloc[0]['account'],
            'time': time_trade,
            'trade_result': trade_result
        })
    
    return pd.DataFrame(trade_metrics)

@st.experimental_dialog("Calendario", width="large")
def calendar_events_create(df, mode, calendar_options):
    df['time_modify'] = list(map(lambda x: str(x).replace(' ', 'T').replace('+00:00', ''), df['time']))
    
    df['time_dt'] = pd.to_datetime(df['time'])

    # Formatear solo la fecha
    df['time_dt'] = list(map(lambda x: x.strftime("%Y-%m-%d"), df['time_dt']))
    
    # df['time'] = pd.to_datetime(df['time'])

    # # Extraer la hora del día de la columna de tiempo
    # df['hour'] = df['time'].dt.hour
    i_account = df['account'].unique()
    
    if len(i_account) == 1:
        df = df[df['account'] == i_account[0]]
        msg = f'Calendario de trades de la cuenta {i_account[0]}'
    else:
        msg = 'Calendario general de trades'
    
    events = []
    for date in df['time_dt'].unique():
        df_ = df[df['time_dt'] == date].reset_index(drop=True)
        saldo = round(df_['saldo_usd'].sum(),2)
        if saldo > 0:
            color = '#3DD56D'
        else:
            color = '#FF6C6C'
        event = {
                "title": f"{saldo}$",
                "color": f"{color}",
                "start": f"{date}",
                "end": f"{date}",
                "resourceId": "a",
        }
        events.append(event)
        try:
            events.append({
            "title": f"{len(df_)} Trades",
            "color": "#1E90FF",
            "start": f"{df_['time_dt'][0]}",
            "end": f"{df_['time_dt'][-1]}",
            "resourceId": "a",
            })
        except:
            events.append({
            "title": f"{len(df_)} Trades",
            "color": "#1E90FF",
            "start": f"{df_['time_dt'][0]}",
            "end": f"{df_['time_dt'][0]}",
            "resourceId": "a",
            })
            
    st.write(msg)
    
    state = calendar(
        events=events,
        options=calendar_options,
        custom_css="""
        .fc-event-past {
            opacity: 0.8;
        }
        .fc-event-time {
            font-style: italic;
        }
        .fc-event-title {
            font-weight: 700;
        }
        .fc-toolbar-title {
            font-size: 2rem;
        }
        """,
        key=mode,
    )
    if state.get("eventsSet") is not None:
        st.session_state["events"] = state["eventsSet"]
        
    if state.get("eventClick") is not None:   
        date_str = str(state["eventClick"]['event']['start'])
        df = df[df['time_dt'] == date_str]
        st.dataframe(df[['account', 'strategy', 'interval', 'trade_type', 'symbol', 'saldo_usd','entry_price','exit_price', 'pnl', 'time', 'contracts']], 
                    hide_index=True, 
                    height=450, 
                    use_container_width=True)
        #st.write(state["eventClick"]['event']['start'])

def prepare_data_for_plot(df):
    df['time_dt'] = pd.to_datetime(df['time']).dt.date
    daily_stats = df.groupby('time_dt').agg(
        total_trades=('trade_id', 'count'),
        total_saldo=('saldo_usd', 'sum')
    ).reset_index()
    return daily_stats



def create_and_fill_symbols_info_table(csv_path, output_csv_path="futures_symbol_info_v1.csv"):
    # Verificar si el archivo de salida ya existe
    if not os.path.exists(output_csv_path):
        # Leer el archivo CSV original
        df = pd.read_csv(csv_path)
        
        # Crear un DataFrame con la información de los símbolos
        df_symbol = pd.DataFrame(symbols_info).T.reset_index()
        df_symbol.rename(columns={'index': 'Symbol'}, inplace=True)
        
        # Realizar la fusión de los DataFrames
        df = df.merge(df_symbol[['Symbol', 'secType', 'symbol_ib', 'ContractMonth', 'str_contract_month']], on='Symbol', how='left')
        df.rename(columns={'Symbol': 'symbol', 'Exchange': 'exchange'}, inplace=True)
        df = df.reset_index().rename(columns={'index': 'ID'})
        df['Multiplier'] = df['Multiplier'].fillna(0).astype(int)
        df['ContractMonth'] = df['ContractMonth'].fillna(0).astype(int).astype(str)
        # Guardar el DataFrame resultante en un nuevo archivo CSV
        df.to_csv(output_csv_path, index=False)
       
    


def update_contract_month(csv_path, symbol_value, symbol_ib, str_contract_month, new_contract_month):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Actualizar el valor de 'ContractMonth' en la fila con el símbolo especificado
    df.loc[df['symbol'] == symbol_value, ['ContractMonth', 'symbol_ib', 'str_contract_month']] = [new_contract_month, symbol_ib, str_contract_month]

    # Guardar los cambios en el archivo CSV
    df.to_csv(csv_path, index=False)

def generar_simbolo_futuro(simbolo, month, fecha_contrato):
    # Mapeo de meses y códigos de mes
    meses = {
        'ENE' : 'F',
        'FEB' : 'G',
        'MAR' : 'H',
        'APR' : 'J',
        'MAY' : 'K',
        'JUN' : 'M',
        'JUL' : 'N',
        'AGO' : 'Q',
        'SEP' : 'U',
        'OCT' : 'V',
        'NOV' : 'X',
        'DEC' : 'Z'
    }
    
    # Extraer el mes y el año de la fecha de contrato
    year = fecha_contrato[0:4]

    # Obtener el código de mes correspondiente
    codigo_mes = meses.get(month, 'H')  # Usamos 'H' como valor predeterminado

    # Construir el símbolo de futuro
    futuro = simbolo + codigo_mes + year[-1]

    return futuro
