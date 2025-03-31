import random
import csv
import os

# Nombre del archivo CSV
CSV_FILE = 'generated_ids.csv'

def load_generated_ids():
    try:
        with open(CSV_FILE, 'r') as file:
            reader = csv.reader(file)
            return set(map(int, next(reader, [])))
    except FileNotFoundError:
        return set()

def save_generated_ids(generated_ids):
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(generated_ids))

def generate_unique_id(generated_ids):
    while True:
        new_id = random.randint(10, 10000)
        if new_id not in generated_ids:
            generated_ids.add(new_id)
            save_generated_ids(generated_ids)
            return new_id

# Ejemplo de uso
existing_ids = load_generated_ids()
# new_id = generate_unique_id(existing_ids)
# print(f"Nuevo ID generado: {new_id}")

def guardar_como_bat(nombre_archivo, contenido):
    """
    Guarda el contenido proporcionado como un archivo .bat.

    Parameters:
    - nombre_archivo (str): El nombre del archivo .bat.
    - contenido (str): El contenido que se almacenará en el archivo .bat.
    """
    # Extraer la ruta del archivo
    ruta_archivo = os.path.abspath(nombre_archivo)
    # Crear las carpetas necesarias si no existen
    carpeta = os.path.dirname(ruta_archivo)
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        print(f"Carpeta '{carpeta}' creada exitosamente.")
        
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(contenido)


account_info = {
    'INTRADAY': {
            'interval':['1m','15m', '5m']
        },
    'SWING': {
        'interval':['1h', '4h']
    },
   
    'DUH782121': {
        'symbol': ['ES', 'MES', 'MNQ', 'NQ'],
        'info_by_symbol': {
            'ES':{
                'trading_quantity':{
                                        'LONG':{
                                            'INTRADAY':'8',
                                            'SWING':'3'
                                        },
                                        'SHORT':{
                                            'INTRADAY':'8',
                                            'SWING':'3'
                                        }
                                        
                                    },
                'strategies': ['TA', 
                               'TREND_EMAS_CLOUD', 'TREND_MASTER', 'MACD_TREND']
                 },
            'MCL':{
                'trading_quantity':{
                                        'LONG':{
                                            'INTRADAY':'20',
                                            'SWING':'20'
                                        },
                                        'SHORT':{
                                            'INTRADAY':'10',
                                            'SWING':'10'
                                        }
                                        
                                    },
                    'strategies': ['TREND_MASTER', 'SCALPER_BOT', 'EMAS_CLOUD']
                 },
            'MES':{
                'trading_quantity':{
                                        'LONG':{
                                            'INTRADAY':'80',
                                            'SWING':'30'
                                        },
                                        'SHORT':{
                                            'INTRADAY':'40',
                                            'SWING':'16'
                                        }
                                        
                                    },
                'strategies': ['TA', 
                               'TREND_EMAS_CLOUD', 'TREND_MASTER', 'MACD_TREND', 'MACD_TREND']
            },
            'MGC':{
                'trading_quantity':{
                                        'LONG':{
                                            'INTRADAY':'20',
                                            'SWING':'20'
                                        },
                                        'SHORT':{
                                            'INTRADAY':'10',
                                            'SWING':'10'
                                        }
                                        
                                    },
                'strategies': ['TREND_MASTER', 'SCALPER_BOT', 'EMAS_CLOUD', 'MACD_TREND']
                 },
            'CL':{
                'trading_quantity':{
                                        'LONG':{
                                            'INTRADAY':'2',
                                            'SWING':'2'
                                        },
                                        'SHORT':{
                                            'INTRADAY':'1',
                                            'SWING':'1'
                                        }
                                        
                                    },
                'strategies': ['TREND_MASTER', 'SCALPER_BOT', 'EMAS_CLOUD']
                 },
            'MNQ':{
                'trading_quantity':{
                                        'LONG':{
                                            'INTRADAY':'40',
                                            'SWING':'20'
                                        },
                                        'SHORT':{
                                            'INTRADAY':'20',
                                            'SWING':'10'
                                        }
                                        
                                    },
                'strategies': ['TA', 
                               'TREND_EMAS_CLOUD', 'TREND_MASTER', 'MACD_TREND']                
            },
            'NQ':{
                'trading_quantity':{
                                        'LONG':{
                                            'INTRADAY':'4',
                                            'SWING':'2'
                                        },
                                        'SHORT':{
                                            'INTRADAY':'2',
                                            'SWING':'2'
                                        }
                                        
                                    },
                'strategies': ['TA', 'MACD_TREND'] 
            },
            'GC':{
                'trading_quantity':{
                                        'LONG':{
                                            'INTRADAY':'1',
                                            'SWING':'1'
                                        },
                                        'SHORT':{
                                            'INTRADAY':'1',
                                            'SWING':'1'
                                        }
                                        
                                    },
                'strategies': ['TREND_MASTER', 'SCALPER_BOT', 'EMAS_CLOUD']
                 },
        },
        'client_id': {
            'ES':{'long':'13',
                      'short': '14'
                      }, 
            'MCL':{'long':'15',
                      'short': '16'
                      }, 
            'MES':{'long':'17',
                      'short': '18'
                      }, 
            'MGC':{'long':'19',
                      'short': '20'
                      }, 
            'MNQ':{'long':'21',
                      'short': '22'
                      }, 
            'NQ':{'long':'23',
                      'short': '24'
                      }, 
            'CL':{'long':'25',
                      'short': '26'
                      }, 
            'GC': {'long':'27',
                      'short': '28'
                      }
        },
        'account': 'DUH782121',
        'alias': 'Jemirson Account',
        'is_paper': True,
        'port':7497,
        'ip': '127.0.0.1'
    }
}
symbols_info = {
    'ES': {
        'exchange': 'CME',
        'secType': 'FUT',
        'ContractMonth': '20240920',
        'symbol_ib': 'ESU4',
        'multiplier': '50',
        'client_id': '20',
        'str_contract_month':'SEP'
    }, 
     'MES': {
        'exchange': 'CME',
        'secType': 'FUT',
        'ContractMonth': '20240920',
        'symbol_ib': 'MESU4',
        'multiplier': '5',
        'client_id': '22',
        'str_contract_month':'SEP'
    }, 
     'NQ': {
        'exchange': 'CME',
        'secType': 'FUT',
        'ContractMonth': '20240920',
        'symbol_ib': 'NQU4',
        'multiplier': '20',
        'client_id': '25',
        'str_contract_month':'SEP'
    },
     'MNQ': {
        'exchange': 'CME',
        'secType': 'FUT',
        'ContractMonth': '20240920',
        'symbol_ib': 'MNQU4',
        'multiplier': '2',
        'client_id': '24',
        'str_contract_month':'SEP'
    }, 
     'CL': {
        'exchange': 'NYMEX',
        'secType': 'FUT',
        'ContractMonth': '20240820',
        'symbol_ib': 'CLQ4',
        'multiplier': '1000',
        'client_id': '26',
        'str_contract_month':'SEP'
    },
    'MCL': {
        'exchange': 'NYMEX',
        'secType': 'FUT',
        'ContractMonth': '20240819',
        'symbol_ib': 'MCLQ4',
        'multiplier': '100',
        'client_id': '21',
        'str_contract_month':'SEP'
    }, 
    'GC': {
        'exchange': 'COMEX',
        'secType': 'FUT',
        'ContractMonth': '20240828',
        'symbol_ib': 'GCQ4',
        'multiplier': '100',
        'client_id': '27',
        'str_contract_month':'AGO'
    },
    'MGC': {
        'exchange': 'COMEX',
        'secType': 'FUT',
        'ContractMonth': '20240828',
        'symbol_ib': 'MGCQ4',
        'multiplier': '10',
        'client_id': '23',
        'str_contract_month':'AGO'
    }, 
    'RTY': {
        'exchange': 'CME',
        'secType': 'FUT',
        'ContractMonth': '20240920',
        'symbol_ib': 'RTYU4',
        'multiplier': '50',
        'client_id': '29',
        'str_contract_month':'SEP'
    },
    'M2K': {
        'exchange': 'CME',
        'secType': 'FUT',
        'ContractMonth': '20240920',
        'symbol_ib': 'M2KU4',
        'multiplier': '5',
        'client_id': '28',
        'str_contract_month':'SEP'
    },
    
}


strategies_info = {
    'TREND_MASTER': {
        'file_name':'zenit_strategy_bot.py',
        'accept_trade': {
            'LONG':['long'],
            'SHORT': ['long']
        },
        'accept_trade_list': ['long'],
        'interval':['1m','5m', '15m', '1h']
    }, 
    'MACD_TREND': {
        'file_name':'zenit_MACDTREND_strategy.py',
        'accept_trade': {
            'LONG':['long'],
            'SHORT': ['long']
        },
        'accept_trade_list': ['long','short'],
        'interval':['1m','5m', '15m', '1h', '4h']
    }, 
    'SCALPER_BOT': {
        'file_name': 'zenit_EMAS_strategy.py',
        'accept_trade': {
            'LONG': ['long100'],
            'SHORT':['short100']
        },
        'accept_trade_list': ['long', 'short'],
        'interval':['1m','15m']
    }, 
    'EMAS_CLOUD': {
        'file_name': 'zenit_EMASCLOUD_strategies.py',
        'accept_trade': {
            'LONG':['long'],
            'SHORT':['short']
        },
        'accept_trade_list': ['long', 'short'],
        'interval':['5m', '15m']
    },
    'TREND_EMAS_CLOUD': {
        'file_name': 'zenit_TRENDEMASCLOUD_strategy.py',
        'accept_trade': {
            'LONG':['long'],
            'SHORT':['short']
        },
        'accept_trade_list': ['long', 'short'],
        'interval':['1m','5m', '15m','1h']
    },
    'TA': {
        'file_name': 'zenit_CRUCEEMAS_strategy.py',
        'accept_trade': {
            'LONG':['long'],
            'SHORT':['short']
        },
        'accept_trade_list': ['long', 'short'],
        'interval':['1m', '5m', '15m', '1h']
    }
}


accounts = ['DUH782121']
accounts1 = ['DU7186453']
trading_type = ['INTRADAY']
strategies = ['TREND_MASTER', 'SCALPER_BOT', 'EMAS_CLOUD', 'TREND_EMAS_CLOUD', 'TA', 'MACD_TREND']
trade_type = ['LONG', 'SHORT']




def jermison_strategy_content(
    strategy_name,
    symbol,
    interval,
    alias,
    filename,
    symbol_ib,
    exchange,
    sectype,
    account,
    cliend_id,
    multiplier,
    contract_date,
    quantity,
    accept_trade='long100',
    port=7496,
    is_paper=False
    ):
    if strategy_name == 'TREND_MASTER':
        string = f'''
@echo off
title {strategy_name} {symbol} {accept_trade} {interval} {alias}
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file={filename}

REM Parámetros para el script Python
set port={port}
set symbol={symbol_ib}
set exchange={exchange}
set secType={sectype}
set account={account}
set client={cliend_id}
set is_paper={is_paper}
set order_type=MARKET
set interval={interval}
set trading_class={symbol}
set multiplier={multiplier}
set accept_trade=ab
set lastTradeDateOrContractMonth={contract_date}
set quantity={quantity}

REM Activar el entorno virtual
call %env_name%\Scripts\\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth% 
'''
    elif strategy_name == 'SCALPER_BOT':
        string = f'''
@echo off
title {strategy_name} {symbol} {accept_trade} {interval} {alias}
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file={filename}

REM Parámetros para el script Python
set port={port}
set symbol={symbol_ib}
set exchange={exchange}
set secType={sectype}
set account={account}
set client={cliend_id}
set is_paper={is_paper}
set order_type=MARKET
set interval={interval}
set trading_class={symbol}
set multiplier={multiplier}
set accept_trade={accept_trade}
set lastTradeDateOrContractMonth={contract_date}
set quantity={quantity}

REM Activar el entorno virtual
call %env_name%\Scripts\\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth%
'''
    elif strategy_name in ['TREND_EMAS_CLOUD', 'TA']: 
        string = f'''
@echo off
title {strategy_name} {symbol} {accept_trade} {interval} {alias}
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file={filename}

REM Parámetros para el script Python
set port={port}
set symbol={symbol_ib}
set exchange={exchange}
set secType={sectype}
set account={account}
set client={cliend_id}
set is_paper={is_paper}
set order_type=MARKET
set interval={interval}
set trading_class={symbol}
set multiplier={multiplier}
set accept_trade={accept_trade}
set lastTradeDateOrContractMonth={contract_date}
set quantity={quantity}

REM Activar el entorno virtual
call %env_name%\Scripts\\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth% 
'''   
    else:
        string = f'''
@echo off
title {strategy_name} {symbol} {accept_trade} {interval} {alias}
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file={filename}

REM Parámetros para el script Python
set port={port}
set symbol={symbol_ib}
set exchange={exchange}
set secType={sectype}
set account={account}
set client={cliend_id}
set is_paper={is_paper}
set order_type=MARKET
set interval={interval}
set trading_class={symbol}
set multiplier={multiplier}
set accept_trade={accept_trade}
set lastTradeDateOrContractMonth={contract_date}
set quantity={quantity}
set trade_type=sf

REM Activar el entorno virtual
call %env_name%\Scripts\\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth% --trade_type %trade_type%
'''


    return string


def jermison_strategy_content_sh(
    strategy_name,
    symbol,
    interval,
    alias,
    filename,
    symbol_ib,
    exchange,
    sectype,
    account,
    cliend_id,
    multiplier,
    contract_date,
    quantity,
    accept_trade='long100',
    port=7496,
    is_paper=False
    ):
    if strategy_name == 'TREND_MASTER':
        string = f'''#!/bin/bash
# Nombre del entorno virtual
env_name=$HOME/zenit

# Ruta al archivo Python
python_file={filename}

# Parámetros para el script Python
port={port}
symbol={symbol_ib}
exchange={exchange}
secType={sectype}
account={account}
client={cliend_id}
is_paper={is_paper}
order_type=MARKET
interval={interval}
trading_class={symbol}
multiplier={multiplier}
quantity={quantity}
lastTradeDateOrContractMonth={contract_date}
quantity_type=noFixed

# Activar el entorno virtual
source $env_name/bin/activate

# Cambiar al directorio específico
cd $HOME/Documents/strategies

# Ejecutar el archivo Python con los parámetros especificados
python $python_file --port $port --quantity $quantity --symbol $symbol --exchange $exchange --secType $secType --account $account --client $client --is_paper $is_paper --order_type $order_type --interval $interval --trading_class $trading_class --multiplier $multiplier --lastTradeDateOrContractMonth $lastTradeDateOrContractMonth --quantity_type $quantity_type
'''
    elif strategy_name in ['TREND_EMAS_CLOUD', 'TA']: 
        string = f'''#!/bin/bash
# Nombre del entorno virtual
env_name=$HOME/zenit

# Ruta al archivo Python
python_file={filename}

# Parámetros para el script Python
port={port}
symbol={symbol_ib}
exchange={exchange}
secType={sectype}
account={account}
client={cliend_id}
is_paper={is_paper}
order_type=MARKET
interval={interval}
trading_class={symbol}
multiplier={multiplier}
accept_trade={accept_trade}
lastTradeDateOrContractMonth={contract_date}
quantity={quantity}

# Activar el entorno virtual
source $env_name/bin/activate

# Cambiar al directorio específico
cd $HOME/Documents/strategies

# Ejecutar el archivo Python con los parámetros especificados
python $python_file --port $port --quantity $quantity --symbol $symbol --exchange $exchange --secType $secType --accept_trade $accept_trade --account $account --client $client --is_paper $is_paper --order_type $order_type --interval $interval --trading_class $trading_class --multiplier $multiplier --lastTradeDateOrContractMonth $lastTradeDateOrContractMonth
'''   
    else:
        string = f'''#!/bin/bash
# Nombre del entorno virtual
env_name=$HOME/zenit

# Ruta al archivo Python
python_file={filename}

# Parámetros para el script Python
port={port}
symbol={symbol_ib}
exchange={exchange}
secType={sectype}
account={account}
client={cliend_id}
is_paper={is_paper}
order_type=MARKET
interval={interval}
trading_class={symbol}
multiplier={multiplier}
accept_trade={accept_trade}
lastTradeDateOrContractMonth={contract_date}
quantity={quantity}

# Activar el entorno virtual
source $env_name/bin/activate

# Cambiar al directorio específico
cd $HOME/Documents/strategies

# Ejecutar el archivo Python con los parámetros especificados
python $python_file --port $port --quantity $quantity --symbol $symbol --exchange $exchange --secType $secType --accept_trade $accept_trade --account $account --client $client --is_paper $is_paper --order_type $order_type --interval $interval --trading_class $trading_class --multiplier $multiplier --lastTradeDateOrContractMonth $lastTradeDateOrContractMonth
'''

    return string



teams_info = {
     
    'DUH782121':{
        'teams':[
                 'DTC_Beta_Live_Smart_TA',
                 'DTC_Beta_Live_Smart_Trend',
                 'DTC_Beta_Live_Smart',
                 'DTC_Beta_Live_Long',
                 'DTC_Beta_Live_Short']        
    }, 
    
}




#  meses = {
#         1: 'F',
#         2: 'G',
#         3: 'H',
#         4: 'J',
#         5: 'K',
#         6: 'M',
#         7: 'N',
#         8: 'Q',
#         9: 'U',
#         10: 'V',
#         11: 'X',
#         12: 'Z'
#     }