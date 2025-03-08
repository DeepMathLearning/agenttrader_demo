
@echo off
title EMAS_CLOUD NQ long 5m Paper Account 1
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file=zenit_EMASCLOUD_strategies.py

REM Parámetros para el script Python
set port=7497
set symbol=NQM4
set exchange=CME
set secType=FUT
set account=DU7186454
set client=39226
set is_paper=False
set order_type=MARKET
set interval=5m
set trading_class=NQ
set multiplier=20
set accept_trade=long
set lastTradeDateOrContractMonth=20240621
set quantity=4
set trade_type=sf

REM Activar el entorno virtual
call %env_name%\Scripts\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth% --trade_type %trade_type%
