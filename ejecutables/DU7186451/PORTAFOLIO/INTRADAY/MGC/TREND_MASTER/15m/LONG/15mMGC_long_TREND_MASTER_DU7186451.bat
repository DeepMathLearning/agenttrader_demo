
@echo off
title TREND_MASTER MGC long 15m Paper Account 2
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file=zenit_strategy_bot.py

REM Parámetros para el script Python
set port=7497
set symbol=MGCF4
set exchange=COMEX
set secType=FUT
set account=DU7186451
set client=52293
set is_paper=False
set order_type=MARKET
set interval=15m
set trading_class=MGC
set multiplier=10
set accept_trade=ab
set lastTradeDateOrContractMonth=20240315
set quantity=0

REM Activar el entorno virtual
call %env_name%\Scripts\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth% 
