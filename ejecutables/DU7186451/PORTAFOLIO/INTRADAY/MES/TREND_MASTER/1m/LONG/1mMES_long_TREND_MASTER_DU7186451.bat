
@echo off
title TREND_MASTER MES long 1m Paper Account 2
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file=zenit_strategy_bot.py

REM Parámetros para el script Python
set port=7497
set symbol=MESM4
set exchange=CME
set secType=FUT
set account=DU7186451
set client=50271
set is_paper=False
set order_type=MARKET
set interval=1m
set trading_class=MES
set multiplier=5
set accept_trade=ab
set lastTradeDateOrContractMonth=20240621
set quantity=100

REM Activar el entorno virtual
call %env_name%\Scripts\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth% 
