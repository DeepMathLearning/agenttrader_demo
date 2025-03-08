
@echo off
title SCALPER_BOT MNQ long100 1m Paper Account 1
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file=zenit_EMAS_strategy.py

REM Parámetros para el script Python
set port=7497
set symbol=MNQM4
set exchange=CME
set secType=FUT
set account=DU7186454
set client=37196
set is_paper=False
set order_type=MARKET
set interval=1m
set trading_class=MNQ
set multiplier=2
set accept_trade=long100
set lastTradeDateOrContractMonth=20240621
set quantity=40

REM Activar el entorno virtual
call %env_name%\Scripts\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth%
