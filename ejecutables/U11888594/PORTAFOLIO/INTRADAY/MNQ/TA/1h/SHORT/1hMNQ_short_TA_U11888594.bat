
@echo off
title TA MNQ short 1h Jemirson Account
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file=zenit_CRUCEEMAS_strategy.py

REM Parámetros para el script Python
set port=7496
set symbol=MNQM4
set exchange=CME
set secType=FUT
set account=U11888594
set client=22111
set is_paper=False
set order_type=MARKET
set interval=1h
set trading_class=MNQ
set multiplier=2
set accept_trade=short
set lastTradeDateOrContractMonth=20240621
set quantity=20

REM Activar el entorno virtual
call %env_name%\Scripts\activate


REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Ejecutar el archivo Python con los parámetros especificados
python %python_file% --port %port% --quantity %quantity% --symbol %symbol% --exchange %exchange% --secType %secType% --accept_trade %accept_trade% --account %account% --client %client% --is_paper %is_paper% --order_type %order_type% --interval %interval% --trading_class %trading_class% --multiplier %multiplier% --lastTradeDateOrContractMonth %lastTradeDateOrContractMonth% 
