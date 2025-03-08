@echo off
title ZENIT BOT MICROSERVICE
REM Nombre del entorno virtual
set env_name=%userprofile%\zenit

REM Ruta al archivo Python
set python_file=Crear_Bot.py

REM Activar el entorno virtual
call %env_name%\Scripts\activate

REM Ruta especifica 
cd %userprofile%\Documents\strategies

REM Instalar o actualizar Streamlit
pip install --upgrade streamlit
pip install --upgrade psutil

REM Ejecutar el archivo Python con los parámetros especificados
streamlit run %python_file%