@echo off
set current_directory=%userprofile%
REM Ruta especifica 
cd %current_directory%\Documents\strategies

echo Ejecutando el script para actualizar el repositorio...

git restore .

git add .
git commit -m "cambios locales"

REM Hacer un pull del repositorio remoto
git pull

echo TODO ESTA ACTUALIZADO!!!!
pause