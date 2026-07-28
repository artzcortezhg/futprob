@echo off
REM Sobe o painel (dashboard.py) e o bot de Telegram (bot.py) juntos, em
REM janelas minimizadas, com saida redirecionada para logs/. Roda uma vez
REM manualmente ou automaticamente no login (ver instalar_inicializacao.bat).
setlocal
set PYTHON_EXE=C:\Users\Arthur\AppData\Local\Python\pythoncore-3.14-64\python.exe
set PROJETO=C:\Users\Arthur\futprob

if not exist "%PROJETO%\logs" mkdir "%PROJETO%\logs"
cd /d "%PROJETO%"

start "futprob-painel" /min cmd /c ""%PYTHON_EXE%" src\dashboard.py >> logs\painel.log 2>&1"
start "futprob-bot" /min cmd /c ""%PYTHON_EXE%" src\bot.py >> logs\bot.log 2>&1"

endlocal
