@echo off
REM Sobe o watchdog (scripts\watchdog.py), que por sua vez sobe o painel
REM (dashboard.py) e o bot de Telegram (bot.py) e os reinicia sozinho se
REM algum cair (avisando no Telegram). Uma unica janela minimizada; bot.py
REM e dashboard.py fazem seu proprio log com rotacao em logs\bot.log e
REM logs\painel.log — o watchdog cuida do dele em logs\watchdog.log.
REM Roda uma vez manualmente ou automaticamente no login (ver
REM instalar_inicializacao.bat).
setlocal
set PYTHON_EXE=C:\Users\Arthur\AppData\Local\Python\pythoncore-3.14-64\python.exe
set PROJETO=C:\Users\Arthur\futprob

if not exist "%PROJETO%\logs" mkdir "%PROJETO%\logs"
cd /d "%PROJETO%"

start "futprob-watchdog" /min cmd /c ""%PYTHON_EXE%" scripts\watchdog.py"

endlocal
