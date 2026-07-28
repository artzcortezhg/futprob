@echo off
REM Registra o futprob (painel + bot) para subir sozinho no login do Windows,
REM copiando um atalho pra pasta Startup do usuario que so chama
REM iniciar_futprob.bat do repositorio. Rode este script UMA VEZ; para
REM desfazer, apague o arquivo iniciar_futprob_startup.bat da pasta Startup
REM (o caminho e' impresso no final).
setlocal
set PROJETO=C:\Users\Arthur\futprob
set STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set DESTINO=%STARTUP%\iniciar_futprob_startup.bat

echo @echo off > "%DESTINO%"
echo call "%PROJETO%\scripts\iniciar_futprob.bat" >> "%DESTINO%"

echo Instalado em: %DESTINO%
echo Para desfazer, apague esse arquivo.
endlocal
