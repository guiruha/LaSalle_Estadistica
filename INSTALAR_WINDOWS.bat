@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Preparar los cuadernos de Estadistica

rem Ejecutar con doble clic desde la carpeta completa del curso.
rem No se necesitan permisos de administrador ni PowerShell ni winget.
cd /d "%~dp0"
set "CURSO=%CD%"

echo =====================================================
echo   PREPARAR ESTADISTICA EN WINDOWS
echo =====================================================
echo.
echo Este programa descargara VS Code y uv desde sus sitios oficiales,
echo instalara las extensiones Python y Jupyter, y preparara el curso.
echo Necesitas Internet. Puede tardar varios minutos.
echo.

if not exist "pyproject.toml" goto :falta_carpeta
if not exist "uv.lock" goto :falta_carpeta
if not exist "datos\" goto :falta_carpeta
if not exist "sesiones\sesion1\s1.ipynb" goto :falta_carpeta

call :requisitos
if errorlevel 1 goto :error
call :vscode
if errorlevel 1 goto :error
call :uv
if errorlevel 1 goto :error

echo.
echo [3/4] Descargando Python 3.12 y las bibliotecas del curso...
"%UV%" sync --locked
if errorlevel 1 goto :error

echo.
echo [4/4] Instalando las extensiones de VS Code...
call "%CODECLI%" --install-extension ms-python.python
if errorlevel 1 goto :error
call "%CODECLI%" --install-extension ms-toolsai.jupyter
if errorlevel 1 goto :error

echo.
echo =====================================================
echo   TODO LISTO
echo =====================================================
echo Se abrira la carpeta del curso en VS Code.
echo Abre sesiones ^> sesion1 ^> s1.ipynb.
echo Arriba a la derecha, elige el kernel Python 3.12 de .venv
echo y ejecuta las celdas de arriba abajo.
echo.
call "%CODECLI%" "%CURSO%"
echo Si VS Code pregunta si confias en la carpeta, acepta solo si
echo el material te lo ha proporcionado el profesorado.
echo.
pause
exit /b 0

:falta_carpeta
echo.
echo ERROR: faltan archivos del curso junto a este instalador.
echo Descarga y EXTRAE la carpeta COMPLETA antes de hacer doble clic.
echo Deben estar aqui pyproject.toml, uv.lock, datos y sesiones.
goto :error

:requisitos
where curl.exe >nul 2>&1
if errorlevel 1 goto :requisitos_fallidos
where tar.exe >nul 2>&1
if errorlevel 1 goto :requisitos_fallidos
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    set "ARCH=arm64"
    set "UV_ARCH=aarch64"
)
if /i "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    set "ARCH=x64"
    set "UV_ARCH=x86_64"
)
if /i "%PROCESSOR_ARCHITECTURE%"=="x86" if defined PROCESSOR_ARCHITEW6432 (
    set "ARCH=x64"
    set "UV_ARCH=x86_64"
)
if not defined ARCH goto :requisitos_fallidos
exit /b 0

:requisitos_fallidos
echo.
echo Este Windows necesita ser de 64 bits y disponer de curl.exe
echo y tar.exe. Consulta con el soporte informatico de tu centro.
exit /b 1

:buscar_vscode
set "CODECLI="
if exist "%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd" set "CODECLI=%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd"
if defined CODECLI exit /b 0
if exist "%ProgramFiles%\Microsoft VS Code\bin\code.cmd" set "CODECLI=%ProgramFiles%\Microsoft VS Code\bin\code.cmd"
if defined CODECLI exit /b 0
where code.cmd >nul 2>&1
if not errorlevel 1 set "CODECLI=code.cmd"
if defined CODECLI exit /b 0
exit /b 1

:vscode
echo [1/4] Comprobando VS Code...
call :buscar_vscode
if not errorlevel 1 exit /b 0
echo VS Code no esta instalado. Descargando desde Microsoft...
set "INSTALADOR=%TEMP%\LaSalle-VSCode-UserSetup.exe"
curl.exe --fail --location --show-error --retry 2 --output "%INSTALADOR%" "https://update.code.visualstudio.com/latest/win32-%ARCH%-user/stable"
if errorlevel 1 (
    echo Descarga fallida: https://code.visualstudio.com/Download
    exit /b 1
)
echo Instalando VS Code solo para tu usuario, sin permisos de administrador...
start /wait "" "%INSTALADOR%" /SILENT /NORESTART
if errorlevel 1 exit /b 1
call :buscar_vscode
if errorlevel 1 (
    echo No se encontro VS Code tras el instalador.
    echo Consulta con el soporte informatico de tu centro.
    exit /b 1
)
exit /b 0

:uv
echo.
echo [2/4] Comprobando uv...
set "UV=%LOCALAPPDATA%\LaSalleEstadistica\uv\uv.exe"
if exist "%UV%" exit /b 0
where uv.exe >nul 2>&1
if not errorlevel 1 (
    set "UV=uv.exe"
    exit /b 0
)
set "UV_DIR=%LOCALAPPDATA%\LaSalleEstadistica\uv"
if not exist "%UV_DIR%" (
    mkdir "%UV_DIR%"
    if errorlevel 1 exit /b 1
)
echo Descargando uv desde GitHub (Astral)...
curl.exe --fail --location --show-error --retry 2 --output "%UV_DIR%\uv.zip" "https://github.com/astral-sh/uv/releases/latest/download/uv-%UV_ARCH%-pc-windows-msvc.zip"
if errorlevel 1 (
    echo Descarga fallida: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)
tar.exe -xf "%UV_DIR%\uv.zip" -C "%UV_DIR%"
if errorlevel 1 exit /b 1
if not exist "%UV%" (
    echo No se encontro uv.exe dentro de la descarga.
    exit /b 1
)
del /q "%UV_DIR%\uv.zip" >nul 2>&1
exit /b 0

:error
echo.
echo =====================================================
echo   NO SE HA COMPLETADO LA INSTALACION
echo =====================================================
echo No cambies el kernel ni instales paquetes por separado.
echo Haz una foto del mensaje de error que aparece ARRIBA y
echo ensenasela al profesorado o al soporte informatico.
echo Puedes volver a ejecutar este archivo despues de resolverlo.
echo.
pause
exit /b 1