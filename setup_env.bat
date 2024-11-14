@echo off
setlocal

set ENV_NAME=myenv

conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo Fehler beim Erstellen der Conda-Umgebung.
    exit /b %ERRORLEVEL%
)

call conda activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo Fehler beim Aktivieren der Conda-Umgebung.
    exit /b %ERRORLEVEL%
)

pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Fehler beim Installieren der Pakete aus requirements.txt.
    exit /b %ERRORLEVEL%
)

echo Alle Pakete wurden erfolgreich installiert.
endlocal
