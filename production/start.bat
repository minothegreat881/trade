@echo off
REM ================================================================
REM QUICK START SCRIPT - Phase 1 Production Deployment
REM ================================================================

echo ================================================================================
echo STARTING ML TRADING SYSTEM - PHASE 1
echo ================================================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo [OK] Docker is running
echo.

REM Start Docker containers
echo Starting InfluxDB and Grafana...
docker-compose up -d
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to start Docker containers!
    pause
    exit /b 1
)
echo [OK] Docker containers started
echo.

REM Wait for services to initialize
echo Waiting 30 seconds for services to initialize...
timeout /t 30 /nobreak >nul
echo [OK] Services initialized
echo.

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH!
    pause
    exit /b 1
)
echo [OK] Python is available
echo.

REM Create logs directory
if not exist "logs" mkdir logs
echo [OK] Logs directory ready
echo.

echo ================================================================================
echo SETUP COMPLETE!
echo ================================================================================
echo.
echo Services:
echo   - InfluxDB: http://localhost:8086
echo   - Grafana:  http://localhost:3000
echo.
echo Grafana Login:
echo   - Username: admin
echo   - Password: admin
echo.
echo ================================================================================
echo.
echo Starting data pipeline...
echo Press Ctrl+C to stop the pipeline
echo.
echo ================================================================================
echo.

REM Start the data pipeline
python realtime_data_pipeline.py

REM If the pipeline stops, ask user what to do
echo.
echo ================================================================================
echo PIPELINE STOPPED
echo ================================================================================
echo.
echo Do you want to stop Docker containers? (Y/N)
set /p STOP_DOCKER=

if /i "%STOP_DOCKER%"=="Y" (
    echo Stopping Docker containers...
    docker-compose down
    echo [OK] Containers stopped
)

echo.
echo ================================================================================
echo GOODBYE!
echo ================================================================================
pause
