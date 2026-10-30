
@echo off
setlocal
cd /d C:\Users\HP\climate-twin

if not exist logs mkdir logs
set "LOG=logs\scheduled_forecast.log"

echo. >> "%LOG%"
echo ======================================== >> "%LOG%"
echo Pipeline started: %date% %time% >> "%LOG%"

echo [1/3] Fetching live weather grid...
python src\data\fetch_live_grid.py >> "%LOG%" 2>&1
if errorlevel 1 goto fetch_failed

echo [2/3] Building live weather data...
python src\data\build_live_dashboard_data.py >> "%LOG%" 2>&1
if errorlevel 1 goto build_failed

echo [3/3] Generating live-compatible forecast...
python src\models\generate_live_compatible_forecast.py >> "%LOG%" 2>&1
if errorlevel 1 goto forecast_failed

echo Pipeline completed successfully: %date% %time% >> "%LOG%"
echo SUCCESS: Live pipeline completed.
exit /b 0

:fetch_failed
echo ERROR: Live grid fetch failed. >> "%LOG%"
echo FAILED: Live grid fetch.
exit /b 1

:build_failed
echo ERROR: Live dashboard-data build failed. >> "%LOG%"
echo FAILED: Live data preparation.
exit /b 1

:forecast_failed
echo ERROR: Live-compatible forecast failed. >> "%LOG%"
echo FAILED: Forecast generation.
exit /b 1
