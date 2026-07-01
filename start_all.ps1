# start_all.ps1
Write-Host "🚀 Starting Fraud Detection System..." -ForegroundColor Cyan

# 1. Start Docker containers
Write-Host "🐳 Starting Docker containers (PostgreSQL, Redis, Kafka)..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker compose failed. Make sure Docker is running." -ForegroundColor Red
    exit 1
}

Write-Host "⏳ Waiting 10 seconds for services to initialise..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# 2. Activate virtual environment (we'll use absolute path to python from .venv)
$venvPython = ".\\.venv\\Scripts\\python.exe"
$venvActivate = ".\\\\.venv\\Scripts\\activate"

# 3. Start each component in a new PowerShell window
Write-Host "📦 Starting Aggregator (Redis feature updater)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; & $venvActivate; python backend\aggregator.py"

Write-Host "📡 Starting Kafka Consumer (scores transactions)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; & $venvActivate; python backend\consumer_predict_confluent.py"

Write-Host "🔄 Starting Transaction Producer (sends live data)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; & $venvActivate; python backend\producer_live.py"

Write-Host "⚡ Starting FastAPI server (port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; & $venvActivate; uvicorn backend.main:main --host 0.0.0.0 --port 8000 --reload"

Write-Host "📊 Starting Live Monitor Dashboard (port 8501)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; & $venvActivate; streamlit run frontend/dashboard.py --server.port 8501"

Write-Host "🔍 Starting Account Investigation Dashboard (port 8502)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; & $venvActivate; streamlit run frontend/get.py --server.port 8502"

Write-Host "✅ All components launched!" -ForegroundColor Green
Write-Host ""
Write-Host "📌 Access points:"
Write-Host "   🔗 FastAPI Swagger:    http://localhost:8000/docs"
Write-Host "   📊 Live Monitor:       http://localhost:8501"
Write-Host "   🔍 Account Investigation: http://localhost:8502"
Write-Host ""
Write-Host "⚠️  To stop everything, simply close all PowerShell windows."