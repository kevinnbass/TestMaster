.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"
uvicorn backend.codebase_monitor.service:app --host 127.0.0.1 --port 8088