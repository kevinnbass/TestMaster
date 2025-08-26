@echo off
echo Starting Codebase Monitor Service...
echo.
powershell.exe -ExecutionPolicy Bypass -File "scripts\scan_server.ps1" -Reload
pause