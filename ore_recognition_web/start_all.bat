@echo off & start "Ore Backend MySQL" cmd /k "%~dp0start_backend.bat" & start "Ore Frontend" cmd /k "%~dp0start_frontend.bat"
