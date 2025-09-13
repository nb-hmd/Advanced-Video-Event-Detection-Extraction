@echo off
echo Starting Streamlit Video Event Detection Server...
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please ensure the venv folder exists in the project directory.
    pause
    exit /b 1
)

REM Activate virtual environment and start server
call venv\Scripts\activate.bat

REM Add FFmpeg to PATH
set PATH=%PATH%;C:\ffmpeg\bin

REM Start Streamlit with error handling
echo Starting Streamlit server...
streamlit run src/web/streamlit_app.py --logger.level info --server.headless true

REM If we reach here, the server stopped
echo.
echo Server has stopped. Press any key to exit...
pause >nul