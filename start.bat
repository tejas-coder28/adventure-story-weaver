@echo off
setlocal
title Adventure Story Weaver
cd /d "%~dp0"

echo =====================================================================
echo                     ADVENTURE STORY WEAVER
echo =====================================================================
echo.

:: 1. Find Python executable
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

:: Verify Python works
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Please install Python 3.12+ or set up your virtual environment.
    echo.
    pause
    exit /b 1
)

:: 2. Ensure .env exists if .env.example is present
if not exist ".env" if exist ".env.example" copy .env.example .env >nul

:: 3. Check for story corpus
if not exist "data\corpus.txt" (
    echo [*] Story corpus not found. Downloading adventure texts...
    "%PYTHON_EXE%" scripts\build_corpus.py --download-sample
    if errorlevel 1 (
        echo [ERROR] Failed to build corpus.
        pause
        exit /b 1
    )
)

:: 4. Check for tokenizer
if not exist "artifacts\tokenizer.json" (
    echo [*] Tokenizer vocabulary not found. Training byte-level BPE tokenizer...
    "%PYTHON_EXE%" -m app.cli train-tokenizer --vocab-size 1000
    if errorlevel 1 (
        echo [ERROR] Failed to train tokenizer.
        pause
        exit /b 1
    )
)

:: 5. Check for model checkpoint
if not exist "checkpoints\model.pt" (
    echo [*] Model checkpoint not found. Starting short initial training run...
    "%PYTHON_EXE%" -m app.cli train --max_steps 200 --batch_size 16 --lr 1e-3
    if errorlevel 1 (
        echo [ERROR] Failed to train initial model.
        pause
        exit /b 1
    )
)

:menu
cls
echo =====================================================================
echo                     ADVENTURE STORY WEAVER
echo =====================================================================
echo.
echo   [1] Interactive Story Mode  (Enter prompts, get story continuations)
echo   [2] Quick Single Prompt     (Enter a custom prompt)
echo   [3] Train Model - Fast      (200 steps demo run)
echo   [4] Train Model - Full      (1000 steps full run)
echo   [5] Run All Tests           (pytest test suite)
echo   [6] Exit
echo.
echo =====================================================================
choice /c 123456 /m "Select an option [1-6]: "

if errorlevel 6 goto end
if errorlevel 5 goto run_tests
if errorlevel 4 goto train_full
if errorlevel 3 goto train_demo
if errorlevel 2 goto single_prompt
if errorlevel 1 goto interactive

goto menu

:interactive
cls
echo Launching Interactive Story Mode...
echo.
"%PYTHON_EXE%" -m app.cli --interactive
echo.
pause
goto menu

:single_prompt
cls
echo Enter your story prompt below:
echo.
set /p user_prompt="Story Prompt > "
if "%user_prompt%"=="" goto menu
echo.
echo Generating continuation...
echo.
"%PYTHON_EXE%" -m app.cli "%user_prompt%"
echo.
pause
goto menu

:train_demo
cls
echo Starting short training run (200 steps)...
echo.
"%PYTHON_EXE%" -m app.cli train --max_steps 200 --batch_size 16 --lr 1e-3
echo.
pause
goto menu

:train_full
cls
echo Starting full training run (1000 steps)...
echo.
"%PYTHON_EXE%" -m app.cli train --max_steps 1000 --batch_size 16 --lr 1e-3
echo.
pause
goto menu

:run_tests
cls
echo Running test suite...
echo.
"%PYTHON_EXE%" -m pytest -v
echo.
pause
goto menu

:end
echo Farewell, adventurer!
