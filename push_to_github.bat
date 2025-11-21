@echo off
echo ===================================
echo GitHub Push Helper Script
echo ===================================
echo.
echo This script will help you push to GitHub using your token.
echo.
echo Step 1: Get your GitHub token
echo   - Go to: https://github.com/settings/tokens
echo   - Generate a new token with 'repo' scope
echo   - Copy the token
echo.
echo Step 2: Push with token
echo.
set /p TOKEN="Paste your GitHub token here: "
echo.
echo Pushing to GitHub...
git push https://%TOKEN%@github.com/Mayank-iitj/mask.git main
echo.
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS! Code pushed to GitHub.
) else (
    echo FAILED. Please check your token and try again.
)
pause
