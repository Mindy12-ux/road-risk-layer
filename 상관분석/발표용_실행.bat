@echo off
REM 상관분석 6개 + 발표용 종합 그래프를 한 번에 실행 (저장소 맨 위 폴더에서: 상관분석\발표용_실행.bat)
REM 먼저: conda activate roadrisk  /  dvc pull 로 model\results\model_input 받기
chcp 65001 >nul
cd /d "%~dp0.."
for %%f in (target_sc\spearman.py target_sc\phi.py target_sc\V.py target_sc\anova.py multi\pearson.py multi\VIF.py 발표용_종합.py) do (
    echo ===== %%f
    python "상관분석\%%f" || goto :err
)
echo.
echo 끝! 그림은 reports\ 폴더에 있습니다. 발표용 종합은 reports\발표용\
goto :eof
:err
echo [실패] 위 오류 메시지를 확인하세요.
exit /b 1
