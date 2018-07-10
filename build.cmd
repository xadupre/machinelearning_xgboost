@echo off

@echo [build.cmd] update xgboost submodule
cd xgboost
git submodule init
git submodule update
cd ..

@echo [build.cmd] cmake xgboost
if exist xgboost\build\xgboost.sln goto bboost:
cd xgboost
if not exist build mkdir build
cd build
cmake .. -G"Visual Studio 15 2017 Win64"
cd ..
cd ..

:bboost:
@echo [build.cmd] build xgboost
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin
cd xgboost
cd build
msbuild xgboost.sln
cd ..
cd ..


@echo [build.cmd] Publish Release
if not exist dist\Release mkdir dist\Release
copy xgboost\lib\*.dll dist\Release

@echo [build.cmd] Publish Debug
if not exist machinelearning\dist\Debug mkdir dist\Debug
copy xgboost\lib\*.dll dist\Debug

@echo [build.cmd] build machinelearning_xgboost
cd machinelearning_xgboost
cmd /C dotnet build -c Debug
cmd /C dotnet build -c Release
cd ..

:end:
if not exist machinelearning\bin\x64.Debug @echo [build.cmd] Cannot build.
@echo [build.cmd] Completed.

