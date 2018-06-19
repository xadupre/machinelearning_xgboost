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
cd xgboost
cd build
msbuild xgboost.sln
cd ..
cd ..

:maclea:
@echo [build.cmd] build machinelearning
cd machinelearning
if exist bin\x64.Release goto mldeb:
cmd /C build.cmd -release
:mldeb:
if exist bin\x64.Debug goto mlrel:
cmd /C build.cmd -debug
:mlrel:
cd ..

if not exist machinelearning\bin\x64.Debug goto end:

@echo [build.cmd] Publish Release
if not exist machinelearning\dist\Release mkdir machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Api\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Maml\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\x64.Release\Native\*.dll machinelearning\dist\Release
copy xgboost\lib\*.dll machinelearning\dist\Release

@echo [build.cmd] Publish Debug
if not exist machinelearning\dist\Debug mkdir machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Api\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Maml\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearning\dist\Debug
copy xgboost\lib\*.dll machinelearning\dist\Debug

@echo [build.cmd] build machinelearning_xgboost
cd machinelearning_xgboost
cmd /C dotnet build -c Debug
cmd /C dotnet build -c Release
cd ..

:end:
if not exist machinelearning\bin\x64.Debug @echo [build.cmd] Cannot build.
@echo [build.cmd] Completed.

