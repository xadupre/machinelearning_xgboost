cd machinelearning
bash build.sh -release
bash build.sh -debug
cd ..

# missing instruction for XGBoost

cd machinelearning
bash -c "dotnet publish Microsoft.ML.sln -o ../../dist/Debug -c Debug --self-contained" || true
bash -c "dotnet publish Microsoft.ML.sln -o ../../dist/Release -c Release --self-contained" || true
cd ..

copy machinelearning/bin/x64.Debug/Native/*.so machinelearning/dist/Debug
copy machinelearning/bin/x64.Release/Native/*.so machinelearning/dist/Release
copy xgboost/lib/*.so machinelearning/dist/Debug
copy xgboost/lib/*.so machinelearning/dist/Release

cd machinelearning_xgboost
dotnet build -c Debug
dotnet build -c Release
cd ..
