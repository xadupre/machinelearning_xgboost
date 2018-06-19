# XGBoost Wrapper for ML.net

This project proposes an extension to
[machinelearning](https://github.com/dotnet/machinelearning)
written in C# which wraps [XGBoost](https://github.com/dmlc/xgboost) 
into the C# API. The wrapper requires changes to XGBoost source code
to enable one-off predictions available in this repository:
[XGBoost](https://github.com/xadupre/xgboost).

[![TravisCI](https://travis-ci.org/xadupre/machinelearning_xgboost.svg?branch=master)](https://travis-ci.org/xadupre/machinelearning_xgboost)
[![Build status](https://ci.appveyor.com/api/projects/status/cb0xos4p3xe1bqmg?svg=true)](https://ci.appveyor.com/project/xadupre/machinelearning_xgboost)
[![CircleCI](https://circleci.com/gh/xadupre/machinelearning_xgboost.svg?style=svg)](https://circleci.com/gh/xadupre/machinelearning_xgboost)

## Build

On windows: ``build.cmd``.

On Linux: ``build.sh``.

## Documentation

``doxygen conf.dox``
