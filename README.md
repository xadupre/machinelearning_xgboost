# XGBoost Wrapper for Microsoft.ML

This project proposes an extension to
[machinelearning](https://github.com/dotnet/machinelearning)
written in C# which wraps [XGBoost](https://github.com/dmlc/xgboost) 
into the C# API. The wrapper requires changes to XGBoost source code
to enable one-off predictions available in this repository:
[XGBoost](https://github.com/xadupre/xgboost).

[![TravisCI](https://travis-ci.org/xadupre/machinelearning_xgboost.svg?branch=master)](https://travis-ci.org/xadupre/machinelearning_xgboost)
[![Build status](https://ci.appveyor.com/api/projects/status/7m8x515b7ek3pddk?svg=true)](https://ci.appveyor.com/project/xadupre/machinelearning_xgboost)
[![CircleCI](https://circleci.com/gh/xadupre/machinelearning_xgboost.svg?style=svg)](https://circleci.com/gh/xadupre/machinelearning_xgboost)

## Build

On windows: ``build.cmd``.

On Linux: ``build.sh``.

Build the documentation: ``doxygen conf.dox``.

## Usage

Once the project ``machinelearning_xgboost`` is built,
the binaires *XGBoost* needs to be copied in the same folder as
Microsoft.ML's.
Example:
