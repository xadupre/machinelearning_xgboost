// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using XGBoostRegressorTrainer = Microsoft.ML.XGBoostWrappers.XGBoostRegressorTrainer;
using XGBoostArguments = Microsoft.ML.XGBoostWrappers.XGBoostArguments;

[assembly: LoadableClass(XGBoostRegressorTrainer.Summary, typeof(XGBoostRegressorTrainer), typeof(XGBoostArguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer) },
    "eXGBoost Regressor", "eXGBoostRegression", "eXGBoostR", "exgbr", "xgbr")]

namespace Microsoft.ML.XGBoostWrappers
{
    public sealed class XGBoostRegressorTrainer : XGBoostTrainerBase<Float, XGBoostRegressionPredictor>
    {
        public const string Summary = "XGBoost Regression: https://github.com/dmlc/xgboost.";

        public XGBoostRegressorTrainer(IHostEnvironment env, XGBoostArguments args) : base(env, args, PredictionKind.Regression, "eXGBoostRegressor")
        {
        }

        public override XGBoostRegressionPredictor CreatePredictor()
        {
            _host.CheckValue(_model, "Must have trained a model before creating a predictor.");

            // create a new predictor by loading from the serialized model
            return new XGBoostRegressionPredictor(_model, _host, _nbFeaturesXGboost, _nbFeaturesML);
        }

        protected override void UpdateXGBoostOptions(IChannel ch, Dictionary<string, string> options, Float[] labels, uint[] groups)
        {
            Contracts.AssertValue(ch, nameof(ch));
            ch.AssertValue(options, nameof(options));
            ch.AssertValue(labels, nameof(labels));
            if (!options.ContainsKey("objective"))
                options["objective"] = "reg:linear";
        }
    }
}
