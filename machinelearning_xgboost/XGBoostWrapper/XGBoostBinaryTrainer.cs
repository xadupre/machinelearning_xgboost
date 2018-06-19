// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using XGBoostBinaryTrainer = Microsoft.ML.XGBoostWrappers.XGBoostBinaryTrainer;
using XGBoostArguments = Microsoft.ML.XGBoostWrappers.XGBoostArguments;

[assembly: LoadableClass(XGBoostBinaryTrainer.Summary, typeof(XGBoostBinaryTrainer), typeof(XGBoostArguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    "eXGBoost Binary Classification", "eXGBoostBinary", "eXGBoost", "exgb", "xgb")]

namespace Microsoft.ML.XGBoostWrappers
{
    /// <summary>
    /// Trainer for a XGBoost binary classification.
    /// </summary>
    public sealed class XGBoostBinaryTrainer : XGBoostTrainerBase<Float, XGBoostBinaryPredictor>
    {
        public const string Summary = "XGBoost Binary Classifier: https://github.com/dmlc/xgboost.";

        public XGBoostBinaryTrainer(IHostEnvironment env, XGBoostArguments args) : base(env, args, PredictionKind.BinaryClassification, "eXGBoostBinary")
        {
        }

        public override XGBoostBinaryPredictor CreatePredictor()
        {
            _host.CheckValue(_model, "model", "Must have trained a model before creating a predictor.");
            // Create a new predictor by loading from the serialized model.
            return new XGBoostBinaryPredictor(_host, _model, _nbFeaturesXGboost, _nbFeaturesML);
        }

        protected override void UpdateXGBoostOptions(IChannel ch, Dictionary<string, string> options, Float[] labels, uint[] groups)
        {
            Contracts.AssertValue(ch, nameof(ch));
            ch.AssertValue(options, nameof(options));
            ch.AssertValue(labels, nameof(labels));
            if (!options.ContainsKey("objective"))
                options["objective"] = "binary:logistic";
            if (options.ContainsKey("scale_pos_weight"))
            {
                float value;
                if (!float.TryParse(options["scale_pos_weight"], out value))
                    throw _host.ExceptParam("scalePosWeight", string.Format("Unable to convert '{0}' into a float.", options["scale_pos_weight"]));
                if (value < 0)
                    throw _host.ExceptParam("scalePosWeight", "Must be positive or null.");
                if (value == 0)
                {
                    // We give a default value, see https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py#L31.
                    var s0 = labels.Where(x => x != 1).Count();
                    var s1 = labels.Where(x => x == 1).Count();
                    var ratio = System.Math.Max(s0, 1f) / System.Math.Max(s1, 1f);
                    options["scale_pos_weight"] = string.Format("{0}", ratio);
                    ch.Info("scale_pos_weight == 0 --> change it into {0}.", ratio);
                }
            }
        }
    }
}
