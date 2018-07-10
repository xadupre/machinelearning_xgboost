// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using XGBoostMulticlassTrainer = Scikit.ML.XGBoostWrapper.XGBoostMulticlassTrainer;
using XGBoostArguments = Scikit.ML.XGBoostWrapper.XGBoostArguments;

[assembly: LoadableClass(XGBoostMulticlassTrainer.Summary, typeof(XGBoostMulticlassTrainer), typeof(XGBoostArguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    "eXGBoost Multi-class Classifier", "eXGBoostMulticlass", "eXGBoostMC", "exgbmc", "xgbmc")]

namespace Scikit.ML.XGBoostWrapper
{
    public sealed class XGBoostMulticlassTrainer : XGBoostTrainerBase<VBuffer<Float>, XGBoostMulticlassPredictor>
    {
        public const string Summary = "XGBoost Multi Class Classifier: https://github.com/dmlc/xgboost";

        private int _nbClass;
        private int[] _classMapping;
        private bool _isFloatLabel;

        public XGBoostMulticlassTrainer(IHostEnvironment env, XGBoostArguments args) : base(env, args, PredictionKind.MultiClassClassification, "eXGBoostMulticlass")
        {
        }

        public override XGBoostMulticlassPredictor CreatePredictor()
        {
            _host.CheckValue(_model, "model", "Must have trained a model before creating a predictor.");
            _host.Check(_nbClass > 0, "Must know the number of classes before creating a predictor.");
            // Create a new predictor by loading from the serialized model.
            return new XGBoostMulticlassPredictor(_host, _model, _nbClass, _classMapping, _isFloatLabel, 
                            _nbFeaturesXGboost, _nbFeaturesML);
        }

        protected override void UpdateXGBoostOptions(IChannel ch, Dictionary<string, string> options, Float[] labels, uint[] groups)
        {
            Contracts.AssertValue(ch, nameof(ch));
            ch.AssertValue(options, nameof(options));
            ch.AssertValue(labels, nameof(labels));
            if (!options.ContainsKey("num_class"))
            {
                options["num_class"] = (_classMapping == null ? _nbClass : _classMapping.Length).ToString();
                ch.Info("Estimated number of classes: {0}.", options["num_class"]);
            }
            else
                ch.Info("Number of classes: {0}.", options["num_class"]);

            if (!options.ContainsKey("objective"))
                options["objective"] = "multi:softprob";
            var mini = labels.Min();
            if (mini > 0)
                throw _host.Except("First class must be 0 for XGBoost Multiclass. Current range: {0}-{1}", mini, labels.Max());
        }

        protected override void PostProcessLabelsBeforeCreatingXGBoostContainer(IChannel ch, RoleMappedData data, Float[] labels)
        {
            Contracts.Assert(PredictionKind == PredictionKind.MultiClassClassification);

            int[] classMapping;

            // This builds the mapping from XGBoost classes to Microsoft.ML classes.
            // XGBoost classes must start at 0. The mapping removes empty classes as XGBoost
            // multiplies the number of tree by the number of classes, this reduces the complexity.
            classMapping = labels.Select(c => (int)c).Distinct().OrderBy(c => c).ToArray();
            ch.Check(classMapping[0] >= 0, "Negative labels are not allowed.");
            var map = classMapping.Select((c, i) => new { c = c, i = i })
                .ToDictionary(item => item.c, item => item.i);
            for (int i = 0; i < labels.Length; ++i)
            {
                ch.Assert(!Single.IsNaN(labels[i]));
                labels[i] = (float)map[(int)labels[i]];
            }

            _nbClass = classMapping.Length;

            // The classMapping is used by the prediction when the label is R4
            // or when the label is Key with a different range than XGBoost one.
            if (data.Schema.Label.Type.IsKey)
            {
                _isFloatLabel = false;
                var labelType = data.Schema.Label.Type.AsKey;
                if (labelType.Count == classMapping.Length)
                    classMapping = null;
                else
                {
                    // There are fewer classes in the training database than the label type provides.
                    // We keep the mapping to compute the final prediction
                    // returned as a sparse vector.
                    ulong max = (ulong)labelType.Count;
                    if (max >= int.MaxValue)
                        throw ch.Except("Labels must be < {0}.", int.MaxValue);
                    _nbClass = labelType.Count;
                    // Mapping starts at zero.
                    var mini = classMapping.Min();
                    for (int i = 0; i < classMapping.Length; ++i)
                        classMapping[i] -= mini;
                }
            }
            else if (data.Schema.Label.Type == NumberType.R4)
                _isFloatLabel = true;
            else
                throw ch.ExceptParam(nameof(data), "Label type must be a key or a float R4.");

            _classMapping = classMapping;
        }

        protected override void ValidateTrainInput(IChannel ch, RoleMappedData data)
        {
            _host.CheckParam(data.Schema.Label != null, nameof(data), "Need a label column");
            var labelType = data.Schema.Label.Type;
            if (labelType.RawKind != DataKind.R4 && !labelType.IsKey)
                throw _host.ExceptParam(nameof(data), "Label column 'Label' is of type '{0}', but must be R4 or a key.", labelType);
        }
    }
}
