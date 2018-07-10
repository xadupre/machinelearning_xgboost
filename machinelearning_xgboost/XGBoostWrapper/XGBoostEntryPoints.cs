// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.CommandLine;
using EntryPointDefXGBoostBinaryTrainer = Scikit.ML.XGBoostWrapper.EntryPointDefXGBoostBinaryTrainer;

[assembly: LoadableClass(typeof(void), typeof(EntryPointDefXGBoostBinaryTrainer), null,
    typeof(SignatureEntryPointModule), EntryPointDefXGBoostBinaryTrainer.EntryPointName)]


namespace Scikit.ML.XGBoostWrapper
{
    #region binary

    public static partial class EntryPointDefXGBoostBinaryTrainer
    {
        public const string EntryPointName = "XGBoostBinaryClassifier";

        [TlcModule.EntryPoint(
            Name = "Scikit.ML." + EntryPointName,
            Desc = XGBoostBinaryTrainer.Summary,
            UserName = EntryPointName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, XGBoostArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Train" + EntryPointName);
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<XGBoostArguments,
                                           CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new XGBoostBinaryTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }

    public static class EntryPointXGBoostBinary
    {
        public static XGBoostBinaryClassifier.Output Add(this Microsoft.ML.Runtime.Experiment exp, XGBoostBinaryClassifier input)
        {
            var output = new XGBoostBinaryClassifier.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, XGBoostBinaryClassifier input, XGBoostBinaryClassifier.Output output)
        {
            // TODO: Internal API not available.
            // _jsonNodes.Add(Serialize("Scikit.XGBoostBinary", input, output));
            // It could be replaced by:
            // exp.AddSerialize("Scikit.XGBoostBinary", input, output);
        }
    }

    public enum XGBoostArgumentsEvalMetricType
    {
        DefaultMetric = 0,
        Rmse = 1,
        Mae = 2,
        Logloss = 3,
        Error = 4,
        Merror = 5,
        Mlogloss = 6,
        Auc = 7,
        Ndcg = 8,
        Map = 9
    }


    /// <summary>
    /// Train a LightGBM binary classification model.
    /// </summary>
    /// <remarks>Light GBM is an open source implementation of boosted trees.
    /// <a href='https://github.com/Microsoft/LightGBM/wiki'>GitHub: LightGBM</a></remarks>
    public sealed partial class XGBoostBinaryClassifier : 
                Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInputWithGroupId, 
        Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInputWithWeight, 
        Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInputWithLabel, 
        Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInput, 
        Microsoft.ML.ILearningPipelineItem
    {


        /// <summary>
        /// Number of iterations.
        /// </summary>
        [TlcModule.SweepableDiscreteParamAttribute("NumBoostRound", new object[] { 10, 20, 50, 100, 150, 200 })]
        public int NumBoostRound { get; set; } = 100;

        /// <summary>
        /// Shrinkage rate for trees, used to prevent over-fitting. Range: (0,1].
        /// </summary>
        [TlcModule.SweepableFloatParamAttribute("LearningRate", 0.025f, 0.4f, isLogScale: true)]
        public double? LearningRate { get; set; }

        /// <summary>
        /// Maximum leaves for trees.
        /// </summary>
        [TlcModule.SweepableLongParamAttribute("NumLeaves", 2, 128, stepSize: 4, isLogScale: true)]
        public int? NumLeaves { get; set; }

        /// <summary>
        /// Minimum number of instances needed in a child.
        /// </summary>
        [TlcModule.SweepableDiscreteParamAttribute("MinDataPerLeaf", new object[] { 1, 10, 20, 50 })]
        public int? MinDataPerLeaf { get; set; }

        /// <summary>
        /// Max number of bucket bin for features.
        /// </summary>
        public int MaxBin { get; set; } = 255;

        /// <summary>
        /// Which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.
        /// </summary>
        [JsonConverter(typeof(ComponentSerializer))]
        public BoosterParameterFunction Booster { get; set; } = new GbdtBoosterParameterFunction();

        /// <summary>
        /// Verbose
        /// </summary>
        public bool VerboseEval { get; set; } = false;

        /// <summary>
        /// Printing running messages.
        /// </summary>
        public bool Silent { get; set; } = true;

        /// <summary>
        /// Number of parallel threads used to run LightGBM.
        /// </summary>
        public int? NThread { get; set; }

        /// <summary>
        /// Evaluation metrics.
        /// </summary>
        public XGBoostArgumentsEvalMetricType EvalMetric { get; set; } = XGBoostArgumentsEvalMetricType.DefaultMetric;

        /// <summary>
        /// Use softmax loss for the multi classification.
        /// </summary>
        [TlcModule.SweepableDiscreteParamAttribute("UseSoftmax", new object[] { true, false })]
        public bool? UseSoftmax { get; set; }

        /// <summary>
        /// Rounds of early stopping, 0 will disable it.
        /// </summary>
        public int EarlyStoppingRound { get; set; }

        /// <summary>
        /// Comma seperated list of gains associated to each relevance label.
        /// </summary>
        public string CustomGains { get; set; } = "0,3,7,15,31,63,127,255,511,1023,2047,4095";

        /// <summary>
        /// Number of entries in a batch when loading data.
        /// </summary>
        public int BatchSize { get; set; } = 1048576;

        /// <summary>
        /// Enable categorical split or not.
        /// </summary>
        [TlcModule.SweepableDiscreteParamAttribute("UseCat", new object[] { true, false })]
        public bool? UseCat { get; set; }

        /// <summary>
        /// Enable missing value auto infer or not.
        /// </summary>
        [TlcModule.SweepableDiscreteParamAttribute("UseMissing", new object[] { true, false })]
        public bool UseMissing { get; set; } = false;

        /// <summary>
        /// Min number of instances per categorical group.
        /// </summary>
        [TlcModule.Range(Inf = 0, Max = 2147483647)]
        [TlcModule.SweepableDiscreteParamAttribute("MinDataPerGroup", new object[] { 10, 50, 100, 200 })]
        public int MinDataPerGroup { get; set; } = 100;

        /// <summary>
        /// Max number of categorical thresholds.
        /// </summary>
        [TlcModule.Range(Inf = 0, Max = 2147483647)]
        [TlcModule.SweepableDiscreteParamAttribute("MaxCatThreshold", new object[] { 8, 16, 32, 64 })]
        public int MaxCatThreshold { get; set; } = 32;

        /// <summary>
        /// Lapalace smooth term in categorical feature spilt. Avoid the bias of small categories.
        /// </summary>
        [TlcModule.Range(Min = 0d)]
        [TlcModule.SweepableDiscreteParamAttribute("CatSmooth", new object[] { 1, 10, 20 })]
        public double CatSmooth { get; set; } = 10d;

        /// <summary>
        /// L2 Regularization for categorical split.
        /// </summary>
        [TlcModule.Range(Min = 0d)]
        [TlcModule.SweepableDiscreteParamAttribute("CatL2", new object[] { 0.1f, 0.5f, 1, 5, 10 })]
        public double CatL2 { get; set; } = 10d;

        /// <summary>
        /// Parallel LightGBM Learning Algorithm
        /// </summary>
        [JsonConverter(typeof(ComponentSerializer))]
        public ParallelLightGBM ParallelTrainer { get; set; } = new SingleParallelLightGBM();

        /// <summary>
        /// Column to use for example groupId
        /// </summary>
        public Microsoft.ML.Runtime.EntryPoints.Optional<string> GroupIdColumn { get; set; }

        /// <summary>
        /// Column to use for example weight
        /// </summary>
        public Microsoft.ML.Runtime.EntryPoints.Optional<string> WeightColumn { get; set; }

        /// <summary>
        /// Column to use for labels
        /// </summary>
        public string LabelColumn { get; set; } = "Label";

        /// <summary>
        /// The data to be used for training
        /// </summary>
        public Var<Microsoft.ML.Runtime.Data.IDataView> TrainingData { get; set; } = new Var<Microsoft.ML.Runtime.Data.IDataView>();

        /// <summary>
        /// Column to use for features
        /// </summary>
        public string FeatureColumn { get; set; } = "Features";

        /// <summary>
        /// Normalize option for the feature column
        /// </summary>
        public Microsoft.ML.Models.NormalizeOption NormalizeFeatures { get; set; } = Microsoft.ML.Models.NormalizeOption.Auto;

        /// <summary>
        /// Whether learner should cache input training data
        /// </summary>
        public Microsoft.ML.Models.CachingOptions Caching { get; set; } = Microsoft.ML.Models.CachingOptions.Auto;


        public sealed class Output : Microsoft.ML.Runtime.EntryPoints.CommonOutputs.IBinaryClassificationOutput, Microsoft.ML.Runtime.EntryPoints.CommonOutputs.ITrainerOutput
        {
            /// <summary>
            /// The trained model
            /// </summary>
            public Var<Microsoft.ML.Runtime.EntryPoints.IPredictorModel> PredictorModel { get; set; } = new Var<Microsoft.ML.Runtime.EntryPoints.IPredictorModel>();

        }
        public Var<IDataView> GetInputData() => TrainingData;

        public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
        {
            if (previousStep != null)
            {
                if (!(previousStep is ILearningPipelineDataStep dataStep))
                {
                    throw new InvalidOperationException($"{ nameof(XGBoostBinaryClassifier)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                }

                TrainingData = dataStep.Data;
            }
            Output output = EntryPointXGBoostBinary.Add(experiment, this);
            return new XGBoostBinaryClassifierBinaryClassifierPipelineStep(output);
        }

        private class XGBoostBinaryClassifierBinaryClassifierPipelineStep : ILearningPipelinePredictorStep
        {
            public XGBoostBinaryClassifierBinaryClassifierPipelineStep(Output output)
            {
                Model = output.PredictorModel;
            }

            public Var<IPredictorModel> Model { get; }
        }
    }



    #endregion
}
