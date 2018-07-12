// See the LICENSE file in the project root for more information.

using System;
using Newtonsoft.Json;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using EntryPointDefXGBoostBinaryTrainer = Scikit.ML.XGBoostWrapper.EntryPointDefXGBoostBinaryTrainer;
using XGBoostArguments = Scikit.ML.XGBoostWrapper.XGBoostArguments;

[assembly: LoadableClass(typeof(void), typeof(EntryPointDefXGBoostBinaryTrainer), null,
    typeof(SignatureEntryPointModule), EntryPointDefXGBoostBinaryTrainer.EntryPointName)]

[assembly: EntryPointModule(typeof(XGBoostArguments.TreeBooster.Arguments))]
[assembly: EntryPointModule(typeof(XGBoostArguments.DartBooster.Arguments))]
[assembly: EntryPointModule(typeof(XGBoostArguments.LinearBooster.Arguments))]

namespace Scikit.ML.XGBoostWrapper
{
    /// <summary>
    /// Creates a similar class to ComponentKind in ML.net
    /// unavailable here due to internal members.
    /// </summary>
    public abstract class ComponentKindXGBoost
    {
        /*internal*/ public ComponentKindXGBoost() { }

        [JsonIgnore]
        /*internal*/ public abstract string ComponentName { get; }
    }

    #region binary

    public static partial class EntryPointDefXGBoostBinaryTrainer
    {
        public const string EntryPointName = "XGBoostBinary";

        [TlcModule.EntryPoint(
            Name = "ScikitML." + EntryPointName,
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
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }

    /// <summary>
    /// This is only executed if this DLL and its dependencies is placed
    /// beside the other DLL from the nuget package: Microsoft.ML.dll, Microsoft.ML.Core.dll, ...
    /// </summary>
    public static class EntryPointXGBoostBinary
    {
        public static Scikit.ML.XGBoostWrapper.XGBoostBinary.Output Add(this Microsoft.ML.Runtime.Experiment exp, Scikit.ML.XGBoostWrapper.XGBoostBinary input)
        {
            var output = new Scikit.ML.XGBoostWrapper.XGBoostBinary.Output();
            exp.Add(input, output);
            return output;
        }

        public static void Add(this Microsoft.ML.Runtime.Experiment exp, Scikit.ML.XGBoostWrapper.XGBoostBinary input, Scikit.ML.XGBoostWrapper.XGBoostBinary.Output output)
        {
            // TODO: Internal API not available.
            // _jsonNodes.Add(Serialize("ScikitML.XGBoostBinary", input, output));
            // It could be replaced by:
            // exp.AddSerialize("ScikitML.XGBoostBinary", input, output);
        }
    }

    public enum XGBoostArgumentsEvalMetric
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
        Map = 9,
        NdcgAT1 = 10,
        NdcgAT3 = 11,
        NdcgAT5 = 12,
        MapAT1 = 13,
        MapAT3 = 14,
        MapAT5 = 15
    }

    public abstract class BoosterParameterFunction : ComponentKindXGBoost { }

    public enum XGBoostArgumentsDartBoosterArgumentsSampleType
    {
        Uniform = 0,
        Weighted = 1
    }

    public enum XGBoostArgumentsDartBoosterArgumentsNormalizeType
    {
        Tree = 0,
        Forest = 1
    }

    public enum XGBoostArgumentsTreeBoosterArgumentsTreeMethodEnum
    {
        Auto = 0,
        Approx = 1,
        Exact = 2
    }

    /// <summary>
    /// Dropouts meet Multiple Additive Regresion Trees (XGBoost).
    /// </summary>
    public sealed class EdartBoosterParameterFunction : BoosterParameterFunction
    {
        /// <summary>
        /// Type of sampling algorithm
        /// </summary>
        [JsonProperty("sampleType")]
        public XGBoostArgumentsDartBoosterArgumentsSampleType SampleType { get; set; } = XGBoostArgumentsDartBoosterArgumentsSampleType.Uniform;

        /// <summary>
        /// Type of normalization algorithm
        /// </summary>
        [JsonProperty("normalizeType")]
        public XGBoostArgumentsDartBoosterArgumentsNormalizeType NormalizeType { get; set; } = XGBoostArgumentsDartBoosterArgumentsNormalizeType.Tree;

        /// <summary>
        /// Dropout rate.  Range: [0, 1]
        /// </summary>
        [JsonProperty("rateDrop")]
        public double RateDrop { get; set; }

        /// <summary>
        /// Probability of skip dropout.  Range: [0, 1].
        /// </summary>
        [JsonProperty("skipDrop")]
        public double SkipDrop { get; set; }

        /// <summary>
        /// Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features and eta actually shrinks the feature weights to make the boosting process more conservative. Range: [0,1].
        /// </summary>
        [JsonProperty("learningRate")]
        public double LearningRate { get; set; } = 0.3d;

        /// <summary>
        /// Minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be. Range: [0,inf[.
        /// </summary>
        [JsonProperty("gamma")]
        public double Gamma { get; set; }

        /// <summary>
        /// Maximum depth of a tree, increase this value will make model more complex / likely to be overfitting. Range: [1,inf[.
        /// </summary>
        [JsonProperty("maxDepth")]
        public int MaxDepth { get; set; } = 6;

        /// <summary>
        /// Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. Range: [0,inf[.
        /// </summary>
        [JsonProperty("minChildWeight")]
        public int MinChildWeight { get; set; } = 1;

        /// <summary>
        /// Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update. Range: [0,inf[.
        /// </summary>
        [JsonProperty("maxDeltaStep")]
        public double MaxDeltaStep { get; set; }

        /// <summary>
        /// Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting. Range: (0,1].
        /// </summary>
        [JsonProperty("subSample")]
        public double SubSample { get; set; } = 1d;

        /// <summary>
        /// Subsample ratio of columns when constructing each tree. Range: (0,1].
        /// </summary>
        [JsonProperty("colSampleByTree")]
        public double ColSampleByTree { get; set; } = 1d;

        /// <summary>
        /// Subsample ratio of columns for each split, in each level. Range: (0,1[.
        /// </summary>
        [JsonProperty("colSampleByLevel")]
        public double ColSampleByLevel { get; set; } = 1d;

        /// <summary>
        /// L1 regularization term on weights, increase this value will make model more conservative.
        /// </summary>
        [JsonProperty("alpha")]
        public double Alpha { get; set; }

        /// <summary>
        /// L2 regularization term on weights, increase this value will make model more conservative.
        /// </summary>
        [JsonProperty("lambda")]
        public double Lambda { get; set; } = 1d;

        /// <summary>
        /// The tree construction algorithm used in XGBoost(see description in the reference paper), Distributed and external memory version only support approximate algorithm. Choices: {'auto', 'exact', 'approx'} 'auto': Use heuristic to choose faster one. For small to medium dataset, exact greedy will be used. For very large-dataset, approximate algorithm will be choosed. Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is choosed to notify this choice.  'exact': Exact greedy algorithm. 'approx': Approximate greedy algorithm using sketching and histogram.
        /// </summary>
        [JsonProperty("treeMethod")]
        public XGBoostArgumentsTreeBoosterArgumentsTreeMethodEnum TreeMethod { get; set; } = XGBoostArgumentsTreeBoosterArgumentsTreeMethodEnum.Auto;

        /// <summary>
        /// This is only used for approximate greedy algorithm. This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical ganrantee with sketch accuracy. Usuaully user do not have to tune this. but consider set to lower number for more accurate enumeration. Range: (0,1).
        /// </summary>
        [JsonProperty("sketchEps")]
        public double SketchEps { get; set; } = 0.03d;

        /// <summary>
        /// Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases). If the value is 0, the parameter will be estimated as suggested in https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py#L31 for a binary classification problem.
        /// </summary>
        [JsonProperty("scalePosWeight")]
        public double ScalePosWeight { get; set; } = 1d;

        public override string ComponentName => "edart";
    }

    /// <summary>
    /// Traditional Gradient Boosting Decision Tree.
    /// </summary>
    public sealed class EgbtreeBoosterParameterFunction : BoosterParameterFunction
    {
        /// <summary>
        /// Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features and eta actually shrinks the feature weights to make the boosting process more conservative. Range: [0,1].
        /// </summary>
        [JsonProperty("learningRate")]
        public double LearningRate { get; set; } = 0.3d;

        /// <summary>
        /// Minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be. Range: [0,inf[.
        /// </summary>
        [JsonProperty("gamma")]
        public double Gamma { get; set; }

        /// <summary>
        /// Maximum depth of a tree, increase this value will make model more complex / likely to be overfitting. Range: [1,inf[.
        /// </summary>
        [JsonProperty("maxDepth")]
        public int MaxDepth { get; set; } = 6;

        /// <summary>
        /// Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. Range: [0,inf[.
        /// </summary>
        [JsonProperty("minChildWeight")]
        public int MinChildWeight { get; set; } = 1;

        /// <summary>
        /// Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update. Range: [0,inf[.
        /// </summary>
        [JsonProperty("maxDeltaStep")]
        public double MaxDeltaStep { get; set; }

        /// <summary>
        /// Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting. Range: (0,1].
        /// </summary>
        [JsonProperty("subSample")]
        public double SubSample { get; set; } = 1d;

        /// <summary>
        /// Subsample ratio of columns when constructing each tree. Range: (0,1].
        /// </summary>
        [JsonProperty("colSampleByTree")]
        public double ColSampleByTree { get; set; } = 1d;

        /// <summary>
        /// Subsample ratio of columns for each split, in each level. Range: (0,1[.
        /// </summary>
        [JsonProperty("colSampleByLevel")]
        public double ColSampleByLevel { get; set; } = 1d;

        /// <summary>
        /// L1 regularization term on weights, increase this value will make model more conservative.
        /// </summary>
        [JsonProperty("alpha")]
        public double Alpha { get; set; }

        /// <summary>
        /// L2 regularization term on weights, increase this value will make model more conservative.
        /// </summary>
        [JsonProperty("lambda")]
        public double Lambda { get; set; } = 1d;

        /// <summary>
        /// The tree construction algorithm used in XGBoost(see description in the reference paper), Distributed and external memory version only support approximate algorithm. Choices: {'auto', 'exact', 'approx'} 'auto': Use heuristic to choose faster one. For small to medium dataset, exact greedy will be used. For very large-dataset, approximate algorithm will be choosed. Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is choosed to notify this choice.  'exact': Exact greedy algorithm. 'approx': Approximate greedy algorithm using sketching and histogram.
        /// </summary>
        [JsonProperty("treeMethod")]
        public XGBoostArgumentsTreeBoosterArgumentsTreeMethodEnum TreeMethod { get; set; } = XGBoostArgumentsTreeBoosterArgumentsTreeMethodEnum.Auto;

        /// <summary>
        /// This is only used for approximate greedy algorithm. This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical ganrantee with sketch accuracy. Usuaully user do not have to tune this. but consider set to lower number for more accurate enumeration. Range: (0,1).
        /// </summary>
        [JsonProperty("sketchEps")]
        public double SketchEps { get; set; } = 0.03d;

        /// <summary>
        /// Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases). If the value is 0, the parameter will be estimated as suggested in https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py#L31 for a binary classification problem.
        /// </summary>
        [JsonProperty("scalePosWeight")]
        public double ScalePosWeight { get; set; } = 1d;

        public override string ComponentName => "egbtree";
    }

    /// <summary>
    /// Linear based booster.
    /// </summary>
    public sealed class ElinearBoosterParameterFunction : BoosterParameterFunction
    {
        /// <summary>
        /// Regularization term on bias, default 0 (no L1 reg on bias because it is not important
        /// </summary>
        [JsonProperty("lambdaBias")]
        public double LambdaBias { get; set; } = 1d;

        /// <summary>
        /// L1 regularization term on weights, increase this value will make model more conservative.
        /// </summary>
        [JsonProperty("alpha")]
        public double Alpha { get; set; }

        /// <summary>
        /// L2 regularization term on weights, increase this value will make model more conservative.
        /// </summary>
        [JsonProperty("lambda")]
        public double Lambda { get; set; }

        public override string ComponentName => "elinear";
    }

    /// <summary>
    /// XGBoost Binary Classifier: https://github.com/dmlc/xgboost.
    /// </summary>
    public sealed partial class XGBoostBinary : Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInputWithGroupId, Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInputWithWeight, Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInputWithLabel, Microsoft.ML.Runtime.EntryPoints.CommonInputs.ITrainerInput, Microsoft.ML.ILearningPipelineItem
    {
        /// <summary>
        /// Number of iterations
        /// </summary>
        [JsonProperty("numBoostRound")]
        public int NumBoostRound { get; set; } = 10;

        /// <summary>
        /// Which booster to use, can be egbtree, gblinear or dart. egbtree and dart use tree based model while gblinear uses linear function.
        /// </summary>
        [JsonConverter(typeof(ComponentSerializer))]
        [JsonProperty("booster")]
        public BoosterParameterFunction Booster { get; set; } = new EgbtreeBoosterParameterFunction();

        /// <summary>
        /// Verbose
        /// </summary>
        [JsonProperty("verboseEval")]
        public bool VerboseEval { get; set; } = false;

        /// <summary>
        /// Printing running messages.
        /// </summary>
        [JsonProperty("silent")]
        public bool Silent { get; set; } = true;

        /// <summary>
        /// Number of parallel threads used to run xgboost.
        /// </summary>
        [JsonProperty("nthread")]
        public int? Nthread { get; set; }

        /// <summary>
        /// Random number seed.
        /// </summary>
        [JsonProperty("seed")]
        public double Seed { get; set; }

        /// <summary>
        /// Evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking). Available choices: rmse, mae, logloss, error, merror, mlogloss, auc, ndcg, map, ndcg, gammaDeviance
        /// </summary>
        [JsonProperty("evalMetric")]
        public XGBoostArgumentsEvalMetric EvalMetric { get; set; } = XGBoostArgumentsEvalMetric.DefaultMetric;

        /// <summary>
        /// Saves internal XGBoost DMatrix as binary format for debugging puropose
        /// </summary>
        [JsonProperty("saveXGBoostDMatrixAsBinary")]
        public string SaveXGBoostDMatrixAsBinary { get; set; }

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
        public NormalizeOption NormalizeFeatures { get; set; } = NormalizeOption.Auto;

        /// <summary>
        /// Whether learner should cache input training data
        /// </summary>
        public CachingOptions Caching { get; set; } = CachingOptions.Auto;


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
                    throw new InvalidOperationException($"{ nameof(XGBoostBinary)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                }

                TrainingData = dataStep.Data;
            }
            Output output = EntryPointXGBoostBinary.Add(experiment, this);
            return new XGBoostBinaryPipelineStep(output);
        }

        private class XGBoostBinaryPipelineStep : ILearningPipelinePredictorStep
        {
            public XGBoostBinaryPipelineStep(Output output)
            {
                Model = output.PredictorModel;
            }

            public Var<IPredictorModel> Model { get; }
        }
    }
    #endregion
}
