// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;

using XGBoostArguments = Scikit.ML.XGBoostWrapper.XGBoostArguments;
using SignatureBooster = Scikit.ML.XGBoostWrapper.SignatureBooster;

[assembly: LoadableClass(XGBoostArguments.TreeBooster.Summary, typeof(XGBoostArguments.TreeBooster), typeof(XGBoostArguments.TreeBooster.Arguments),
    typeof(SignatureBooster), "egbtree")]
[assembly: LoadableClass(XGBoostArguments.DartBooster.Summary, typeof(XGBoostArguments.DartBooster), typeof(XGBoostArguments.DartBooster.Arguments),
    typeof(SignatureBooster), "edart")]
[assembly: LoadableClass(XGBoostArguments.LinearBooster.Summary, typeof(XGBoostArguments.LinearBooster), typeof(XGBoostArguments.LinearBooster.Arguments),
    typeof(SignatureBooster), "egblinear")]

namespace Scikit.ML.XGBoostWrapper
{
    public delegate void SignatureBooster();

    /// <summary>
    /// Parameters names comes from XGBoost library. 
    /// They were not renamed. XGBoost documentation still applies.
    /// See http://xgboost.readthedocs.io/en/latest/parameter.html,
    /// https://github.com/dmlc/xgboost/blob/master/doc/parameter.md for detailed explanation.
    /// About sweeping, see https://github.com/dmlc/xgboost/blob/master/doc/how_to/param_tuning.md.
    /// </summary>
    public sealed class XGBoostArguments : LearnerInputBaseWithGroupId
    {
        [TlcModule.ComponentKind("XGBoosterParameterFunction")]
        public interface ISupportXGBoosterParameterFactory : IComponentFactory<IBoosterParameter>
        {
        }

        public interface IBoosterParameter
        {
            void UpdateParameters(Dictionary<string, string> res);
        }

        public abstract class BoosterParameter<TArgs> : IBoosterParameter
            where TArgs : class, new()
        {
            protected TArgs _args { private set; get; }

            protected BoosterParameter(TArgs args)
            {
                _args = args;
            }

            public virtual void UpdateParameters(Dictionary<string, string> res)
            {
            }
        }

        public sealed class TreeBooster : BoosterParameter<TreeBooster.Arguments>
        {
            public const string Summary = "Parameters for TreeBooster, boost=egbtree{...}.";
            public const string Name = "xgbdt";
            public const string FriendlyName = "Tree XGBooster";

            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Traditional Gradient XGBoosting Decision Tree.")]
            public class Arguments : ISupportXGBoosterParameterFactory
            {
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly " +
                        "get the weights of new features and eta actually shrinks the feature weights to make the boosting process more conservative. Range: [0,1].",
                        ShortName = "lr")]
                public double learningRate = 0.3;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, " +
                        "the more conservative the algorithm will be. Range: [0,inf[.")]
                public double gamma = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Maximum depth of a tree, increase this value will make model more complex / likely to be overfitting. Range: [1,inf[.")]
                public int maxDepth = 6;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf " +
                        "node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, " +
                        "this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. Range: [0,inf[.")]
                public int minChildWeight = 1;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there " +
                        "is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might " +
                        "help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update. Range: [0,inf[.")]
                public double maxDeltaStep = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected " +
                        "half of the data instances to grow trees and this will prevent overfitting. Range: (0,1].")]
                public double subSample = 1;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Subsample ratio of columns when constructing each tree. Range: (0,1].")]
                public double colSampleByTree = 1;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Subsample ratio of columns for each split, in each level. Range: (0,1[.")]
                public double colSampleByLevel = 1;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "L2 regularization term on weights, increase this value will make model more conservative.",
                    ShortName = "l2")]
                [TGUI(Label = "Lambda(L2)", SuggestedSweeps = "0,0.5,1")]
                public double lambda = 1;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "L1 regularization term on weights, increase this value will make model more conservative.",
                    ShortName = "l1")]
                [TGUI(Label = "Alpha(L1)", SuggestedSweeps = "0,0.5,1")]
                public double alpha = 0;

                public enum TreeMethodEnum
                {
                    Auto,
                    Approx,
                    Exact
                };

                [Argument(ArgumentType.AtMostOnce,
                    HelpText =
                        "The tree construction algorithm used in XGBoost(see description in the reference paper), " +
                        "Distributed and external memory version only support approximate algorithm. " +
                        "Choices: {'auto', 'exact', 'approx'} " +
                        "'auto': Use heuristic to choose faster one. " +
                        "For small to medium dataset, exact greedy will be used. " +
                        "For very large-dataset, approximate algorithm will be choosed. " +
                        "Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is choosed to notify this choice. " +
                        " 'exact': Exact greedy algorithm. " +
                        "'approx': Approximate greedy algorithm using sketching and histogram.")]
                public TreeMethodEnum treeMethod = TreeMethodEnum.Auto;

                [Argument(ArgumentType.AtMostOnce, HelpText = "This is only used for approximate greedy algorithm. " +
                                                              "This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical ganrantee with sketch accuracy. " +
                                                              "Usuaully user do not have to tune this. but consider set to lower number for more accurate enumeration. Range: (0,1).")]
                public double sketchEps = 0.03;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases). " +
                    "If the value is 0, the parameter will be estimated as suggested in " +
                    "https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py#L31 for a binary classification problem.")]
                public double scalePosWeight = 1;

                public virtual IBoosterParameter CreateComponent(IHostEnvironment env) => new TreeBooster(this);
            }

            public TreeBooster(Arguments args)
                : base(args)
            {
                Contracts.CheckUserArg(_args.gamma >= 0, nameof(_args.gamma), "must be >= 0.");
                Contracts.CheckUserArg(_args.maxDepth > 1, nameof(_args.maxDepth), "must be > 1.");
                Contracts.CheckUserArg(_args.minChildWeight >= 0, nameof(_args.minChildWeight), "must be >= 0.");
                Contracts.CheckUserArg(_args.maxDeltaStep >= 0, nameof(_args.maxDeltaStep), "must be >= 0.");
                Contracts.CheckUserArg(!(_args.subSample <= 0 || _args.subSample > 1), nameof(_args.subSample), "must be in (0,1].");
                Contracts.CheckUserArg(!(_args.colSampleByTree <= 0 || _args.colSampleByTree > 1), nameof(_args.colSampleByTree), "must be in (0,1].");
                Contracts.CheckUserArg(!(_args.colSampleByLevel <= 0 || _args.colSampleByLevel > 1), nameof(_args.colSampleByLevel), "must be in (0,1].");
                Contracts.CheckUserArg(!(_args.sketchEps <= 0 || _args.sketchEps > 1), nameof(_args.sketchEps), "must be in (0,1].");
                Contracts.CheckUserArg(!(_args.scalePosWeight < 0 || _args.scalePosWeight > 1), nameof(_args.scalePosWeight), "must be in [0,1].");
            }

            public override void UpdateParameters(Dictionary<string, string> res)
            {
                base.UpdateParameters(res);
                res["learning_rate"] = _args.learningRate.ToString();
                res["gamma"] = _args.gamma.ToString();
                res["max_depth"] = _args.maxDepth.ToString();
                res["min_child_weight"] = _args.minChildWeight.ToString();
                res["max_delta_step"] = _args.maxDeltaStep.ToString();
                res["subsample"] = _args.subSample.ToString();
                res["colsample_bytree"] = _args.colSampleByTree.ToString();
                res["colsample_bylevel"] = _args.colSampleByLevel.ToString();
                res["alpha"] = _args.alpha.ToString();
                res["tree_method"] = _args.treeMethod.ToString().ToLower();
                res["sketch_eps"] = _args.sketchEps.ToString();
                res["scale_pos_weight"] = _args.scalePosWeight.ToString();
                res["lambda"] = _args.lambda.ToString();
                res["booster"] = "gbtree";
            }
        }

        public sealed class DartBooster : BoosterParameter<DartBooster.Arguments>
        {
            public const string Summary = "Parameters for DartBooster (includes parameters for TreeBooster), boost=edart{...}.";
            public const string Name = "xgdart";
            public const string FriendlyName = "Tree Dropout Tree XGBooster";

            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Dropouts meet Multiple Additive Regresion Trees (XGBoost).")]
            public class Arguments : TreeBooster.Arguments
            {
                public enum SampleType
                {
                    Uniform,
                    Weighted
                };

                [Argument(ArgumentType.AtMostOnce, HelpText = "Type of sampling algorithm")]
                public SampleType
                    sampleType = SampleType.Uniform;

                public enum NormalizeType
                {
                    Tree,
                    Forest
                };

                [Argument(ArgumentType.AtMostOnce, HelpText = "Type of normalization algorithm")]
                public NormalizeType normalizeType = NormalizeType.Tree;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Dropout rate.  Range: [0, 1]")]
                public double rateDrop = 0.0;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Probability of skip dropout.  Range: [0, 1].")]
                public double skipDrop = 0.0;
            }

            public DartBooster(Arguments args)
                : base(args)
            {
                Contracts.CheckUserArg(!(_args.rateDrop < 0 || _args.rateDrop > 1), "rateDrop", "must be in [0,1].");
                Contracts.CheckUserArg(!(_args.skipDrop < 0 || _args.skipDrop > 1), "skipDrop", "must be in [0,1].");
            }

            public override void UpdateParameters(Dictionary<string, string> res)
            {
                base.UpdateParameters(res);
                var tb = new TreeBooster(_args);
                tb.UpdateParameters(res);
                res["sample_type"] = _args.sampleType.ToString().ToLower();
                res["normalize_type"] = _args.normalizeType.ToString().ToLower();
                res["rate_drop"] = _args.rateDrop.ToString();
                res["skip_drop"] = _args.skipDrop.ToString();
                res["booster"] = "dart";
            }
        }

        public sealed class LinearBooster : BoosterParameter<LinearBooster.Arguments>
        {
            public const string Summary = "Parameters for LinearBooster, boost=elinear{...}.";
            public const string Name = "xglinear";
            public const string FriendlyName = Name;

            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Linear training (XGBoost).")]
            public class Arguments
            {
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Regularization term on bias, default 0 (no L1 reg on bias because it is not important",
                    ShortName = "l2b")]
                public double lambdaBias = 1;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "L2 regularization term on weights, increase this value will make model more conservative.",
                    ShortName = "l2")]
                [TGUI(Label = "Lambda(L2)", SuggestedSweeps = "0,0.5,1")]
                public double lambda = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText =
                        "L1 regularization term on weights, increase this value will make model more conservative.",
                    ShortName = "l1")]
                [TGUI(Label = "Alpha(L1)", SuggestedSweeps = "0,0.5,1")]
                public double alpha = 0;
            }

            public LinearBooster(Arguments args)
                : base(args)
            {
            }

            public override void UpdateParameters(Dictionary<string, string> res)
            {
                base.UpdateParameters(res);
                res["alpha"] = _args.alpha.ToString();
                res["lambda"] = _args.lambda.ToString();
                res["lambda_bias"] = _args.lambdaBias.ToString();
                res["booster"] = "gblinear";
            }
        }

        public enum EvalMetric
        {
            DefaultMetric,
            Rmse,
            Mae,
            Logloss,
            Error,
            Merror,
            Mlogloss,
            Auc,
            Ndcg,
            Map,
            NdcgAT1,
            NdcgAT3,
            NdcgAT5,
            MapAT1,
            MapAT3,
            MapAT5,
        };

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations", SortOrder = 1, ShortName = "iter")]
        [TGUI(Label = "Number of boosting iterations", SuggestedSweeps = "10,20,50,100,150,200")]
        public int numBoostRound = 10;

        [Argument(ArgumentType.Multiple, HelpText = "Which booster to use, can be egbtree, gblinear or dart. egbtree and dart use tree based model while gblinear uses linear function.", SortOrder = 3)]
        public SubComponent<IBoosterParameter, SignatureBooster> booster = new SubComponent<IBoosterParameter, SignatureBooster>("egbtree");

        [Argument(ArgumentType.AtMostOnce, HelpText = "Verbose", ShortName = "v")]
        public bool verboseEval = false;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Printing running messages.")]
        public bool silent = true;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of parallel threads used to run xgboost.", ShortName = "nt")]
        public int? nthread;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Random number seed.")]
        public double seed = 0;

        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Evaluation metrics for validation data, a default metric will be assigned according to objective " +
            "(rmse for regression, and error for classification, mean average precision for ranking). Available choices: " +
            "rmse, mae, logloss, error, merror, mlogloss, auc, ndcg, map, ndcg, gammaDeviance",
            ShortName = "em")]
        public EvalMetric evalMetric = EvalMetric.DefaultMetric;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Saves internal XGBoost DMatrix as binary format for debugging puropose",
            ShortName = "xgbdmsave")]
        public string saveXGBoostDMatrixAsBinary = null;

        public Dictionary<string, string> ToDict(IHostEnvironment env)
        {
            Dictionary<string, string> res = new Dictionary<string, string>();
            res["silent"] = silent ? "1" : "0";
            if (nthread.HasValue)
                res["nthread"] = nthread.Value.ToString();
            res["seed"] = seed.ToString();

            string metric = null;
            switch (evalMetric)
            {
                case EvalMetric.DefaultMetric:
                    break;
                case EvalMetric.Rmse:
                case EvalMetric.Mae:
                case EvalMetric.Logloss:
                case EvalMetric.Error:
                case EvalMetric.Merror:
                case EvalMetric.Mlogloss:
                case EvalMetric.Auc:
                case EvalMetric.Ndcg:
                case EvalMetric.Map:
                    metric = evalMetric.ToString().ToLower();
                    break;
                case EvalMetric.NdcgAT1:
                case EvalMetric.NdcgAT3:
                case EvalMetric.NdcgAT5:
                case EvalMetric.MapAT1:
                case EvalMetric.MapAT3:
                case EvalMetric.MapAT5:
                    metric = evalMetric.ToString().Replace("AT", "@").ToLower();
                    break;
            }
            if (!string.IsNullOrEmpty(metric))
                res["eval_metric"] = metric;
            var boosterParams = booster.CreateInstance(env);
            boosterParams.UpdateParameters(res);
            return res;
        }
    }
}
