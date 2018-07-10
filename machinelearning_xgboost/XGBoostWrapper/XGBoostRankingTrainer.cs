// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using XGBoostRankingTrainer = Scikit.ML.XGBoostWrapper.XGBoostRankingTrainer;
using XGBoostArguments = Scikit.ML.XGBoostWrapper.XGBoostArguments;

[assembly: LoadableClass(XGBoostRankingTrainer.Summary, typeof(XGBoostRankingTrainer), typeof(XGBoostArguments),
    new[] { typeof(SignatureRankerTrainer), typeof(SignatureTrainer) },
    "eXGBoost Ranking", "eXGBoostRanking", "eXGBoostRank", "exgbrk", "xgbrk")]

namespace Scikit.ML.XGBoostWrapper
{
    public sealed class XGBoostRankingTrainer : XGBoostTrainerBase<Float, XGBoostRankingPredictor>
    {
        public const string Summary = "XGBoost Ranking: https://github.com/dmlc/xgboost.";

        public XGBoostRankingTrainer(IHostEnvironment env, XGBoostArguments args) : base(env, args, PredictionKind.Ranking, "eXGBoostRanking")
        {
        }

        protected override void ValidateTrainInput(IChannel ch, RoleMappedData data)
        {
            base.ValidateTrainInput(ch, data);
            _host.CheckParam(data.Schema.Group != null, nameof(data.Schema.Group), "Need a group column");
            var groupType = data.Schema.Group.Type;
            _host.CheckParam(groupType.RawKind == DataKind.U4 || (groupType.IsKey && groupType.AsKey.RawKind == DataKind.U4),
                "data", $"Group column '{data.Schema.Name}' is of type '{groupType}', but must be U4 or Key<U4, *>.");
            if (!(groupType.RawKind == DataKind.U4 || (groupType.IsKey && groupType.AsKey.RawKind == DataKind.U4)))
                throw _host.ExceptParam(nameof(data), "Label column 'Group' is of type '{0}', but must be U4 or Key<U4, *>.", groupType);
        }

        public override XGBoostRankingPredictor CreatePredictor()
        {
            _host.CheckValue(_model, "Must have trained a model before creating a predictor.");
            return new XGBoostRankingPredictor(_model, _host, _nbFeaturesXGboost, _nbFeaturesML);
        }

        protected override void UpdateXGBoostOptions(IChannel ch, Dictionary<string, string> options, Float[] labels, uint[] groups)
        {
            Contracts.AssertValue(ch, nameof(ch));
            ch.AssertValue(options, nameof(options));
            ch.AssertValue(labels, nameof(labels));
            ch.AssertValue(groups, nameof(groups));
            if (!options.ContainsKey("objective"))
                options["objective"] = "rank:pairwise";
            ch.CheckValue(groups, nameof(groups));
        }
    }
}
