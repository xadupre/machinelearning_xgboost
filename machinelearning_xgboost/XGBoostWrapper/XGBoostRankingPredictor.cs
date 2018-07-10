// See the LICENSE file in the project root for more information.

using Float = System.Single;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using XGBoostRankingPredictor = Scikit.ML.XGBoostWrapper.XGBoostRankingPredictor;

[assembly: LoadableClass(typeof(XGBoostRankingPredictor), null, typeof(SignatureLoadModel),
    XGBoostRankingPredictor.LoaderSignature)]

namespace Scikit.ML.XGBoostWrapper
{
    public class XGBoostRankingPredictor : XGBoostPredictorBase<Float>
    {
        public const string LoaderSignature = "eXGBoostRanking";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EXGBPRNK",
                verWrittenCur: 0x00010003,
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public override ColumnType OutputType { get { return NumberType.R4; } }

        public static XGBoostRankingPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var h = env.Register(LoaderSignature);
            return h.Apply("Loading Model", ch => new XGBoostRankingPredictor(env, ctx));
        }

        private XGBoostRankingPredictor(IHostEnvironment env, ModelLoadContext ctx) : base(env, LoaderSignature, ctx)
        {
        }

        internal XGBoostRankingPredictor(byte[] model, IHostEnvironment env, int nbFeaturesXGBoost, int nbFeaturesML) : 
            base(env, LoaderSignature, model, nbFeaturesXGBoost, nbFeaturesML)
        {
        }

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.Ranking; }
        }

        public override ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            return new XGBoostScalarRowMapper(schema, this, env, new ScoreMapperSchema(NumberType.Float, MetadataUtils.Const.ScoreColumnKind.Ranking));
        }

        public override ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapperFloat() as ValueMapper<TSrc, TDst>;
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }
    }
}
