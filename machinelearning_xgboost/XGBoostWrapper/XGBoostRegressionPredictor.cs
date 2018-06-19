// See the LICENSE file in the project root for more information.

using Float = System.Single;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using XGBoostRegressionPredictor = Microsoft.ML.XGBoostWrappers.XGBoostRegressionPredictor;

[assembly: LoadableClass(typeof(XGBoostRegressionPredictor), null, typeof(SignatureLoadModel),
    XGBoostRegressionPredictor.LoaderSignature)]

namespace Microsoft.ML.XGBoostWrappers
{
    public class XGBoostRegressionPredictor : XGBoostPredictorBase<Float>
    {
        public const string LoaderSignature = "eXGBoostRegression";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EXGBSIRG",
                verWrittenCur: 0x00010003,
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public override ColumnType OutputType { get { return NumberType.R4; } }

        public static XGBoostRegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            var h = env.Register(LoaderSignature);
            return h.Apply("Loading Model", ch => new XGBoostRegressionPredictor(env, ctx));
        }

        private XGBoostRegressionPredictor(IHostEnvironment env, ModelLoadContext ctx) : base(env, LoaderSignature, ctx)
        {
        }

        internal XGBoostRegressionPredictor(byte[] model, IHostEnvironment env, int nbFeaturesXGBoost, int nbFeaturesML) : 
            base(env, LoaderSignature, model, nbFeaturesXGBoost, nbFeaturesML)
        {
        }

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.Regression; }
        }

        public override ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            return new XGBoostScalarRowMapper(schema, this, env, new ScoreMapperSchema(NumberType.Float, MetadataUtils.Const.ScoreColumnKind.Regression));
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
