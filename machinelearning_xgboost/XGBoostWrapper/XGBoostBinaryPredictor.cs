// See the LICENSE file in the project root for more information.

using Float = System.Single;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using XGBoostBinaryPredictor = Scikit.ML.XGBoostWrapper.XGBoostBinaryPredictor;

[assembly: LoadableClass(typeof(XGBoostBinaryPredictor), null, typeof(SignatureLoadModel),
    XGBoostBinaryPredictor.LoaderSignature)]

namespace Scikit.ML.XGBoostWrapper
{
    /// <summary>
    /// XGBoost binary predictor.
    /// </summary>
    public sealed class XGBoostBinaryPredictor : XGBoostPredictorBase<Float>
    {
        public const string LoaderSignature = "eXGBoostBinary";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EXGBBINC",
                verWrittenCur: 0x00010003,
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public override ColumnType OutputType { get { return NumberType.R4; } }

        public static XGBoostBinaryPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, "env");
            env.CheckValue(ctx, "ctx");
            ctx.CheckAtModel(GetVersionInfo());
            var h = env.Register(LoaderSignature);
            return h.Apply("Loading Model", ch => new XGBoostBinaryPredictor(env, ctx));
        }

        private XGBoostBinaryPredictor(IHostEnvironment env, ModelLoadContext ctx) : base(env, LoaderSignature, ctx)
        {
        }

        internal XGBoostBinaryPredictor(IHostEnvironment env, byte[] model, int nbFeaturesXGBoost, int nbFeaturesML)
            : base(env, LoaderSignature, model, nbFeaturesXGBoost, nbFeaturesML)
        {
        }

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.BinaryClassification; }
        }

        public override ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            return new XGBoostScalarRowMapper(schema, this, env, new ScoreMapperSchema(NumberType.Float, MetadataUtils.Const.ScoreColumnKind.BinaryClassification));
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public override ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapperFloat() as ValueMapper<TSrc, TDst>;
        }
    }
}
