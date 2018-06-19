// See the LICENSE file in the project root for more information.

using System;
using Float = System.Single;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using XGBoostMulticlassPredictor = Microsoft.ML.XGBoostWrappers.XGBoostMulticlassPredictor;

[assembly: LoadableClass(typeof(XGBoostMulticlassPredictor), null, typeof(SignatureLoadModel),
    XGBoostMulticlassPredictor.LoaderSignature)]

namespace Microsoft.ML.XGBoostWrappers
{
    public class XGBoostMulticlassPredictor : XGBoostPredictorBase<VBuffer<Float>>
    {
        public const string LoaderSignature = "eXGBoostMulticlass";

        /// <summary>
        /// Number of non empty classes. XGBoost does not allow empty classes and requires to know this number before training.
        /// This number is equal to _classMapping.Length when this array is populated.
        /// </summary>
        private readonly int _nbClass;

        /// <summary>
        /// The output is different when the label is R4.
        /// </summary>
        private readonly bool _isFloatLabel;

        /// <summary>
        /// Mapping from XGBoost classes to Microsoft.ML classes.
        /// _classMapping[i] = ith non empty Microsoft.ML class
        /// If this array is null, the mapping is the identity.
        /// </summary>
        private readonly int[] _classMapping;

        public override ColumnType OutputType { get { return new VectorType(NumberType.R4, _nbClass); } }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EXGBMCLC",
                verWrittenCur: 0x00010003,
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public static XGBoostMulticlassPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var h = env.Register(LoaderSignature);
            return h.Apply("Loading Model", ch => new XGBoostMulticlassPredictor(env, ctx));
        }

        private XGBoostMulticlassPredictor(IHostEnvironment env, ModelLoadContext ctx) : base(env, LoaderSignature, ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // <base>
            // byte[]: xgboost model
            // Int32: number of non empty classes (same number for Microsoft.ML and XGBoost)
            // Int32[]: bijective mapping from xgboost classes to non empty Microsoft.ML classes
            // bool: isFloatLabel

            _nbClass = ctx.Reader.ReadInt32();
            Host.CheckDecode(_nbClass > 0);
            if (ctx.Header.ModelVerWritten > 0x00010001)
            {
                _classMapping = ctx.Reader.ReadIntArray();
                Host.CheckDecode(_classMapping == null || (Utils.IsIncreasing(0, _classMapping, Int32.MaxValue) && _classMapping.Length <= _nbClass));
                _isFloatLabel = ctx.Reader.ReadBoolByte();
            }
            else
                _classMapping = null;
        }

        /// <summary>
        /// Internal constructor.
        /// </summary>
        /// <param name="env">IHostEnvironment</param>
        /// <param name="model">XGBoost model serialized as a binary string</param>
        /// <param name="nbClass">Number of non empty classes</param>
        /// <param name="classMapping">Arrays of Microsoft.ML classes: classMapping[i] = ith Microsoft.ML non empty class, null means identity</param>
        /// <param name="isFloatLabel">Is the label R4 (true) or a Key (false)</param>
        /// <param name="nbFeatrues">number of features</param>
        internal XGBoostMulticlassPredictor(IHostEnvironment env, byte[] model, int nbClass, int[] classMapping, bool isFloatLabel,
                        int nbFeaturesXGBoost, int nbFeaturesML)
            : base(env, LoaderSignature, model, nbFeaturesXGBoost, nbFeaturesML)
        {
            Host.Assert(nbClass > 0, "nbClass");
            Host.Assert(classMapping == null || (classMapping.Length <= nbClass &&
                Utils.IsIncreasing(0, classMapping, Int32.MaxValue)), "classMapping");
            _nbClass = nbClass;
            _classMapping = classMapping;
            _isFloatLabel = isFloatLabel;
        }

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.MultiClassClassification; }
        }

        public override ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            var outSchema = new MulticlassSchema(Host, _nbClass, _classMapping, _isFloatLabel, schema);
            return new XGBoostMulticlassRowMapper(schema, this, env, outSchema, outSchema.ClassMapping, outSchema.NbClass);
        }

        public override ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapperVFloat() as ValueMapper<TSrc, TDst>;
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // <base>
            // byte[]: xgboost model
            // Int32: number of non empty classes (same number for Microsoft.ML and XGBoost)
            // Int32[]: bijective mapping from xgboost classes to non empty Microsoft.ML classes
            // bool: isFloatLabel

            base.SaveCore(ctx);
            Host.Assert(_nbClass > 0);
            ctx.Writer.Write(_nbClass);
            Host.Assert(_classMapping == null || _classMapping.Length <= _nbClass);
            Host.Assert(Utils.IsIncreasing(0, _classMapping, Int32.MaxValue));
            ctx.Writer.WriteIntArray(_classMapping);
            ctx.Writer.WriteBoolByte(_isFloatLabel);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private sealed class MulticlassSchema : ScoreMapperSchemaBase
        {
            private readonly int[] _classMapping;

            public override int ColumnCount { get { return 1; } }

            public int[] ClassMapping { get { return _classMapping; } }
            public int NbClass { get { return ScoreType.AsVector.GetDim(0); } }

            private static int ExtractNbClass(IHost host, int nbClass, int[] classMapping, bool isFloatLabel)
            {
                if (isFloatLabel)
                {
                    host.CheckValue(classMapping, nameof(classMapping));
                    return classMapping.Max() + 1;
                }
                else
                    return nbClass;
            }

            public MulticlassSchema(IHost host, int nbClass, int[] classMapping, bool isFloatLabel, RoleMappedSchema unused)
                : base(new VectorType(NumberType.Float, ExtractNbClass(host, nbClass, classMapping, isFloatLabel)),
                       MetadataUtils.Const.ScoreColumnKind.MultiClassClassification)
            {
                host.CheckValue(unused, nameof(unused));
                _classMapping = classMapping;
                if (unused.Label != null)
                {
                    // unused.Label is usually null and does not necessary reflect the label type used to train the model.
                    if (unused.Label.Type == NumberType.R4)
                    {
                        host.CheckValue(classMapping, nameof(classMapping), "Must not be null when the label is R4.");
                        Contracts.Assert(nbClass == classMapping.Distinct().Count());
                    }
                    else if (!unused.Label.Type.IsKey)
                        throw host.ExceptNotSupp("Label must be a key or R4.");
                }
                else
                {
                    Contracts.Assert(classMapping == null || nbClass >= classMapping.Distinct().Count());
                }
            }
        }
    }
}
