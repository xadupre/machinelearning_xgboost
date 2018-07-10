// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;

namespace Scikit.ML.XGBoostWrapper
{
    /// <summary>
    /// Implements a base class for all predictor in XGBoost.
    /// </summary>
    /// <typeparam name="TOutput"></typeparam>
    public abstract class XGBoostPredictorBase<TOutput> : PredictorBase<TOutput>, ISchemaBindableMapper, ICanSaveModel, IValueMapper
    {
        /// <summary>
        /// XGBoost does not return the same output as Microsoft.ML. 
        /// For Binary Classification, the raw output is [0, 1].
        /// </summary>
        /// <param name="output"></param>
        public delegate void UpdateOutputType(ref TOutput output);

        private readonly Booster _booster;
        private readonly int _numFeaturesML;
        private ColumnType _inputType;

        public ColumnType InputType { get { return _inputType; } }
        public abstract ColumnType OutputType { get; }

        internal Booster GetBooster() { return _booster; }

        public int GetNumTrees() { return _booster.GetNumTrees(); }

        /// <summary>
        /// This function create a a delegate function which post process the output of the predictor.
        /// It should be empty. Check parameter outputMargin when calling the predictions before tweaking XGBoost output.
        /// </summary>
        public virtual UpdateOutputType GetOutputPostProcessor()
        {
            return (ref TOutput src) => { };
        }

        protected XGBoostPredictorBase(IHostEnvironment env, string name, byte[] model, int numFeaturesXGBoost, int numFeaturesML) : base(env, name)
        {
            env.Check(numFeaturesXGBoost > 0, nameof(numFeaturesXGBoost));
            env.Check(numFeaturesML >= numFeaturesXGBoost, nameof(numFeaturesML));
            _booster = new Booster(model, numFeaturesXGBoost);
            _numFeaturesML = numFeaturesML;
            _inputType = new VectorType(NumberType.R4, _numFeaturesML);
        }

        protected XGBoostPredictorBase(IHostEnvironment env, string name, ModelLoadContext ctx) : base(env, name, ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // <base>
            // int
            // int (version > 0x00010003)

            byte[] model = null;
            bool load = ctx.TryLoadBinaryStream("xgboost.model", br =>
            {
                using (var memStream = new MemoryStream())
                {
                    br.BaseStream.CopyTo(memStream);
                    model = memStream.ToArray();
                }
            });

            Host.CheckDecode(load);
            Host.CheckDecode(model != null && model.Length > 0);
            int numFeatures = ctx.Reader.ReadInt32();
            Host.CheckDecode(numFeatures > 0);
            // The XGBoost model is loaded, if it fails, it probably means that the model is corrupted
            // or XGBoost library changed its format. The error message comes from XGBoost.
            _booster = new Booster(model, numFeatures);
            if (ctx.Header.ModelVerWritten >= 0x00010003)
                _numFeaturesML = ctx.Reader.ReadInt32();
            else
                _numFeaturesML = _booster.NumFeatures;
            Host.CheckDecode(_numFeaturesML >= numFeatures);
            _inputType = new VectorType(NumberType.R4, _numFeaturesML);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();

            // *** Binary format ***
            // <base>
            // int (_booster.NumFeatures)
            // int (_numFeaturesML) (version >= 0x00010003)

            base.SaveCore(ctx);
            ctx.SaveBinaryStream("xgboost.model", bw => bw.Write(_booster.SaveRaw()));
            ctx.Writer.Write(_booster.NumFeatures);
            ctx.Writer.Write(_numFeaturesML);
        }

        public abstract ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema);

        public abstract ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>();

        protected ValueMapper<VBuffer<float>, VBuffer<float>> GetMapperVFloat()
        {
            var buffer = Booster.CreateInternalBuffer();
            return (ref VBuffer<float> src, ref VBuffer<float> dst) =>
            {
                _booster.PredictOneOff(ref src, ref dst, ref buffer);
            };
        }

        protected ValueMapper<VBuffer<float>, float> GetMapperFloat()
        {
            var buffer = Booster.CreateInternalBuffer();
            VBuffer<float> dstBuffer = new VBuffer<float>();
            return (ref VBuffer<float> src, ref float dst) =>
            {
                _booster.PredictOneOff(ref src, ref dstBuffer, ref buffer);
                dst = dstBuffer.Values[0];
            };
        }
    }
}
