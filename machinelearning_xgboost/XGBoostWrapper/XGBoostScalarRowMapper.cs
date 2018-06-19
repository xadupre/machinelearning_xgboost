// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.XGBoostWrappers
{
    internal sealed class XGBoostScalarRowMapper : XGBoostScalarRowMapperBase<Float>
    {
        public XGBoostScalarRowMapper(RoleMappedSchema schema, XGBoostPredictorBase<Float> parent, IHostEnvironment env, ISchema outputSchema)
            : base(schema, parent, env, outputSchema)
        {
            env.CheckParam(outputSchema.ColumnCount == 1, nameof(outputSchema));
            env.CheckParam(outputSchema.GetColumnType(0).IsNumber, nameof(outputSchema));
        }

        protected override Delegate[] CreatePredictionGetters(Booster xgboostModel, IRow input, Func<int, bool> predicate)
        {
            var active = Utils.BuildArray(OutputSchema.ColumnCount, predicate);
            xgboostModel.LazyInit();
            var getters = new Delegate[1];
            if (active[0])
            {
                var featureGetter = RowCursorUtils.GetVecGetterAs<Float>(PrimitiveType.FromKind(DataKind.R4), input, InputSchema.Feature.Index);
                VBuffer<Float> features = default(VBuffer<Float>);
                var postProcessor = Parent.GetOutputPostProcessor();
                VBuffer<Float> prediction = default(VBuffer<Float>);
                int expectedLength = input.Schema.GetColumnType(InputSchema.Feature.Index).VectorSize;
                var xgboostBuffer = Booster.CreateInternalBuffer();

                ValueGetter<Float> localGetter = (ref Float value) =>
                {
                    featureGetter(ref features);
                    Contracts.Assert(features.Length == expectedLength);
                    xgboostModel.Predict(ref features, ref prediction, ref xgboostBuffer);
                    value = prediction.Values[0];
                    postProcessor(ref value);
                };

                getters[0] = localGetter;
            }
            return getters;
        }
    }
}
