// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Scikit.ML.XGBoostWrapper
{
    internal sealed class XGBoostMulticlassRowMapper : XGBoostScalarRowMapperBase<VBuffer<Float>>
    {
        private readonly int[] _classMapping;
        private readonly int _numberOfClasses;

        public XGBoostMulticlassRowMapper(RoleMappedSchema schema, XGBoostMulticlassPredictor parent, IHostEnvironment env,
                                   ISchema outputSchema, int[] classMapping, int numberOfClasses)
            : base(schema, parent, env, outputSchema)
        {
            env.Assert(outputSchema.ColumnCount == 1, "outputSchema");
            env.Assert(outputSchema.GetColumnType(0).IsVector, "outputSchema");
            env.Assert(outputSchema.GetColumnType(0).ItemType.IsNumber, "outputSchema");
            env.Assert(classMapping == null || Utils.IsIncreasing(0, _classMapping, int.MaxValue), "classMapping");
            _classMapping = classMapping;
            _numberOfClasses = numberOfClasses;
        }

        protected override Delegate[] CreatePredictionGetters(Booster xgboostModel, IRow input, Func<int, bool> predicate)
        {
            var active = Utils.BuildArray(OutputSchema.ColumnCount, predicate);
            xgboostModel.LazyInit();
            var getters = new Delegate[1];
            if (active[0])
            {
                var featureGetter = RowCursorUtils.GetVecGetterAs<Float>(PrimitiveType.FromKind(DataKind.R4), input, InputSchema.Feature.Index);
                VBuffer<Float> features = new VBuffer<Float>();
                var postProcessor = Parent.GetOutputPostProcessor();
                int expectedLength = input.Schema.GetColumnType(InputSchema.Feature.Index).VectorSize;
                var xgboostBuffer = Booster.CreateInternalBuffer();
                int nbMappedClasses = _classMapping == null ? 0 : _numberOfClasses;

                if (nbMappedClasses == 0)
                {
                    ValueGetter<VBuffer<Float>> localGetter = (ref VBuffer<Float> prediction) =>
                    {
                        featureGetter(ref features);
                        Contracts.Assert(features.Length == expectedLength);
                        xgboostModel.Predict(ref features, ref prediction, ref xgboostBuffer);
                        postProcessor(ref prediction);
                    };
                    getters[0] = localGetter;
                }
                else
                {
                    ValueGetter<VBuffer<Float>> localGetter = (ref VBuffer<Float> prediction) =>
                    {
                        featureGetter(ref features);
                        Contracts.Assert(features.Length == expectedLength);
                        xgboostModel.Predict(ref features, ref prediction, ref xgboostBuffer);
                        Contracts.Assert(prediction.IsDense);
                        postProcessor(ref prediction);
                        var indices = prediction.Indices;
                        if (indices == null || indices.Length < _classMapping.Length)
                            indices = new int[_classMapping.Length];
                        Array.Copy(_classMapping, indices, _classMapping.Length);
                        prediction = new VBuffer<float>(nbMappedClasses, _classMapping.Length, prediction.Values, indices);
                    };
                    getters[0] = localGetter;
                }
            }
            return getters;
        }
    }
}
