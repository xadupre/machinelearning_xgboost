// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.XGBoostWrappers
{
    /// <summary>
    /// Wraps <see cref="XGBoost"/> into Microsoft.ML <see cref="ISchemaBoundRowMapper"/> to produce prediction for given input.
    /// </summary>
    internal abstract class XGBoostScalarRowMapperBase<TOutput> : ISchemaBoundRowMapper
    {
        private readonly XGBoostPredictorBase<TOutput> _parent;
        private readonly RoleMappedSchema _inputSchema;
        private readonly ISchema _outputSchema;
        private readonly List<int> _inputCols;
        private readonly Booster _booster;

        protected XGBoostPredictorBase<TOutput> Parent { get { return _parent; } }

        public XGBoostScalarRowMapperBase(RoleMappedSchema schema, XGBoostPredictorBase<TOutput> parent, IHostEnvironment env, ISchema outputSchema)
        {
            Contracts.AssertValue(env, "env");
            env.AssertValue(schema, "schema");
            env.AssertValue(parent, "parent");
            env.AssertValue(schema.Feature, "schema");

            // REVIEW xadupre: only one feature columns is allowed.
            // This should be revisited in the future.
            // XGBoost has plans for others types.
            // Look at https://github.com/dmlc/xgboost/issues/874.
            env.Check(schema.Feature != null, "Unexpected number of feature columns, 1 expected.");

            _parent = parent;
            var columns = new[] { schema.Feature };
            var fc = new[] { new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, columns[0].Name) };
            _inputSchema = RoleMappedSchema.Create(schema.Schema, fc);
            _outputSchema = outputSchema;

            _inputCols = new List<int>();
            foreach (var kvp in columns)
            {
                int index;
                if (schema.Schema.TryGetColumnIndex(kvp.Name, out index))
                    _inputCols.Add(index);
                else
                    Contracts.Assert(false);
            }

            _booster = _parent.GetBooster();
        }

        public IRow GetOutputRow(IRow input, Func<int, bool> predicate, out Action disposer)
        {
            Contracts.AssertValue(input);
            Contracts.AssertValue(predicate);
            Contracts.CheckParam(input.Schema == _inputSchema.Schema, nameof(input), "Input schema mismatches bound schema");

            var getters = CreatePredictionGetters(_booster, input, predicate);
            disposer = null;
            return new SimpleRow(OutputSchema, input, getters);
        }

        protected abstract Delegate[] CreatePredictionGetters(Booster xgboostModel, IRow input, Func<int, bool> predicate);

        public RoleMappedSchema InputSchema { get { return _inputSchema; } }

        public ISchema OutputSchema
        {
            get { return _outputSchema; }
        }

        public ISchemaBindableMapper Bindable { get { return _parent; } }

        public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
        {
            return _inputSchema.GetColumnRoles().Select(kvp =>
                new KeyValuePair<RoleMappedSchema.ColumnRole, string>(kvp.Key, kvp.Value.Name));
        }

        public Func<int, bool> GetDependencies(Func<int, bool> predicate)
        {
            if (Enumerable.Range(0, _outputSchema.ColumnCount).Any(predicate))
                return index => _inputCols.Any(ife => ife == index);
            else
                return _ => false;
        }
    }
}
