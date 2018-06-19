// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.XGBoostWrappers
{
    // Implements the wrapper for XGBoost (https://github.com/dmlc/xgboost)

    /// <summary>
    /// XGBoost DMatrix (inspired from Python code)
    /// DMatrix is a internal data structure that used by XGBoost
    /// which is optimized for both memory efficiency and training speed.
    /// You can construct DMatrix from numpy.arrays
    /// </summary>
    internal sealed class DMatrix
    {
#if(DEBUG)
        // For debugging purposes.
        struct GcKeep
        {
            public unsafe Float[] data;
            public unsafe uint[] groups;
            public unsafe Float[] weights;
            public unsafe Float[] labels;
            public unsafe /*size_t*/ ulong[] indptr;
            public unsafe uint[] indices;
        }
        private GcKeep _gcKeep;
#endif

        readonly string[] _featureNames;
        readonly string[] _featureTypes;
        readonly IntPtr _handle;

        public string[] FeatureNames => _featureNames; 
        public string[] FeatureTypes => _featureTypes; 
        public IntPtr Handle => _handle;

        /// <summary>
        /// Create a dense matrix used in XGBoost.
        /// </summary>
        /// <param name="data">Matrix as a Float array</param>
        /// <param name="nrow">Number of rows</param>
        /// <param name="ncol">Number of columns</param>
        /// <param name="labels">Labels</param>
        /// <param name="missing">Missing value</param>
        /// <param name="weights">Vector of weights (can be null)</param>
        /// <param name="groups">Vector of groups (can be null)</param>
        /// <param name="featureNames">Set names for features.</param>
        /// <param name="featureTypes">Set types for features.</param>
        public DMatrix(Float[] data, uint nrow, uint ncol, Float[] labels = null, Float missing = Float.NaN,
                 Float[] weights = null, uint[] groups = null,
                 IEnumerable<string> featureNames = null, IEnumerable<string> featureTypes = null)
        {
#if(DEBUG)
            _gcKeep = new GcKeep()
            {
                data = data,
                labels = labels,
                weights = weights,
                groups = groups
            };
#endif

            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixCreateFromMat(data, nrow, ncol, missing, ref _handle));

            if (labels != null)
                SetLabel(labels, nrow);
            if (weights != null)
                SetWeight(weights, nrow);
            if (groups != null)
                SetGroups(groups, nrow);

            _featureNames = featureNames == null ? null : featureNames.ToArray();
            _featureTypes = featureTypes == null ? null : featureTypes.ToArray();
        }

        /// <summary>
        /// Create a sparse matrix used in XGBoost.
        /// </summary>
        /// <param name="numColumn">number of features or columns</param>
        /// <param name="indptr">Pointer to row headers</param>
        /// <param name="indices">column indices</param>
        /// <param name="data">Matrix as a Float array</param>
        /// <param name="nrow">Rows in the matix</param>
        /// <param name="nelem">Number of nonzero elements in the matrix</param>
        /// <param name="labels">Labels</param>
        /// <param name="weights">Vector of weights (can be null)</param>
        /// <param name="groups">Vector of groups (can be null)</param>
        /// <param name="featureNames">Set names for features.</param>
        /// <param name="featureTypes">Set types for features.</param>
        public DMatrix(/*bst_ulong*/ uint numColumn, /*size_t*/ ulong[] indptr, uint[] indices, Float[] data,
                 uint nrow, uint nelem, Float[] labels = null,
                 Float[] weights = null, uint[] groups = null,
                 IEnumerable<string> featureNames = null, IEnumerable<string> featureTypes = null)
        {
            Contracts.Assert(nrow + 1 == indptr.Length);
#if(DEBUG)
            _gcKeep = new GcKeep()
            {
                indptr = indptr,
                indices = indices,
                data = data,
                labels = labels,
                weights = weights,
                groups = groups
            };
#endif

#if (XGB_EXTENDED)
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixCreateFromCSREx(indptr,
                indices, data, (ulong)indptr.Length, nelem, numColumn, ref _handle));
#else
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixCreateFromCSR(indptr,
                indices, data, (uint)indptr.Length, nelem, ref _handle));
#endif

            if (labels != null)
                SetLabel(labels, nrow);
            if (weights != null)
                SetWeight(weights, nrow);
            if (groups != null)
                SetGroups(groups, nrow);

            _featureNames = featureNames == null ? null : featureNames.ToArray();
            _featureTypes = featureTypes == null ? null : featureTypes.ToArray();

            Contracts.Assert(nrow == (int)GetNumRows());
            Contracts.Assert((int)GetNumCols() == numColumn);
        }

        ~DMatrix()
        {
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixFree(_handle));
        }

        public void SaveBinary(string name, int silent = 0)
        {
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixSaveBinary(_handle, name, silent));
        }

        public uint GetNumRows()
        {
            uint nb = 0;
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixNumRow(_handle, ref nb));
            return nb;
        }

        public uint GetNumCols()
        {
            uint nb = 0;
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixNumCol(_handle, ref nb));
            return nb;
        }

        /// <summary>
        /// Set label of dmatrix
        /// </summary>
        public void SetLabel(Float[] label, uint nrow)
        {
            SetFloatInfo("label", label, nrow);
        }

        /// <summary>
        /// Set weight of each instance.
        /// </summary>
        public void SetWeight(Float[] weight, uint nrow)
        {
            SetFloatInfo("weight", weight, nrow);
        }

        /// <summary>
        /// Set base margin of booster to start from.
        /// This can be used to specify a prediction value of
        /// existing model to be base_margin
        /// However, remember margin is needed, instead of transformed prediction
        /// e.g. for logistic regression: need to put in value before logistic transformation.
        /// </summary>
        public void SetBaseMargin(Float[] margin, uint nrow)
        {
            SetFloatInfo("base_margin", margin, nrow);
        }

        /// <summary>
        /// Set group size of DMatrix (used for ranking).
        /// </summary>
        public void SetGroups(IEnumerable<uint> group, uint nrow)
        {
            var agroup = group.ToArray();
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixSetGroup(_handle, agroup, nrow));
        }

        /// <summary>
        /// Set float type property into the DMatrix.
        /// </summary>
        /// <param name="field">The field name of the information</param>
        /// <param name="data">The array of data to be set</param>
        /// <param name="nrow">Number of rows</param>
        private void SetFloatInfo(string field, IEnumerable<Float> data, uint nrow)
        {
            Float[] cont = data.ToArray();
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixSetFloatInfo(_handle, field, cont, nrow));
        }
    }
}

