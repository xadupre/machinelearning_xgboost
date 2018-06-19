// See the LICENSE file in the project root for more information.

#define XGB_EXTENDED
using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.XGBoostWrappers
{
    /// <summary>
    /// This class holds buffers. They are allocated by Microsoft.ML and given to modified XGBoost.
    /// The official distribution of XGBoost relies on OMP to distribute computation
    /// over multiple threads. XGBoost is optimized to compute the prediction in a batch,
    /// not oneoff prediction. To go around this constraint, the code of XGBoost was modified
    /// to disable the distribution with OMP made by XGBoost and to move the allocation of 
    /// buffers XGBoost requires in Microsoft.ML (or Microsoft.ML explicitely asks XGBoost to allocate and deallocate). 
    /// That way, the number of new allocations is reduced
    /// each time Microsoft.ML calls XGBoost prediction.
    /// 
    /// To compute the XGBoost prediction, Microsoft.ML makes a first call to get the size of the buffers
    /// it must allocate and then makes a second call to compute the prediction.
    /// 
    /// See also method <cref="PredictOneOff">PredictOneOff</cref> which should remain the
    /// only method to use it.
    /// </summary>
    internal sealed class XGBoostTreeBuffer
    {
        private byte[] _xgboostEntries;
        private float[] _predBuffer;
        private uint[] _predCounter;
        private IntPtr _regTreeFVec;
        private int _regTreeFVecLength;

        public byte[] XGBoostEntries => _xgboostEntries;
        public float[] PredBuffer => _predBuffer;
        public uint[] PredCounter => _predCounter;
        public IntPtr RegTreeFVec => _regTreeFVec;

        public XGBoostTreeBuffer()
        {
            _xgboostEntries = null;
            _predBuffer = null;
            _predCounter = null;
            _regTreeFVec = (IntPtr)0;
            _regTreeFVecLength = 0;
        }

        /// <summary>
        /// Free buffers allocated by XGBoost.
        /// </summary>
        ~XGBoostTreeBuffer()
        {
#if (XGB_EXTENDED)
            if (_regTreeFVec != IntPtr.Zero)
                WrappedXGBoostInterface.XGBoosterPredictNoInsideCacheFree(_regTreeFVec);
#endif
            _regTreeFVec = IntPtr.Zero;
            _regTreeFVecLength = 0;
        }

        /// <summary>
        /// Check the buffer can hold the current input, resize it otherwise.
        /// </summary>
        /// <param name="numSparseFeatures">number of sparsed features (VBuffer.Count), can be different for every observation</param>
        /// <param name="numFeatures">number of features (VBuffer.Length), same for all observations</param>
        public void ResizeEntries(uint numSparseFeatures, int numFeatures)
        {
            uint xgboostEntriesSize = numSparseFeatures * (sizeof(float) + sizeof(uint));
            if (_xgboostEntries == null || _xgboostEntries.Length < xgboostEntriesSize ||
                xgboostEntriesSize > _xgboostEntries.Length * 2)
                _xgboostEntries = new byte[xgboostEntriesSize];

#if(XGB_EXTENDED)
            if (_regTreeFVec == IntPtr.Zero || _regTreeFVecLength < numFeatures || numFeatures > _regTreeFVecLength * 2)
            {
                if (_regTreeFVec != IntPtr.Zero)
                    WrappedXGBoostInterface.XGBoosterPredictNoInsideCacheFree(_regTreeFVec);
                WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterPredictNoInsideCacheAllocate(numFeatures, ref _regTreeFVec));
                _regTreeFVecLength = numFeatures;
            }
#endif
        }

        public void ResizeOutputs(uint length, uint lengthBuffer, ref VBuffer<Float> predictedValues)
        {
            Contracts.Assert(length > 0);
            Contracts.Assert(lengthBuffer >= 0);
            if (length > (ulong)predictedValues.Length)
                predictedValues = new VBuffer<Float>((int)length, new Float[length]);
            else
                predictedValues = new VBuffer<Float>((int)length, predictedValues.Values);

            if (_predBuffer == null || lengthBuffer > (ulong)_predBuffer.Length)
                _predBuffer = new float[lengthBuffer];
            if (_predCounter == null || lengthBuffer > (ulong)_predCounter.Length)
                _predCounter = new uint[lengthBuffer];
        }
    }
}
