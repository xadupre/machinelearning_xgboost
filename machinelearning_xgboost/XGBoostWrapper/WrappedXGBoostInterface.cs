// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.XGBoostWrappers
{
    /// <summary>
    /// The DLL libxgboost.dll can be obtained by compiling: https://github.com/xadupre/xgboost.
    /// It is a modified version of XGBoost which allows one-off predictions (not in a batch).
    /// </summary>
    public static class WrappedXGBoostInterface
    {
        #region helpers

        /// <summary>
        /// Checks if XGBoost has a pending error message. Raises an exception in that case.
        /// </summary>
        /// The class is public (and not internal) to allow unit testing. 
        public static void Check(int res)
        {
            if (res != 0)
            {
                string mes = XGBGetLastError();
                throw Contracts.Except("XGBoost Error, code is {0}, error message is '{1}'.", res, mes);
            }
        }

        /// <summary>
        /// Implements an unsafe memcopy.
        /// </summary>
        public static void Copy(IntPtr src, int srcIndex,
                float[] dst, int count)
        {
            Marshal.Copy(src, dst, srcIndex, count);
        }

        /// <summary>
        /// Implements an unsafe memcopy.
        /// </summary>
        public static string CastString(IntPtr src)
        {
            return Marshal.PtrToStringAnsi(src);
        }

        #endregion

        // REVIEW xadupre: this will not work under linux.

#if(!WIN32)
        private const string DllName = "xgboost.dll";
#else
        private const string DllName = "libxgboost.so";
#endif

        #region API ERROR

        [DllImport(DllName)]
        public static extern string XGBGetLastError();

        #endregion

        #region API DMatrix

        [DllImport(DllName, EntryPoint = "XGDMatrixCreateFromMat", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixCreateFromMat(float[] data, /*bst_ulong*/ uint nrow, /*bst_ulong*/ uint ncol, float missing, ref IntPtr res);

        [DllImport(DllName, EntryPoint = "XGDMatrixCreateFromCSREx", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixCreateFromCSREx(/*size_t* */ ulong[] indptr, uint[] indices, float[] data,
            /*size_t*/ ulong nindptr, /*size_t*/ ulong nelem, /*ulong*/ ulong num_col, ref IntPtr res);

        [DllImport(DllName, EntryPoint = "XGDMatrixFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixFree(IntPtr handle);

        [DllImport(DllName, EntryPoint = "XGDMatrixSetFloatInfo", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixSetFloatInfo(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)]string field, float[] array, /*bst_ulong*/ uint len);

        [DllImport(DllName, EntryPoint = "XGDMatrixSetGroup", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixSetGroup(IntPtr handle, uint[] groups, /*bst_ulong*/ uint length);

        [DllImport(DllName, EntryPoint = "XGDMatrixNumRow", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixNumRow(IntPtr handle, ref /*bst_ulong*/ uint res);

        [DllImport(DllName, EntryPoint = "XGDMatrixNumCol", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixNumCol(IntPtr handle, ref /*bst_ulong*/ uint res);

        [DllImport(DllName, EntryPoint = "XGDMatrixSaveBinary", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixSaveBinary(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)]string fname, int silent);

        [DllImport(DllName, EntryPoint = "XGDMatrixCreateFromFile", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixCreateFromFile([MarshalAs(UnmanagedType.LPStr)]string fname, int silent, out IntPtr handle);

        #endregion

        #region API Booster

        [DllImport(DllName, EntryPoint = "XGBoosterCreate", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterCreate(IntPtr[] handles, /*bst_ulong*/ uint len, ref IntPtr res);

        [DllImport(DllName, EntryPoint = "XGBoosterFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterFree(IntPtr handle);

        [DllImport(DllName, EntryPoint = "XGBoosterSetParam", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterSetParam(IntPtr handle,
                                                   [MarshalAs(UnmanagedType.LPStr)]string name,
                                                   [MarshalAs(UnmanagedType.LPStr)]string value);

        [DllImport(DllName, EntryPoint = "XGBoosterLoadModelFromBuffer", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterLoadModelFromBuffer(IntPtr handle, byte* buf, /*bst_ulong*/ uint len);

        [DllImport(DllName, EntryPoint = "XGBoosterGetModelRaw", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterGetModelRaw(IntPtr handle, ref /*bst_ulong*/ uint outLen, out byte* outDptr);


        [DllImport(DllName, EntryPoint = "XGBoosterGetNumInfoTest", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterGetNumInfoTest(IntPtr handle, IntPtr res, [MarshalAs(UnmanagedType.LPStr)]string nameStr);

        public static double XGBoosterGetNumInfo(IntPtr handle, string nameStr)
        {
            double[] info = new double[1];
            unsafe
            {
                fixed (double* pd = info)
                {
                    IntPtr ptr = (IntPtr)pd;
                    WrappedXGBoostInterface.XGBoosterGetNumInfoTest(handle, ptr, "NumTrees");
                }
            }
            return info[0];
        }

        #endregion

        #region API train

        [DllImport(DllName, EntryPoint = "XGBoosterUpdateOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterUpdateOneIter(IntPtr handle, int iter, IntPtr dtrain);

        [DllImport(DllName, EntryPoint = "XGBoosterBoostOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterBoostOneIter(IntPtr handle, IntPtr dtrain, float[] grad, float[] hess, /*bst_ulong*/ uint len);

        /// outResult is a char** pointer, ANSI encoding.
        [DllImport(DllName, EntryPoint = "XGBoosterEvalOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterEvalOneIter(IntPtr handle, int iter, IntPtr[] dmats,
                                 [In][MarshalAsAttribute(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] evnames,
            /*bst_ulong*/ uint len, out IntPtr outResult);

        #endregion

        #region API Predict

        /// This function returns a pointer on data XGGboost owns. It cannot be freed.
        [DllImport(DllName, EntryPoint = "XGBoosterPredict", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredict(IntPtr handle, IntPtr dmat,
                                                  int optionMask, uint ntreeLimit, ref /*bst_ulong*/ uint outLen,
                                                  ref /*float* */ IntPtr outResult);

        #endregion

        #region Custom API Predict

        [DllImport(DllName, EntryPoint = "XGBoosterPredictNoInsideCache", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredictNoInsideCache(IntPtr handle, /* SparseEntry **/ IntPtr entries,
                                                    /*bst_ulong*/ uint nbEntries, int optionMask, uint ntreeLimit,
                                                    /*bst_ulong*/ uint outLen, /*bst_ulong*/ uint outLenBuffer,
                                                    float* outResult, float* predBuffer, uint* predCounter
#if (XGB_EXTENDED)
                                                    , /*RegVec::FVec*/ IntPtr regVecFVec
#endif
            );

        [DllImport(DllName, EntryPoint = "XGBoosterPredictOutputSize", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredictOutputSize(IntPtr handle, /* SparseEntry **/ IntPtr entries,
            /*bst_ulong*/ uint nbEntries, int optionMask, uint ntreeLimit, ref /*bst_ulong*/ uint outLen, ref /*bst_ulong*/ uint outLenBuffer);


        [DllImport(DllName, EntryPoint = "XGBoosterLazyInit", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterLazyInit(IntPtr handle);

        [DllImport(DllName, EntryPoint = "XGBoosterCopyEntries", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterCopyEntries(IntPtr entries, ref /*bst_ulong*/ uint nbEntries, float* values, int* indices, float missing);

#if (XGB_EXTENDED)
        [DllImport(DllName, EntryPoint = "XGBoosterPredictNoInsideCacheAllocate", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredictNoInsideCacheAllocate(int nb_features,
                                    /*RegTree::FVec* */ ref IntPtr regtreefvec);

        [DllImport(DllName, EntryPoint = "XGBoosterPredictNoInsideCacheFree", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredictNoInsideCacheFree(/*RegTree::FVec* */ IntPtr regtreefvec);
#endif

        #endregion

#if (!XGBOOST_RABIT)
        #region rabbit

        [DllImport(DllName, EntryPoint = "XGBoosterLoadRabitCheckpoint", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterLoadRabitCheckpoint(IntPtr handleBooster, ref int version);

        [DllImport(DllName, EntryPoint = "XGBoosterSaveRabitCheckpoint", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterSaveRabitCheckpoint(IntPtr handleBooster);

        [DllImport(DllName, EntryPoint = "RabitInit", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void RabitInit(int argc, string[] argv);

        [DllImport(DllName, EntryPoint = "RabitFinalize", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void RabitFinalize();

        [DllImport(DllName, EntryPoint = "RabitGetRank", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int RabitGetRank();

        [DllImport(DllName, EntryPoint = "RabitGetWorldSize", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int RabitGetWorldSize();

        [DllImport(DllName, EntryPoint = "RabitIsDistributed", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int RabitIsDistributed();

        [DllImport(DllName, EntryPoint = "RabitTrackerPrint", CallingConvention = CallingConvention.StdCall)]
        private unsafe static extern void RabitGetProcessorName(char* out_name, ref uint out_len, uint max_len);

        public static string RabitGetProcessorName()
        {
            string bname = "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
            uint len = 0;
            unsafe
            {
                fixed (char* name = bname)
                {
                    RabitGetProcessorName(name, ref len, (uint)bname.Length);
                }
            }
            return bname;
        }

        [DllImport(DllName, EntryPoint = "RabitVersionNumber", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int RabitVersionNumber();

        #endregion

        #region Rabit Static Instance

        public sealed class RabitStaticInstance
        {
            readonly static RabitStaticInstance _instance = new RabitStaticInstance();

            private RabitStaticInstance()
            {
                //LibInit();
            }

            ~RabitStaticInstance()
            {
                //LibFinalize();
            }

            void LibInit()
            {
                RabitInit(0, null);
                ;
            }

            void LibFinalize()
            {
                RabitFinalize();
            }
        }

        #endregion
#endif
    }
}