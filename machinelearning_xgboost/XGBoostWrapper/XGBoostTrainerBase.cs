// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.XGBoostWrappers
{
    /// <summary>
    /// Base class for all training with XGBoost (binary classification, regression, ...).
    /// The main change is the objective which defines which types of problem is being
    /// optimized. Check XGBoost documentation about the parameters to know more.
    /// </summary>
    // REVIEW xadupre: implement IIncrementalTrainer to do continuous training.
    public abstract class XGBoostTrainerBase<TOutput, TPredictor> : ITrainer<RoleMappedData, TPredictor>, IIncrementalTrainer<RoleMappedData, TPredictor>
        where TPredictor : XGBoostPredictorBase<TOutput>
    {
        #region members

        private readonly XGBoostArguments _args;

        /// <summary>
        /// The predication kind.
        /// </summary>
        private readonly PredictionKind _predictionKind;

        /// <summary>
        /// The trained model saved as binary.
        /// </summary>
        protected byte[] _model;
        protected int _nbFeaturesXGboost;
        protected int _nbFeaturesML;

        protected readonly IHost _host;

        #endregion

        #region non abstract members

        protected XGBoostTrainerBase(IHostEnvironment env, XGBoostArguments args, PredictionKind predictionKind, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));

            _host = env.Register(name);
            _host.CheckValue(args, nameof(args));

            _args = args;
            _predictionKind = predictionKind;
        }

        public void Train(RoleMappedData data)
        {
            using (var ch = _host.Start("Training with XGBoost"))
            {
                using (var pch = _host.StartProgressChannel("Training with XGBoost"))
                    TrainCore(ch, pch, data, null);
                ch.Done();
            }
        }

        public void Train(RoleMappedData data, TPredictor predictor)
        {
            using (var ch = _host.Start("Continuous Training with XGBoost"))
            {
                using (var pch = _host.StartProgressChannel("Continuous Training with XGBoost"))
                    TrainCore(ch, pch, data, predictor);
                ch.Done();
            }
        }

        protected virtual void ValidateTrainInput(IChannel ch, RoleMappedData data)
        {
            _host.CheckParam(data.Schema.Label != null, nameof(data), "Need a label column");
            var labelType = data.Schema.Label.Type;
            if (!(labelType.RawKind == DataKind.R4 || labelType.RawKind == DataKind.U1 || labelType.RawKind == DataKind.U2 ||
                labelType.RawKind == DataKind.U4 || labelType.IsKey || labelType.RawKind == DataKind.BL))
            {
                throw _host.ExceptParam(nameof(data), $"Label column 'Label' is of type '{labelType}', but must be R4, a key or a boolean.");
            }
        }

        private void TrainCore(IChannel ch, IProgressChannel pch, RoleMappedData data, TPredictor predictor)
        {
            // Verifications.
            _host.AssertValue(ch);
            ch.CheckValue(data, nameof(data));

            ValidateTrainInput(ch, data);

            var featureColumns = data.Schema.GetColumns(RoleMappedSchema.ColumnRole.Feature);
            ch.Check(featureColumns.Count == 1, "Only one vector of features is allowed.");

            // Data dimension.
            int fi = data.Schema.Feature.Index;
            var colType = data.Schema.Schema.GetColumnType(fi);
            ch.Assert(colType.IsVector, "Feature must be a vector.");
            ch.Assert(colType.VectorSize > 0, "Feature dimension must be known.");
            int nbDim = colType.VectorSize;
            IDataView view = data.Data;
            long nbRows = DataViewUtils.ComputeRowCount(view);

            Float[] labels;
            uint[] groupCount;
            DMatrix dtrain;
            // REVIEW xadupre: this can be avoided by using method XGDMatrixCreateFromDataIter from the XGBoost API.
            // XGBoost removes NaN values from a dense matrix and stores it in sparse format anyway.
            bool isDense = DetectDensity(data);
            var dt = DateTime.Now;

            if (isDense)
            {
                dtrain = FillDenseMatrix(ch, nbDim, nbRows, data, out labels, out groupCount);
                ch.Info("Dense matrix created with nbFeatures={0} and nbRows={1} in {2}.", nbDim, nbRows, DateTime.Now - dt);
            }
            else
            {
                dtrain = FillSparseMatrix(ch, nbDim, nbRows, data, out labels, out groupCount);
                ch.Info("Sparse matrix created with nbFeatures={0} and nbRows={1} in {2}.", nbDim, nbRows, DateTime.Now - dt);
            }

            // Some options are filled based on the data.
            var options = _args.ToDict(_host);
            UpdateXGBoostOptions(ch, options, labels, groupCount);

            // For multi class, the number of labels is required.
            ch.Assert(PredictionKind != PredictionKind.MultiClassClassification || options.ContainsKey("num_class"),
                    "XGBoost requires the number of classes to be specified in the parameters.");

            ch.Info("XGBoost objective={0}", options["objective"]);

            int numTrees;
            Booster res = WrappedXGBoostTraining.Train(ch, pch, out numTrees, options, dtrain,
                numBoostRound: _args.numBoostRound,
                obj: null, verboseEval: _args.verboseEval,
                xgbModel: predictor == null ? null : predictor.GetBooster(),
                saveBinaryDMatrix: _args.saveXGBoostDMatrixAsBinary);

            int nbTrees = res.GetNumTrees();
            ch.Info("Training is complete. Number of added trees={0}, total={1}.", numTrees, nbTrees);

            _model = res.SaveRaw();
            _nbFeaturesXGboost = (int)dtrain.GetNumCols();
            _nbFeaturesML = nbDim;
        }

        /// <summary>
        /// Checks the first nbRows rows. If a vector is not dense, the method returns false,
        /// true otherwise.
        /// </summary>
        private bool DetectDensity(RoleMappedData data, int nbRows = 10)
        {
            using (var cursor = new FloatLabelCursor(data, CursOpt.Features))
            {
                while (cursor.MoveNext())
                {
                    if (!cursor.Features.IsDense)
                        return false;
                    if (nbRows < 0)
                        break;
                }
            }
            return true;
        }

        /// <summary>
        /// Some tasks (multi classification) require a processing of the labels after reading the data from a data view
        /// and before creating the XGBoost container for the same data. This method is overloaded by the classes
        /// which need a specific processing.
        /// </summary>
        protected virtual void PostProcessLabelsBeforeCreatingXGBoostContainer(IChannel ch, RoleMappedData data, Float[] labels)
        {
        }

        private DMatrix FillDenseMatrix(IChannel ch, int nbDim, long nbRows,
                        RoleMappedData data, out Float[] labels, out uint[] groupCount)
        {
            // Allocation.
            string errorMessageGroup = string.Format("Group is above {0}.", uint.MaxValue);
            if (nbDim * nbRows >= Utils.ArrayMaxSize)
            {
                throw _host.Except("The training dataset is too big to hold in memory. " +
                    "Number of features ({0}) multiplied by the number of rows ({1}) must be less than {2}.", nbDim, nbRows, Utils.ArrayMaxSize);
            }
            var features = new Float[nbDim * nbRows];
            labels = new Float[nbRows];
            var hasWeights = data.Schema.Weight != null;
            var hasGroup = data.Schema.Group != null;
            var weights = hasWeights ? new Float[nbRows] : null;
            var groupsML = hasGroup ? new uint[nbRows] : null;
            groupCount = hasGroup ? new uint[nbRows] : null;
            var groupId = hasGroup ? new HashSet<uint>() : null;

            int count = 0;
            int lastGroup = -1;
            int fcount = 0;
            var flags = CursOpt.Features | CursOpt.Label | CursOpt.AllowBadEverything | CursOpt.Weight | CursOpt.Group;

            var featureVector = default(VBuffer<float>);
            var labelProxy = float.NaN;
            var groupProxy = ulong.MaxValue;
            using (var cursor = data.CreateRowCursor(flags, null))
            {
                var featureGetter = cursor.GetFeatureFloatVectorGetter(data);
                var labelGetter = cursor.GetLabelFloatGetter(data);
                var weighGetter = cursor.GetOptWeightFloatGetter(data);
                var groupGetter = cursor.GetOptGroupGetter(data);

                while (cursor.MoveNext())
                {
                    featureGetter(ref featureVector);
                    labelGetter(ref labelProxy);

                    labels[count] = labelProxy;
                    if (Single.IsNaN(labels[count]))
                        continue;

                    featureVector.CopyTo(features, fcount, Single.NaN);
                    fcount += featureVector.Count;

                    if (hasWeights)
                        weighGetter(ref weights[count]);
                    if (hasGroup)
                    {
                        groupGetter(ref groupProxy);
                        _host.Check(groupProxy < uint.MaxValue, errorMessageGroup);
                        groupsML[count] = (uint)groupProxy;
                        if (count == 0 || groupsML[count - 1] != groupsML[count])
                        {
                            groupCount[++lastGroup] = 1;
                            ch.Check(!groupId.Contains(groupsML[count]), "Group Id are not contiguous.");
                            groupId.Add(groupsML[count]);
                        }
                        else
                            ++groupCount[lastGroup];
                    }
                    ++count;
                }
            }

            PostProcessLabelsBeforeCreatingXGBoostContainer(ch, data, labels);

            // We create a DMatrix.
            DMatrix dtrain = new DMatrix(features, (uint)count, (uint)nbDim, labels: labels, weights: weights, groups: groupCount);
            return dtrain;
        }

        /// <summary>
        /// Fill a sparse DMatrix using CSR compression.
        /// See http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html.
        /// </summary>
        private DMatrix FillSparseMatrix(IChannel ch, int nbDim, long nbRows, RoleMappedData data,
                                                 out Float[] labels, out uint[] groupCount)
        {
            // Allocation.
            if ((2 * nbRows) >= Utils.ArrayMaxSize)
            {
                throw _host.Except("The training dataset is too big to hold in memory. " +
                "2 features multiplied by the number of rows must be less than {0}.", Utils.ArrayMaxSize);
            }

            var features = new Float[nbRows * 2];
            var indices = new uint[features.Length];
            var indptr = new ulong[nbRows + 1];
            long nelem = 0;

            labels = new Float[nbRows];
            var hasWeights = data.Schema.Weight != null;
            var hasGroup = data.Schema.Group != null;
            var weights = hasWeights ? new Float[nbRows] : null;
            var groupsML = hasGroup ? new uint[nbRows] : null;
            groupCount = hasGroup ? new uint[nbRows] : null;
            var groupId = hasGroup ? new HashSet<uint>() : null;

            int count = 0;
            int lastGroup = -1;
            var flags = CursOpt.Features | CursOpt.Label | CursOpt.AllowBadEverything | CursOpt.Weight | CursOpt.Group;

            var featureVector = default(VBuffer<float>);
            var labelProxy = float.NaN;
            var groupProxy = ulong.MaxValue;
            using (var cursor = data.CreateRowCursor(flags, null))
            {
                var featureGetter = cursor.GetFeatureFloatVectorGetter(data);
                var labelGetter = cursor.GetLabelFloatGetter(data);
                var weighGetter = cursor.GetOptWeightFloatGetter(data);
                var groupGetter = cursor.GetOptGroupGetter(data);
                while (cursor.MoveNext())
                {
                    featureGetter(ref featureVector);
                    labelGetter(ref labelProxy);
                    labels[count] = labelProxy;
                    if (Single.IsNaN(labels[count]))
                        continue;

                    indptr[count] = (ulong)nelem;
                    int nbValues = featureVector.Count;
                    if (nbValues > 0)
                    {
                        if (nelem + nbValues > features.Length)
                        {
                            long newSize = Math.Max(nelem + nbValues, features.Length * 2);
                            if (newSize >= Utils.ArrayMaxSize)
                            {
                                throw _host.Except("The training dataset is too big to hold in memory. " +
                                    "It should be half of {0}.", Utils.ArrayMaxSize);
                            }
                            Array.Resize(ref features, (int)newSize);
                            Array.Resize(ref indices, (int)newSize);
                        }

                        Array.Copy(featureVector.Values, 0, features, nelem, nbValues);
                        if (featureVector.IsDense)
                        {
                            for (int i = 0; i < nbValues; ++i)
                                indices[nelem++] = (uint)i;
                        }
                        else
                        {
                            for (int i = 0; i < nbValues; ++i)
                                indices[nelem++] = (uint)featureVector.Indices[i];
                        }
                    }

                    if (hasWeights)
                        weighGetter(ref weights[count]);
                    if (hasGroup)
                    {
                        groupGetter(ref groupProxy);
                        if (groupProxy >= uint.MaxValue)
                            throw _host.Except($"Group is above {uint.MaxValue}");
                        groupsML[count] = (uint)groupProxy;
                        if (count == 0 || groupsML[count - 1] != groupsML[count])
                        {
                            groupCount[++lastGroup] = 1;
                            ch.Check(!groupId.Contains(groupsML[count]), "Group Id are not contiguous.");
                            groupId.Add(groupsML[count]);
                        }
                        else
                            ++groupCount[lastGroup];
                    }
                    ++count;
                }
            }
            indptr[count] = (uint)nelem;

            if (nelem < features.Length * 3 / 4)
            {
                Array.Resize(ref features, (int)nelem);
                Array.Resize(ref indices, (int)nelem);
            }

            PostProcessLabelsBeforeCreatingXGBoostContainer(ch, data, labels);

            // We create a DMatrix.
            DMatrix dtrain = new DMatrix((uint)nbDim, indptr, indices, features, (uint)count, (uint)nelem, labels: labels, weights: weights, groups: groupCount);
            return dtrain;
        }

        public PredictionKind PredictionKind
        {
            get { return _predictionKind; }
        }

        public void Train(object data)
        {
            Train((RoleMappedData)data);
        }

        IPredictor ITrainer.CreatePredictor()
        {
            return CreatePredictor();
        }

        #endregion

        #region abstract methods

        public abstract TPredictor CreatePredictor();

        /// <summary>
        /// If not present, this function adds parameters needed by XGBoost.
        /// The first one is the objective. It determines the training task and
        /// the error function XGBoost must use.
        /// For the multi-class problem, the number of class is required.
        /// </summary>
        /// <param name="options">XGBoost options (as strings as required by XGBoost API)</param>
        /// <param name="labels">array of labels (mandatory)</param>
        /// <param name="groups">array of groups (can be null)</param>
        /// <param name="ch">to inform the user</param>
        protected abstract void UpdateXGBoostOptions(IChannel ch, Dictionary<string, string> options, Float[] labels, uint[] groups);

        #endregion
    }
}
