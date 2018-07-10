// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Scikit.ML.XGBoostWrapper
{
    /// <summary>
    /// Helpers to train a booster with given parameters.
    /// </summary>
    internal static class WrappedXGBoostTraining
    {
        /// <summary>
        /// Train and returns a booster.
        /// </summary>
        /// <param name="ch">IChannel</param>
        /// <param name="pch">IProgressChannel</param>
        /// <param name="numberOfTrees">Number of trained trees</param>
        /// <param name="parameters">Parameters see <see cref="XGBoostArguments"/></param>
        /// <param name="dtrain">Training set</param>
        /// <param name="numBoostRound">Number of trees to train</param>
        /// <param name="obj">Custom objective</param>
        /// <param name="maximize">Whether to maximize feval.</param>
        /// <param name="verboseEval">Requires at least one item in evals.
        ///     If "verbose_eval" is True then the evaluation metric on the validation set is
        ///     printed at each boosting stage.</param>
        /// <param name="xgbModel">For continuous training.</param>
        /// <param name="saveBinaryDMatrix">Save DMatrix in binary format (for debugging purpose).</param>
        public static Booster Train(IChannel ch, IProgressChannel pch, out int numberOfTrees,
            Dictionary<string, string> parameters, DMatrix dtrain, int numBoostRound = 10,
            Booster.FObjType obj = null, bool maximize = false,
            bool verboseEval = true, Booster xgbModel = null,
            string saveBinaryDMatrix = null)
        {
#if(!XGBOOST_RABIT)
            if (WrappedXGBoostInterface.RabitIsDistributed() == 1)
            {
                var pname = WrappedXGBoostInterface.RabitGetProcessorName();
                ch.Info("[WrappedXGBoostTraining.Train] start {0}:{1}", pname, WrappedXGBoostInterface.RabitGetRank());
            }
#endif

            if (!string.IsNullOrEmpty(saveBinaryDMatrix))
                dtrain.SaveBinary(saveBinaryDMatrix);

            Booster bst = new Booster(parameters, dtrain, xgbModel);
            int numParallelTree = 1;
            int nboost = 0;

            if (parameters != null && parameters.ContainsKey("num_parallel_tree"))
            {
                numParallelTree = Convert.ToInt32(parameters["num_parallel_tree"]);
                nboost /= numParallelTree;
            }
            if (parameters.ContainsKey("num_class"))
            {
                int numClass = Convert.ToInt32(parameters["num_class"]);
                nboost /= numClass;
            }

            var prediction = new VBuffer<Float>();
            var grad = new VBuffer<Float>();
            var hess = new VBuffer<Float>();
            var start = DateTime.Now;

#if(!XGBOOST_RABIT)
            int version = bst.LoadRabitCheckpoint();
            ch.Check(WrappedXGBoostInterface.RabitGetWorldSize() != 1 || version == 0);
#else
            int version = 0;
#endif
            int startIteration = version / 2;
            nboost += startIteration;
            int logten = 0;
            int temp = numBoostRound * 5;
            while (temp > 0)
            {
                logten += 1;
                temp /= 10;
            }
            temp = Math.Max(logten - 2, 0);
            logten = 1;
            while (temp-- > 0)
                logten *= 10;

            var metrics = new List<string>() { "Iteration", "Training Time" };
            var units = new List<string>() { "iterations", "seconds" };
            if (verboseEval)
            {
                metrics.Add("Training Error");
                metrics.Add(parameters["objective"]);
            }
            var header = new ProgressHeader(metrics.ToArray(), units.ToArray());

            int iter = 0;
            double trainTime = 0;
            double trainError = double.NaN;

            pch.SetHeader(header, e =>
            {
                e.SetProgress(0, iter, numBoostRound - startIteration);
                e.SetProgress(1, trainTime);
                if (verboseEval)
                    e.SetProgress(2, trainError);
            });
            for (iter = startIteration; iter < numBoostRound; ++iter)
            {
                if (version % 2 == 0)
                {
                    bst.Update(dtrain, iter, ref grad, ref hess, ref prediction, obj);
#if(!XGBOOST_RABIT)
                    bst.SaveRabitCheckpoint();
#endif
                    version += 1;
                }

#if(!XGBOOST_RABIT)
                ch.Check(WrappedXGBoostInterface.RabitGetWorldSize() == 1 ||
                            version == WrappedXGBoostInterface.RabitVersionNumber());
#endif
                nboost += 1;

                trainTime = (DateTime.Now - start).TotalMilliseconds;

                if (verboseEval)
                {
                    pch.Checkpoint(new double?[] { iter, trainTime, trainError });
                    if (iter == startIteration || iter == numBoostRound - 1 || iter % logten == 0 ||
                        (DateTime.Now - start) > TimeSpan.FromMinutes(2))
                    {
                        string strainError = bst.EvalSet(new[] { dtrain }, new[] { "Train" }, iter);
                        // Example: "[0]\tTrain-error:0.028612"
                        if (!string.IsNullOrEmpty(strainError) && strainError.Contains(":"))
                        {
                            double val;
                            if (double.TryParse(strainError.Split(':').Last(), out val))
                                trainError = val;
                        }
                    }
                }
                else
                {
                    pch.Checkpoint(new double?[] { iter, trainTime });
                }

                version += 1;
            }
            numberOfTrees = numBoostRound * numParallelTree;
            if (WrappedXGBoostInterface.RabitIsDistributed() == 1)
            {
                var pname = WrappedXGBoostInterface.RabitGetProcessorName();
                ch.Info("[WrappedXGBoostTraining.Train] end {0}:{1}", pname, WrappedXGBoostInterface.RabitGetRank());
            }
            return bst;
        }
    }
}