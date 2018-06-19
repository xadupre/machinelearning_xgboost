// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Runtime.Training;


namespace TestXGBoostWrapper
{
    public static class EnvHelper
    {
        /// <summary>
        /// Creates an environment for a unit test.
        /// </summary>
        public static TlcEnvironment NewTestEnvironment(int? seed = null, bool verbose = false,
                            MessageSensitivity sensitivity = (MessageSensitivity)(-1),
                            int conc = 0, TextWriter outWriter = null, TextWriter errWriter = null)
        {
            if (!seed.HasValue)
                seed = 42;
            if (outWriter == null)
                outWriter = new StreamWriter(new MemoryStream());
            if (errWriter == null)
                errWriter = new StreamWriter(new MemoryStream());
            return new TlcEnvironment(seed, verbose, sensitivity, conc, outWriter, errWriter);
        }

        /// <summary>
        /// Computes the prediction given a model as a zip file
        /// and some data in a view.
        /// </summary>
        public static void SavePredictions(TlcEnvironment env, string modelPath,
                                   string outFilePath, IDataView data)
        {
            using (var fs = File.OpenRead(modelPath))
            {
                var deserializedData = env.LoadTransforms(fs, data);
                var saver2 = env.CreateSaver("Text");
                var columns = new int[deserializedData.Schema.ColumnCount];
                for (int i = 0; i < columns.Length; ++i)
                    columns[i] = i;
                using (var fs2 = File.Create(outFilePath))
                    saver2.SaveData(fs2, deserializedData, columns);
            }
        }

        /// <summary>
        /// Computes the prediction given a model as a zip file
        /// and some data in a view.
        /// </summary>
        public static void SavePredictions(TlcEnvironment env, IDataTransform tr, string outFilePath)
        {
            var saver2 = env.CreateSaver("Text");
            var columns = new int[tr.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = i;
            using (var fs2 = File.Create(outFilePath))
                saver2.SaveData(fs2, tr, columns);
        }

        /// <summary>
        /// Saves a model in a zip file.
        /// </summary>
        public static void SaveModel(TlcEnvironment env, IDataTransform tr, string outModelFilePath)
        {
            using (var ch = env.Start("SaveModel"))
            using (var fs = File.Create(outModelFilePath))
            {
                var trainingExamples = env.CreateExamples(tr, null);
                TrainUtils.SaveModel(env, ch, fs, null, trainingExamples);
            }
        }

        /// <summary>
        /// Loads a model.
        /// </summary>
        public static IPredictor LoadModel(TlcEnvironment env, string inputModelFile)
        {
            IPredictor inputPredictor = null;
            using (var ch = env.Start("LoadModel"))
                TrainUtils.TryLoadPredictor(ch, env, inputModelFile, out inputPredictor);
            return inputPredictor;
        }

        public static TRes CreateTrainer<TRes>(IHostEnvironment env, string settings, params object[] extraArgs)
            where TRes : class
        {
            string loadName;
            return CreateInstance<TRes, SignatureTrainer>(env, settings, out loadName, extraArgs);
        }

        private static TRes CreateInstance<TRes, TSig>(IHostEnvironment env, string settings, out string loadName, params object[] extraArgs)
            where TRes : class
        {
            Contracts.AssertValue(env);
            env.AssertValue(settings, "settings");

            var sc = SubComponent.Parse<TRes, TSig>(settings);
            loadName = sc.Kind;
            return sc.CreateInstance(env, extraArgs);
        }
    }
}
