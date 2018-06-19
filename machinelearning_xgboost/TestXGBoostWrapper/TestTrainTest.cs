﻿// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.XGBoostWrappers;

namespace TestXGBoostWrapper
{
    [TestClass]
    public class TestTrainTest
    {
        [TestMethod]
        public void TestXGBoostMultiClassification()
        {
            var methodName = string.Format("{0}", System.Reflection.MethodBase.GetCurrentMethod().Name);
            var dataFilePath = FileHelper.GetTestFile("iris.txt");

            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Features:R4:1-4 header=+}",
                                          new MultiFileSource(dataFilePath));

            var roles = env.CreateExamples(loader, "Features", "Label");
            var trainer = EnvHelper.CreateTrainer<XGBoostMulticlassTrainer>(env, "exgbmc{iter=10}");
            IDataTransform pred = null;
            using (var ch = env.Start("Train"))
            {
                var model = TrainUtils.Train(env, ch, roles, trainer, "Train", null, 0);
                pred = ScoreUtils.GetScorer(model, roles, env, roles.Schema);
            }

            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            EnvHelper.SavePredictions(env, pred, outputDataFilePath);
            Assert.IsTrue(File.Exists(outputDataFilePath));

            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            EnvHelper.SaveModel(env, pred, outModelFilePath);
            Assert.IsTrue(File.Exists(outModelFilePath));

            var d1 = File.ReadAllText(outputDataFilePath);
            Assert.IsTrue(d1.Length > 0);
        }
    }
}