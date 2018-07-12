﻿// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Tools;
using Scikit.ML.XGBoostWrapper;


namespace TestXGBoostWrapper
{
    [TestClass]
    public class TestEntryPoints
    {
        [TestMethod]
        public void TestCSGenerator()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var basePath = FileHelper.GetOutputFile("CSharpApiExt.cs", methodName);
            var cmd = $"? generator=cs{{csFilename={basePath} exclude=System.CodeDom.dll}}";
            using (var std = new StdCapture())
            {
                Maml.Main(new[] { cmd });
                Assert.IsTrue(std.StdOut.Length > 0);
                Assert.IsTrue(std.StdErr.Length == 0);
                Assert.IsFalse(std.StdOut.ToLower().Contains("usage"));
            }
            var text = File.ReadAllText(basePath);
            // TODO: this tests fails because when ML.net is used
            // as a nuget, buget binaries and custom binaries
            // are not in the same folder. The command looks into
            // its folder and fetches every DLL to look into exposed
            // learner and transforms. XGBoostWrapper is in another folder and does not
            // appear it.
            //Assert.IsTrue(text.ToLower().Contains("xgb"));
        }
    }
}
