using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
	internal class Program
	{
		private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
		private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
		private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
		private static TextLoader _textLoader;

		private static void Main(string[] args)
		{
			var mlContext = new MLContext(seed: 0);

			_textLoader = mlContext.Data.CreateTextLoader(
		   columns: new TextLoader.Column[]
			{
				new TextLoader.Column("Label", DataKind.Bool,0),
				new TextLoader.Column("SentimentText", DataKind.Text,1)
			},
			separatorChar: '\t',
			hasHeader: true
			);
			ITransformer model;
			// Let's avoid regenerating the model every time if I can!! (this will get more complex when I move this to AWS!!!)
			if (File.Exists(_modelPath)) // Model has already been created, so just use it
			{
				Console.WriteLine($"Model appears to exist already: {_modelPath}");
				using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
				{
					model = mlContext.Model.Load(stream);
				}
				Console.WriteLine("Model loaded from file.");
			}
			else 
			{
			 	model = Train(mlContext, _trainDataPath);
			}
			
			#if DEBUG
			Evaluate(mlContext, model); // Don't REALLY need to evaluate the model every time if it was loaded from file, but may as well for now.
			#endif

			// TODO: Make this take JSON message coming in from AWS pipeline instead.
			var input = StringifyParams(args);
			var result = Predict(mlContext, model, input);
			var output = Newtonsoft.Json.JsonConvert.SerializeObject(result);
			Console.WriteLine(output); // Simulate returning a JSON object to be interpereted elsewhere
			
			#if DEBUG // in "production" we'd just want a nice clean JSON object written back to stdout for piping into something else probably (this will need reworking for use as lambda anyway probably)
			Console.WriteLine($"Sentiment: {(result.Prediction ? "Positive" : "Negative")} | Confidence: {result.Probability}");
			#endif
		}

		private static ITransformer Train(MLContext mlContext, string dataPath)
		{
			IDataView dataView = _textLoader.Read(dataPath);

			var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: "SentimentText", outputColumnName: "Features")
				.Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

			Console.WriteLine("=============== Create and Train the Model ===============");
			var model = pipeline.Fit(dataView);
			Console.WriteLine("=============== End of training ===============");
			Console.WriteLine();

			return model;
		}

		public static void Evaluate(MLContext mlContext, ITransformer model)
		{
			var dataView = _textLoader.Read(_testDataPath);
			Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
			var predictions = model.Transform(dataView);
			var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

			Console.WriteLine();
			Console.WriteLine("Model quality metrics evaluation");
			Console.WriteLine("--------------------------------");
			Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
			Console.WriteLine($"Auc: {metrics.Auc:P2}");
			Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
			Console.WriteLine("=============== End of model evaluation ===============");

			SaveModelAsFile(mlContext, model); // Save model to disk after evaluation (could be clever and not overwrite an existing model that had a better evaluation?)
		}

		private static SentimentPrediction Predict(MLContext context, ITransformer model, string input) 
		{
			var predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(context); // Wrap model in prediction engine
			var statement = new SentimentData { SentimentText = input};
			var result = predictionFunction.Predict(statement);
			return result;
		}
		private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
		{
			using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
				mlContext.Model.Save(model, fs);

			Console.WriteLine("Model saved to {0}", _modelPath);
		}

		/// If input was meant to be wrapped in quotes this would not be necessary
		private static string StringifyParams(string[] args)
		{
			return string.Join(" ", args);
		}
	}
}