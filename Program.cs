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

			var model = Train(mlContext, _trainDataPath);
			Evaluate(mlContext, model);

			Predict(mlContext, model);

			PredictWithModelLoadedFromFile(mlContext);

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

		private static void Predict(MLContext mlContext, ITransformer model)
		{
			var predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext); // Wrap model in prediction engine

			// Quick test, make this more useful later (paramterize)
			SentimentData sampleStatement = new SentimentData
			{
				SentimentText = "This is a very rude movie"
			};

			var resultprediction = predictionFunction.Predict(sampleStatement);

			Console.WriteLine();
			Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

			Console.WriteLine();
			Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {resultprediction.Probability} ");
			Console.WriteLine("=============== End of Predictions ===============");
			Console.WriteLine();

		}

		public static void PredictWithModelLoadedFromFile(MLContext mlContext)
		{
			IEnumerable<SentimentData> sentiments = new[]
{
			new SentimentData
			{
				SentimentText = "This is a very rude movie"
			},
			new SentimentData
			{
				SentimentText = "I love this article."
			}
			};

			ITransformer loadedModel;
			using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
			{
				loadedModel = mlContext.Model.Load(stream);
			}

			// Create prediction engine
			var sentimentStreamingDataView = mlContext.Data.ReadFromEnumerable(sentiments);
			var predictions = loadedModel.Transform(sentimentStreamingDataView);

			// Use the model to predict whether comment data is toxic (1) or nice (0).
			var predictedResults = mlContext.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

			Console.WriteLine();
			Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");

			var sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction)); // Zipper together original sentiments/comments and predicted "toxicity"

			foreach (var item in sentimentsAndPredictions)
			{
				Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {item.prediction.Probability} ");
			}
			Console.WriteLine("=============== End of predictions ===============");

		}

		private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
		{
			using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
				mlContext.Model.Save(model, fs);

			Console.WriteLine("Model saved to {0}", _modelPath);
		}
	}
}