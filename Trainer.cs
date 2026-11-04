using System;
using System.Diagnostics;
using Backprop;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Training loop around the custom Backprop.NeuralNet (input -> hidden -> output, sigmoid units).
    /// Uses online (per-sample) gradient descent: for each sample the net does a forward pass,
    /// computes the output delta, back-propagates it to the hidden layer and updates all weights/biases
    /// (all of that happens inside NeuralNet.learn()). Samples are reshuffled every epoch.
    /// </summary>
    public static class Trainer
    {
        public static TrainingResult Train(PreparedData data, TrainingConfig config,
                                           Action<EpochRecord> onEpoch, Func<bool> isCancelled)
        {
            NeuralNet.seedRandom(config.Seed);
            NeuralNet net = new NeuralNet(data.InputCount, config.HiddenNodes, 1);
            net.setLearningRates(config.LearningRate, config.LearningRate);

            TrainingResult result = new TrainingResult();
            result.Network = net;
            result.Config = config;

            int n = data.TrainX.Length;
            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            Random shuffler = new Random(config.Seed);

            Stopwatch clock = Stopwatch.StartNew();
            for (int epoch = 1; epoch <= config.Epochs; epoch++)
            {
                if (isCancelled != null && isCancelled()) { result.Cancelled = true; break; }

                DataPreparer.Shuffle(order, shuffler);
                double squaredErrorSum = 0.0;
                for (int k = 0; k < n; k++)
                {
                    int i = order[k];
                    LoadInputs(net, data.TrainX[i]);
                    net.setDesiredOutput(0, data.TrainY[i]);
                    net.learn();                                   // forward + backprop + weight update
                    double err = data.TrainY[i] - net.getOuputData(0); // output computed BEFORE the update
                    squaredErrorSum += err * err;
                }

                EpochRecord record = new EpochRecord();
                record.Epoch = epoch;
                record.TrainLoss = squaredErrorSum / n;
                record.ValLoss = MeanSquaredError(net, data.ValX, data.ValY);
                result.History.Add(record);

                if (double.IsNaN(record.TrainLoss) || double.IsInfinity(record.TrainLoss))
                {
                    result.Diverged = true;
                    break;
                }
                if (record.ValLoss < result.BestValLoss)
                {
                    result.BestValLoss = record.ValLoss;
                    result.BestValEpoch = epoch;
                }
                if (onEpoch != null) onEpoch(record);

                if (IsCheckpoint(config, epoch))
                {
                    Checkpoint cp = new Checkpoint();
                    cp.Epoch = epoch;
                    cp.Train = Evaluate(net, data.TrainX, data.TrainPrice, data.TargetScaler);
                    cp.Validation = Evaluate(net, data.ValX, data.ValPrice, data.TargetScaler);
                    result.Checkpoints.Add(cp);
                }
            }
            clock.Stop();
            result.Seconds = clock.Elapsed.TotalSeconds;

            if (!result.Diverged)
            {
                result.TrainMetrics = Evaluate(net, data.TrainX, data.TrainPrice, data.TargetScaler);
                result.ValMetrics = Evaluate(net, data.ValX, data.ValPrice, data.TargetScaler);
                if (config.EvaluateOnTest)
                {
                    result.TestPredictions = PredictPrices(net, data.TestX, data.TargetScaler);
                    result.TestMetrics = RegressionMetrics.Compute(data.TestPrice, result.TestPredictions);
                }
            }
            return result;
        }

        private static bool IsCheckpoint(TrainingConfig config, int epoch)
        {
            foreach (int e in config.CheckpointEpochs) if (e == epoch) return true;
            return false;
        }

        private static void LoadInputs(NeuralNet net, double[] x)
        {
            for (int j = 0; j < x.Length; j++) net.setInputs(j, x[j]);
        }

        /// <summary>Forward pass only; returns the scaled output (0.1..0.9 range).</summary>
        public static double PredictScaled(NeuralNet net, double[] scaledFeatures)
        {
            LoadInputs(net, scaledFeatures);
            net.run();
            return net.getOuputData(0);
        }

        public static double MeanSquaredError(NeuralNet net, double[][] x, double[] targets)
        {
            double sum = 0.0;
            for (int i = 0; i < x.Length; i++)
            {
                double err = targets[i] - PredictScaled(net, x[i]);
                sum += err * err;
            }
            return sum / x.Length;
        }

        public static double[] PredictPrices(NeuralNet net, double[][] x, TargetScaler targetScaler)
        {
            double[] prices = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
                prices[i] = targetScaler.Unscale(PredictScaled(net, x[i]));
            return prices;
        }

        public static RegressionMetrics Evaluate(NeuralNet net, double[][] x, double[] actualPrices, TargetScaler targetScaler)
        {
            return RegressionMetrics.Compute(actualPrices, PredictPrices(net, x, targetScaler));
        }
    }
}
