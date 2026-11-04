using System.Collections.Generic;
using Backprop;

namespace RealEstatePricePredictionModel
{
    /// <summary>Loss recorded after one epoch (MSE on the scaled targets).</summary>
    public class EpochRecord
    {
        public int Epoch;
        public double TrainLoss;   // running average of squared error while training through the epoch
        public double ValLoss;     // MSE on the validation set after the epoch
    }

    /// <summary>Metrics captured at a chosen epoch (used for the epoch-count experiment).</summary>
    public class Checkpoint
    {
        public int Epoch;
        public RegressionMetrics Train;
        public RegressionMetrics Validation;
    }

    public class TrainingResult
    {
        public NeuralNet Network;
        public TrainingConfig Config;
        public List<EpochRecord> History = new List<EpochRecord>();
        public List<Checkpoint> Checkpoints = new List<Checkpoint>();
        public RegressionMetrics TrainMetrics;
        public RegressionMetrics ValMetrics;
        public RegressionMetrics TestMetrics;      // only when Config.EvaluateOnTest
        public double[] TestPredictions;           // dollars, only when Config.EvaluateOnTest
        public double BestValLoss = double.MaxValue;
        public int BestValEpoch;
        public double Seconds;
        public bool Cancelled;
        public bool Diverged;                      // loss became NaN/Infinity
    }
}
