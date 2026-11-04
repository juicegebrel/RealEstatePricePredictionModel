using System.Globalization;

namespace RealEstatePricePredictionModel
{
    /// <summary>One line of the hyperparameter experiment table (metrics averaged over several seeds).</summary>
    public class ExperimentRow
    {
        public int HiddenNodes;
        public double LearningRate;
        public int Epochs;
        public double TrainRmsle;
        public double ValRmsle;     // selection criterion (lower is better)
        public double ValR2Log;
        public double ValMedianApe;
        public int RunsAveraged;
        public bool Diverged;       // every seed diverged -> no valid metrics

        public static string CsvHeader
        {
            get { return "HiddenNodes,LearningRate,Epochs,TrainRMSLE,ValRMSLE,ValR2Log,ValMedianAPE,RunsAveraged,Diverged"; }
        }

        public string ToCsvLine()
        {
            CultureInfo c = CultureInfo.InvariantCulture;
            return HiddenNodes.ToString(c) + "," + LearningRate.ToString("0.###", c) + "," + Epochs.ToString(c) + ","
                 + TrainRmsle.ToString("0.0000", c) + "," + ValRmsle.ToString("0.0000", c) + "," + ValR2Log.ToString("0.0000", c) + ","
                 + ValMedianApe.ToString("0.00", c) + "," + RunsAveraged.ToString(c) + "," + (Diverged ? "yes" : "no");
        }
    }
}
