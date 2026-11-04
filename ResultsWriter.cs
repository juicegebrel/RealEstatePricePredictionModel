using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace RealEstatePricePredictionModel
{
    /// <summary>Exports logs and results as CSV/text files into the Results folder.</summary>
    public static class ResultsWriter
    {
        public static void WriteTrainingLog(string path, List<EpochRecord> history)
        {
            CultureInfo c = CultureInfo.InvariantCulture;
            using (StreamWriter w = new StreamWriter(path, false, Encoding.UTF8))
            {
                w.WriteLine("Epoch,TrainLoss,ValidationLoss");
                foreach (EpochRecord r in history)
                    w.WriteLine(r.Epoch.ToString(c) + "," + r.TrainLoss.ToString("0.000000", c) + "," + r.ValLoss.ToString("0.000000", c));
            }
        }

        public static void WritePredictions(string path, double[] actual, double[] predicted)
        {
            CultureInfo c = CultureInfo.InvariantCulture;
            using (StreamWriter w = new StreamWriter(path, false, Encoding.UTF8))
            {
                w.WriteLine("ActualPrice,PredictedPrice,Error,PercentError");
                for (int i = 0; i < actual.Length; i++)
                {
                    double err = predicted[i] - actual[i];
                    w.WriteLine(actual[i].ToString("0", c) + "," + predicted[i].ToString("0", c) + ","
                              + err.ToString("0", c) + "," + (err / actual[i] * 100.0).ToString("0.00", c));
                }
            }
        }

        public static void WriteExperiments(string path, List<ExperimentRow> rows)
        {
            using (StreamWriter w = new StreamWriter(path, false, Encoding.UTF8))
            {
                w.WriteLine(ExperimentRow.CsvHeader);
                foreach (ExperimentRow r in rows) w.WriteLine(r.ToCsvLine());
            }
        }

        public static void WriteText(string path, string text)
        {
            File.WriteAllText(path, text, Encoding.UTF8);
        }
    }
}
