using System;
using System.Collections.Generic;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Shuffles the records (fixed seed), splits them into train / validation / test and applies
    /// imputation + scaling. All preprocessing statistics (medians, min/max) come from the TRAIN split only.
    /// Train      = fit the weights.
    /// Validation = choose hyperparameters (tuning never touches the test set).
    /// Test       = one final, unbiased estimate of performance.
    /// </summary>
    public static class DataPreparer
    {
        public static PreparedData Prepare(List<HouseRecord> records, double trainFraction, double validationFraction, int splitSeed)
        {
            int n = records.Count;
            if (n < 20) throw new ArgumentException("Need at least 20 valid rows to split the data.");

            int trainCount = (int)Math.Round(n * trainFraction);
            int valCount = (int)Math.Round(n * validationFraction);
            int testCount = n - trainCount - valCount;
            if (trainCount < 1 || valCount < 1 || testCount < 1)
                throw new ArgumentException("Split fractions leave an empty subset.");

            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            Shuffle(order, new Random(splitSeed));

            List<double[]> trainRaw = new List<double[]>();
            List<double[]> valRaw = new List<double[]>();
            List<double[]> testRaw = new List<double[]>();
            double[] trainPrice = new double[trainCount];
            double[] valPrice = new double[valCount];
            double[] testPrice = new double[testCount];

            for (int k = 0; k < n; k++)
            {
                HouseRecord r = records[order[k]];
                double[] features = FeatureEncoder.Encode(r);   // may contain NaN
                if (k < trainCount) { trainRaw.Add(features); trainPrice[k] = r.Price; }
                else if (k < trainCount + valCount) { valRaw.Add(features); valPrice[k - trainCount] = r.Price; }
                else { testRaw.Add(features); testPrice[k - trainCount - valCount] = r.Price; }
            }

            PreparedData d = new PreparedData();
            d.Imputer = new MedianImputer();
            d.Imputer.Fit(trainRaw);

            List<double[]> trainImputed = Impute(d.Imputer, trainRaw);
            d.FeatureScaler = new MinMaxScaler();
            d.FeatureScaler.Fit(trainImputed);
            d.TargetScaler = new TargetScaler();
            d.TargetScaler.Fit(trainPrice);

            d.TrainX = ScaleAll(d.FeatureScaler, trainImputed);
            d.ValX = ScaleAll(d.FeatureScaler, Impute(d.Imputer, valRaw));
            d.TestX = ScaleAll(d.FeatureScaler, Impute(d.Imputer, testRaw));
            d.TrainPrice = trainPrice;
            d.ValPrice = valPrice;
            d.TestPrice = testPrice;
            d.TrainY = ScaleTargets(d.TargetScaler, trainPrice);
            d.ValY = ScaleTargets(d.TargetScaler, valPrice);
            d.TestY = ScaleTargets(d.TargetScaler, testPrice);

            double logSum = 0;
            foreach (double p in trainPrice) logSum += Math.Log(p);
            d.BaselinePrice = Math.Exp(logSum / trainPrice.Length);
            return d;
        }

        private static List<double[]> Impute(MedianImputer imputer, List<double[]> rows)
        {
            List<double[]> result = new List<double[]>(rows.Count);
            foreach (double[] row in rows) result.Add(imputer.Transform(row));
            return result;
        }

        private static double[][] ScaleAll(MinMaxScaler scaler, List<double[]> rows)
        {
            double[][] result = new double[rows.Count][];
            for (int i = 0; i < rows.Count; i++) result[i] = scaler.Transform(rows[i]);
            return result;
        }

        private static double[] ScaleTargets(TargetScaler scaler, double[] prices)
        {
            double[] result = new double[prices.Length];
            for (int i = 0; i < prices.Length; i++) result[i] = scaler.Scale(prices[i]);
            return result;
        }

        /// <summary>Fisher-Yates shuffle.</summary>
        public static void Shuffle(int[] array, Random rng)
        {
            for (int i = array.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                int tmp = array[i]; array[i] = array[j]; array[j] = tmp;
            }
        }
    }
}
