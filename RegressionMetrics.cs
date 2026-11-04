using System;
using System.Collections.Generic;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Regression metrics computed on real prices (PHP). Because prices are extremely skewed (a few
    /// listings cost hundreds of millions), RMSE/MAE in pesos are dominated by those outliers, so
    /// scale-free "log" and percentage metrics are reported as well and used to choose hyperparameters.
    /// </summary>
    public class RegressionMetrics
    {
        public int Count;
        public double Mse;        // mean squared error (PHP^2)
        public double Rmse;       // root MSE (PHP)
        public double Mae;        // mean absolute error (PHP)
        public double R2;         // 1 - SSres/SStot on raw prices (0 = as good as the mean, <0 = worse)
        public double Rmsle;      // RMSE of ln(price): 0.5 means "typically off by a factor of e^0.5 = 1.65"
        public double R2Log;      // R-squared computed on ln(price)
        public double Mape;       // mean absolute percentage error (%)
        public double MedianApe;  // median absolute percentage error (%), robust to outliers
        public double WithinTwentyFive; // % of houses predicted within +-25% of the real price

        public static RegressionMetrics Compute(double[] actual, double[] predicted)
        {
            int n = actual.Length;
            double mean = 0, logMean = 0;
            for (int i = 0; i < n; i++) { mean += actual[i]; logMean += Math.Log(actual[i]); }
            mean /= n;
            logMean /= n;

            double sse = 0, sae = 0, sst = 0, ape = 0, sseLog = 0, sstLog = 0;
            int within = 0;
            List<double> apes = new List<double>(n);
            for (int i = 0; i < n; i++)
            {
                double err = predicted[i] - actual[i];
                sse += err * err;
                sae += Math.Abs(err);
                sst += (actual[i] - mean) * (actual[i] - mean);

                double logErr = Math.Log(Math.Max(predicted[i], 1.0)) - Math.Log(actual[i]);
                sseLog += logErr * logErr;
                double logDev = Math.Log(actual[i]) - logMean;
                sstLog += logDev * logDev;

                double pct = Math.Abs(err) / actual[i];
                ape += pct;
                apes.Add(pct);
                if (pct <= 0.25) within++;
            }
            apes.Sort();

            RegressionMetrics m = new RegressionMetrics();
            m.Count = n;
            m.Mse = sse / n;
            m.Rmse = Math.Sqrt(m.Mse);
            m.Mae = sae / n;
            m.R2 = sst > 0 ? 1.0 - sse / sst : 0.0;
            m.Rmsle = Math.Sqrt(sseLog / n);
            m.R2Log = sstLog > 0 ? 1.0 - sseLog / sstLog : 0.0;
            m.Mape = ape / n * 100.0;
            m.MedianApe = ((n % 2 == 1) ? apes[n / 2] : (apes[n / 2 - 1] + apes[n / 2]) / 2.0) * 100.0;
            m.WithinTwentyFive = within * 100.0 / n;
            return m;
        }

        /// <summary>Metrics of the naive model that always predicts the same constant price.</summary>
        public static RegressionMetrics Baseline(double[] actual, double constant)
        {
            double[] predicted = new double[actual.Length];
            for (int i = 0; i < predicted.Length; i++) predicted[i] = constant;
            return Compute(actual, predicted);
        }
    }
}
