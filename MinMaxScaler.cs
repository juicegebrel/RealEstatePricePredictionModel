using System;
using System.Collections.Generic;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Feature-wise MinMax scaling to [0,1]:  x' = (x - min) / (max - min).
    /// Fit() must only see TRAINING rows so no information leaks from validation/test data.
    /// Constant columns (max == min) are mapped to 0.
    /// </summary>
    public class MinMaxScaler
    {
        private double[] min;
        private double[] max;

        public void Fit(IList<double[]> rows)
        {
            if (rows.Count == 0) throw new ArgumentException("Cannot fit a scaler on zero rows.");
            int n = rows[0].Length;
            min = new double[n];
            max = new double[n];
            for (int j = 0; j < n; j++) { min[j] = double.MaxValue; max[j] = double.MinValue; }

            foreach (double[] row in rows)
            {
                for (int j = 0; j < n; j++)
                {
                    if (row[j] < min[j]) min[j] = row[j];
                    if (row[j] > max[j]) max[j] = row[j];
                }
            }
        }

        public double[] Transform(double[] row)
        {
            if (min == null) throw new InvalidOperationException("Scaler has not been fitted.");
            double[] scaled = new double[row.Length];
            for (int j = 0; j < row.Length; j++)
            {
                double range = max[j] - min[j];
                scaled[j] = range > 0 ? (row[j] - min[j]) / range : 0.0;
            }
            return scaled;
        }
    }
}
