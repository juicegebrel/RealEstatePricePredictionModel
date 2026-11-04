using System;
using System.Collections.Generic;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Replaces missing (NaN) feature values with the per-feature MEDIAN of the training rows
    /// (median instead of mean because the areas are heavily skewed). Fit on TRAIN only.
    /// </summary>
    public class MedianImputer
    {
        private double[] medians;

        public void Fit(IList<double[]> rows)
        {
            if (rows.Count == 0) throw new ArgumentException("Cannot fit an imputer on zero rows.");
            int n = rows[0].Length;
            medians = new double[n];
            for (int j = 0; j < n; j++)
            {
                List<double> values = new List<double>();
                foreach (double[] row in rows) if (!double.IsNaN(row[j])) values.Add(row[j]);
                if (values.Count == 0) { medians[j] = 0.0; continue; }
                values.Sort();
                int mid = values.Count / 2;
                medians[j] = (values.Count % 2 == 1) ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
            }
        }

        public double[] Transform(double[] row)
        {
            if (medians == null) throw new InvalidOperationException("Imputer has not been fitted.");
            double[] result = new double[row.Length];
            for (int j = 0; j < row.Length; j++) result[j] = double.IsNaN(row[j]) ? medians[j] : row[j];
            return result;
        }
    }
}
