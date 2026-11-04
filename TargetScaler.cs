using System;
using System.Collections.Generic;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Maps price -> network target and back. Prices span PHP 300,000 to PHP 2.5 billion, so the network
    /// learns ln(price): Scale() takes the natural log and maps it linearly to [0.1, 0.9] (the sigmoid output
    /// neuron cannot reach 0 or 1 exactly, so the extremes are avoided). Unscale() inverts both steps and
    /// returns PHP.
    /// </summary>
    public class TargetScaler
    {
        public const double LowerBound = 0.1;
        public const double UpperBound = 0.9;

        private double minLog;
        private double maxLog;

        public void Fit(IEnumerable<double> prices)
        {
            minLog = double.MaxValue;
            maxLog = double.MinValue;
            foreach (double p in prices)
            {
                double l = Math.Log(p);
                if (l < minLog) minLog = l;
                if (l > maxLog) maxLog = l;
            }
            if (!(maxLog > minLog)) throw new ArgumentException("Prices must not all be identical.");
        }

        public double Scale(double price)
        {
            return LowerBound + (UpperBound - LowerBound) * (Math.Log(price) - minLog) / (maxLog - minLog);
        }

        public double Unscale(double scaled)
        {
            double l = minLog + (scaled - LowerBound) / (UpperBound - LowerBound) * (maxLog - minLog);
            return Math.Exp(l);
        }
    }
}
