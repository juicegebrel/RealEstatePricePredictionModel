using System;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Turns a HouseRecord into the (still unscaled) numeric input vector of the network.
    /// Areas are log-transformed because they are extremely right-skewed (22 m2 .. 25,000 m2) and price
    /// grows roughly proportionally to them. Missing values stay NaN here (imputed later) and get an
    /// extra 0/1 "was missing" flag, because e.g. a missing floor area usually means "lot only".
    /// Location information (coordinates, city) is deliberately NOT used.
    /// </summary>
    public static class FeatureEncoder
    {
        public static readonly string[] FeatureNames = new string[]
        {
            "ln(Floor Area)", "ln(Land Area)", "Bedrooms", "Bathrooms",
            "FloorArea missing", "LandArea missing", "Bedrooms missing", "Bathrooms missing"
        };

        public static int FeatureCount { get { return FeatureNames.Length; } }

        public static double[] Encode(HouseRecord r)
        {
            double[] f = new double[FeatureCount];
            f[0] = double.IsNaN(r.FloorArea) ? double.NaN : Math.Log(r.FloorArea);
            f[1] = double.IsNaN(r.LandArea) ? double.NaN : Math.Log(r.LandArea);
            f[2] = r.Bedrooms;
            f[3] = r.Bathrooms;
            f[4] = double.IsNaN(r.FloorArea) ? 1.0 : 0.0;
            f[5] = double.IsNaN(r.LandArea) ? 1.0 : 0.0;
            f[6] = double.IsNaN(r.Bedrooms) ? 1.0 : 0.0;
            f[7] = double.IsNaN(r.Bathrooms) ? 1.0 : 0.0;
            return f;
        }
    }
}
