namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Everything the trainer needs: scaled features/targets for the three splits, the original prices
    /// in PHP (for metrics) and the fitted preprocessing objects (for predicting new houses later).
    /// </summary>
    public class PreparedData
    {
        public double[][] TrainX, ValX, TestX;
        public double[] TrainY, ValY, TestY;               // ln(price) scaled to [0.1, 0.9]
        public double[] TrainPrice, ValPrice, TestPrice;   // original prices in PHP
        public MedianImputer Imputer;
        public MinMaxScaler FeatureScaler;
        public TargetScaler TargetScaler;
        public double BaselinePrice;                       // geometric mean of train prices (naive model)

        public int InputCount { get { return TrainX[0].Length; } }

        /// <summary>Raw encoded features -> imputed -> MinMax scaled, exactly as the training data was.</summary>
        public double[] PrepareFeatures(double[] encoded)
        {
            return FeatureScaler.Transform(Imputer.Transform(encoded));
        }
    }
}
