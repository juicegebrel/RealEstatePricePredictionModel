namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// One validated listing of PH_Housing.csv. Numeric fields that were empty / invalid in the file
    /// are stored as double.NaN ("missing") and handled later by the MedianImputer.
    /// </summary>
    public class HouseRecord
    {
        public double Price;        // PHP, always valid (rows without a price are skipped)
        public double FloorArea;    // square metres, NaN = missing
        public double LandArea;     // square metres, NaN = missing
        public double Bedrooms;     // NaN = missing
        public double Bathrooms;    // NaN = missing
    }
}
