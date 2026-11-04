using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace RealEstatePricePredictionModel
{
    /// <summary>Outcome of reading a CSV file: the valid records plus data-quality statistics.</summary>
    public class LoadResult
    {
        public List<HouseRecord> Records = new List<HouseRecord>();
        public int RowsRead;
        public int RowsSkipped;
        public int ValuesClipped;      // implausible room counts capped at CsvHouseLoader.MaxRooms
        public Dictionary<string, int> MissingCounts = new Dictionary<string, int>();
        public List<string> Warnings = new List<string>();
    }

    /// <summary>
    /// Parses PH_Housing.csv (columns: HouseID, Description, Location, Price, Bedrooms, Bathrooms,
    /// Floor Area, Land Area, Latitude, Longitude). Latitude/Longitude are ignored. Columns are located by header name and quoted fields
    /// (descriptions contain commas) are supported.
    /// Cleaning rules:
    ///  - rows without a valid Price (the target) are skipped and counted;
    ///  - missing / non-numeric / non-positive feature values become NaN (imputed later, from TRAIN data only);
    ///  - Bedrooms / Bathrooms above MaxRooms are treated as data-entry errors and capped (e.g. 43 bathrooms).
    /// HouseID, Description, Location, Latitude and Longitude are not used as network inputs.
    /// </summary>
    public static class CsvHouseLoader
    {
        public const int MaxRooms = 10;
        private const int MaxWarnings = 10;

        private static readonly string[] RequiredColumns = new string[]
        { "Price", "Bedrooms", "Bathrooms", "Floor Area", "Land Area" };

        public static LoadResult Load(string path)
        {
            string[] lines = File.ReadAllLines(path);
            if (lines.Length < 2) throw new InvalidDataException("The CSV file has no data rows.");

            Dictionary<string, int> columns = MapHeader(SplitLine(lines[0]));
            LoadResult result = new LoadResult();
            foreach (string name in new string[] { "Bedrooms", "Bathrooms", "Floor Area", "Land Area" })
                result.MissingCounts[name] = 0;

            for (int i = 1; i < lines.Length; i++)
            {
                if (lines[i].Trim().Length == 0) continue;
                result.RowsRead++;

                string[] fields = SplitLine(lines[i]);
                double price;
                if (!TryNumber(Field(fields, columns, "Price"), out price) || price <= 0)
                {
                    Skip(result, i, "missing/invalid Price");
                    continue;
                }

                HouseRecord r = new HouseRecord();
                r.Price = price;
                r.FloorArea = OptionalPositive(fields, columns, "Floor Area", result);
                r.LandArea = OptionalPositive(fields, columns, "Land Area", result);
                r.Bedrooms = Rooms(OptionalPositive(fields, columns, "Bedrooms", result), result);
                r.Bathrooms = Rooms(OptionalPositive(fields, columns, "Bathrooms", result), result);
                result.Records.Add(r);
            }
            return result;
        }

        private static void Skip(LoadResult result, int lineIndex, string reason)
        {
            result.RowsSkipped++;
            if (result.Warnings.Count < MaxWarnings)
                result.Warnings.Add("Line " + (lineIndex + 1) + " skipped: " + reason);
        }

        /// <summary>Parses an optional positive number; empty/invalid/non-positive -> NaN and counted as missing.</summary>
        private static double OptionalPositive(string[] fields, Dictionary<string, int> columns, string name, LoadResult result)
        {
            double value;
            if (TryNumber(Field(fields, columns, name), out value) && value > 0) return value;
            result.MissingCounts[name]++;
            return double.NaN;
        }

        private static double Rooms(double value, LoadResult result)
        {
            if (double.IsNaN(value)) return value;
            value = Math.Round(value);
            if (value > MaxRooms)
            {
                result.ValuesClipped++;
                return MaxRooms;
            }
            return value;
        }

        private static Dictionary<string, int> MapHeader(string[] header)
        {
            Dictionary<string, int> map = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < header.Length; i++) map[header[i]] = i;

            foreach (string required in RequiredColumns)
            {
                if (!map.ContainsKey(required))
                    throw new InvalidDataException("Missing required column: " + required);
            }
            return map;
        }

        /// <summary>Minimal CSV splitter: commas separate fields unless inside double quotes.</summary>
        private static string[] SplitLine(string line)
        {
            List<string> fields = new List<string>();
            StringBuilder current = new StringBuilder();
            bool inQuotes = false;
            foreach (char c in line)
            {
                if (c == '"') inQuotes = !inQuotes;
                else if (c == ',' && !inQuotes)
                {
                    fields.Add(current.ToString().Trim());
                    current.Length = 0;
                }
                else current.Append(c);
            }
            fields.Add(current.ToString().Trim());
            return fields.ToArray();
        }

        private static string Field(string[] fields, Dictionary<string, int> columns, string name)
        {
            int index = columns[name];
            return index < fields.Length ? fields[index] : string.Empty;
        }

        private static bool TryNumber(string text, out double value)
        {
            return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value)
                   && !double.IsNaN(value) && !double.IsInfinity(value);
        }
    }
}
