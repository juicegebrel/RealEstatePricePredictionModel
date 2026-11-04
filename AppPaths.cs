using System;
using System.IO;

namespace RealEstatePricePredictionModel
{
    /// <summary>Locates the bundled CSV files and the folder where results are written.</summary>
    public static class AppPaths
    {
        public const string DefaultDataset = "PH_Housing.csv";

        /// <summary>Walks up from the exe folder (bin\Debug) to the folder that holds the .csproj; null if not found.</summary>
        public static string FindProjectRoot()
        {
            DirectoryInfo dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
            for (int i = 0; i < 6 && dir != null; i++)
            {
                if (dir.GetFiles("*.csproj").Length > 0) return dir.FullName;
                dir = dir.Parent;
            }
            return null;
        }

        /// <summary>Returns the full path of a bundled data file, or null when it cannot be found.</summary>
        public static string FindDataFile(string fileName)
        {
            string candidate = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName);
            if (File.Exists(candidate)) return candidate;

            string root = FindProjectRoot();
            if (root != null)
            {
                candidate = Path.Combine(root, fileName);
                if (File.Exists(candidate)) return candidate;
            }
            return null;
        }

        /// <summary>'Results' folder in the project root (so logs can be committed to GitHub), created on demand.</summary>
        public static string ResultsDirectory
        {
            get
            {
                string root = FindProjectRoot() ?? AppDomain.CurrentDomain.BaseDirectory;
                string dir = Path.Combine(root, "Results");
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                return dir;
            }
        }
    }
}
