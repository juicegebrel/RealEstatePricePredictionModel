using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace RealEstatePricePredictionModel
{
    /// <summary>Builds the human-readable console/log text: dataset summary, final results and analysis.</summary>
    public static class ReportBuilder
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public static string DatasetSummary(string path, LoadResult load, PreparedData d)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("=== DATASET ===");
            sb.AppendLine("File: " + System.IO.Path.GetFileName(path));
            sb.AppendLine(string.Format(Inv, "Rows read: {0}, usable: {1}, skipped: {2} (no valid price)", load.RowsRead, load.Records.Count, load.RowsSkipped));
            foreach (string w in load.Warnings) sb.AppendLine("  ! " + w);
            StringBuilder missing = new StringBuilder();
            foreach (KeyValuePair<string, int> kv in load.MissingCounts)
            {
                if (missing.Length > 0) missing.Append(", ");
                missing.Append(kv.Key + " " + kv.Value.ToString(Inv));
            }
            sb.AppendLine("Missing feature values among usable rows (median-imputed + flagged): " + missing);
            sb.AppendLine(string.Format(Inv, "Room counts above {0} treated as entry errors and capped: {1} values", CsvHouseLoader.MaxRooms, load.ValuesClipped));
            sb.AppendLine(string.Format(Inv, "Network inputs: {0} ({1}), output: 1 (ln price)", d.InputCount, string.Join(", ", FeatureEncoder.FeatureNames)));
            sb.AppendLine(string.Format(Inv, "Split (shuffled, fixed seed): train {0} / validation {1} / test {2}", d.TrainX.Length, d.ValX.Length, d.TestX.Length));
            sb.AppendLine(string.Format(Inv, "Train price: min PHP {0:N0}, geometric mean PHP {1:N0}, max PHP {2:N0}", d.TrainPrice.Min(), d.BaselinePrice, d.TrainPrice.Max()));
            RegressionMetrics bv = RegressionMetrics.Baseline(d.ValPrice, d.BaselinePrice);
            RegressionMetrics bt = RegressionMetrics.Baseline(d.TestPrice, d.BaselinePrice);
            sb.AppendLine(string.Format(Inv, "Baseline (always predict the geometric-mean price): validation RMSLE {0:F3}, test RMSLE {1:F3}", bv.Rmsle, bt.Rmsle));
            sb.AppendLine("Features: MinMax-scaled to [0,1] (fitted on train only). Target: ln(price) scaled to [0.1,0.9].");
            sb.AppendLine();
            return sb.ToString();
        }

        public static string TrainingReport(TrainingResult r, PreparedData d)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine();
            sb.AppendLine(string.Format(Inv, "=== FINAL RESULTS (hidden={0}, learning rate={1}, epochs={2}, seed={3}) ===",
                r.Config.HiddenNodes, r.Config.LearningRate, r.History.Count, r.Config.Seed));
            if (r.Diverged)
            {
                sb.AppendLine("Training diverged (loss became NaN/Infinity). Lower the learning rate and try again.");
                return sb.ToString();
            }
            if (r.TestMetrics == null)
            {
                sb.AppendLine("Run was cancelled before a final evaluation was made.");
                return sb.ToString();
            }

            RegressionMetrics baseline = RegressionMetrics.Baseline(d.TestPrice, d.BaselinePrice);
            EpochRecord last = r.History[r.History.Count - 1];
            sb.AppendLine(string.Format(Inv, "Training time {0:F1}s. Lowest validation loss {1:F6} at epoch {2}. Final loss: train {3:F6}, validation {4:F6}.",
                r.Seconds, r.BestValLoss, r.BestValEpoch, last.TrainLoss, last.ValLoss));
            sb.AppendLine();
            AppendMetricsTable(sb, r.TrainMetrics, r.ValMetrics, r.TestMetrics, baseline);
            sb.AppendLine();
            sb.AppendLine("--- Automatic interpretation ---");
            foreach (string line in Interpret(r.TrainMetrics, r.TestMetrics, baseline)) sb.AppendLine(" - " + line);
            return sb.ToString();
        }

        private static void AppendMetricsTable(StringBuilder sb, RegressionMetrics train, RegressionMetrics val, RegressionMetrics test, RegressionMetrics baseline)
        {
            sb.AppendLine(string.Format("{0,-24}{1,14}{2,14}{3,14}{4,16}", "Metric", "Train", "Validation", "Test", "Test baseline"));
            sb.AppendLine(Row("RMSLE (ln price)", train.Rmsle, val.Rmsle, test.Rmsle, baseline.Rmsle, "0.000"));
            sb.AppendLine(Row("R-squared (ln price)", train.R2Log, val.R2Log, test.R2Log, baseline.R2Log, "0.000"));
            sb.AppendLine(Row("Median abs % error", train.MedianApe, val.MedianApe, test.MedianApe, baseline.MedianApe, "0.0"));
            sb.AppendLine(Row("Within +-25% (%)", train.WithinTwentyFive, val.WithinTwentyFive, test.WithinTwentyFive, baseline.WithinTwentyFive, "0.0"));
            sb.AppendLine(Row("MAPE (%)", train.Mape, val.Mape, test.Mape, baseline.Mape, "0.0"));
            sb.AppendLine(Row("RMSE (PHP)", train.Rmse, val.Rmse, test.Rmse, baseline.Rmse, "N0"));
            sb.AppendLine(Row("MAE (PHP)", train.Mae, val.Mae, test.Mae, baseline.Mae, "N0"));
            sb.AppendLine(Row("MSE (PHP^2)", train.Mse, val.Mse, test.Mse, baseline.Mse, "0.00e+00"));
            sb.AppendLine(Row("R-squared (raw PHP)", train.R2, val.R2, test.R2, baseline.R2, "0.000"));
        }

        /// <summary>Report for a model restored from a weights file (no training history available).</summary>
        public static string LoadedModelReport(string weightsPath, int hiddenNodes, RegressionMetrics train,
                                               RegressionMetrics val, RegressionMetrics test, PreparedData d)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine();
            sb.AppendLine(string.Format(Inv, "=== LOADED MODEL (hidden={0}) ===", hiddenNodes));
            sb.AppendLine("Weights: " + System.IO.Path.GetFileName(weightsPath));
            sb.AppendLine("Evaluated on the current dataset split. The numbers are only meaningful if the weights were trained on this same dataset.");
            sb.AppendLine();
            AppendMetricsTable(sb, train, val, test, RegressionMetrics.Baseline(d.TestPrice, d.BaselinePrice));
            sb.AppendLine();
            sb.AppendLine("--- Automatic interpretation ---");
            foreach (string line in Interpret(train, test, RegressionMetrics.Baseline(d.TestPrice, d.BaselinePrice))) sb.AppendLine(" - " + line);
            return sb.ToString();
        }

        private static string Row(string label, double a, double b, double c, double d, string format)
        {
            return string.Format("{0,-24}{1,14}{2,14}{3,14}{4,16}", label,
                a.ToString(format, Inv), b.ToString(format, Inv), c.ToString(format, Inv), d.ToString(format, Inv));
        }

        private static List<string> Interpret(RegressionMetrics train, RegressionMetrics test, RegressionMetrics baseline)
        {
            List<string> lines = new List<string>();
            lines.Add(string.Format(Inv, "Test RMSLE {0:F3} vs {1:F3} for the predict-the-typical-price baseline: a typical prediction is off by a factor of about {2:F2}x (baseline: {3:F2}x).",
                test.Rmsle, baseline.Rmsle, Math.Exp(test.Rmsle), Math.Exp(baseline.Rmsle)));

            if (test.R2Log < 0.05)
                lines.Add(string.Format(Inv, "R-squared (ln price) = {0:F3}: the model explains essentially none of the price variation on unseen listings.", test.R2Log));
            else if (test.R2Log < 0.7)
                lines.Add(string.Format(Inv, "R-squared (ln price) = {0:F3}: the model captures part of the price variation; more tuning or better features may help.", test.R2Log));
            else
                lines.Add(string.Format(Inv, "R-squared (ln price) = {0:F3}: the model explains most of the price variation on unseen listings.", test.R2Log));

            lines.Add(string.Format(Inv, "Half of the test listings are predicted within {0:F1}% of the real price; {1:F1}% are within +-25%.", test.MedianApe, test.WithinTwentyFive));
            lines.Add(string.Format(Inv, "Raw-peso RMSE (PHP {0:N0}) is much larger than the MAE (PHP {1:N0}) because a few extremely expensive listings dominate squared errors; this is why log-scale metrics are used for model selection.", test.Rmse, test.Mae));

            double ratio = test.Rmsle / Math.Max(train.Rmsle, 1e-9);
            if (ratio > 1.2)
                lines.Add(string.Format(Inv, "Test RMSLE is {0:F2}x the training RMSLE: the network is overfitting (memorising training houses). Try fewer hidden nodes or fewer epochs.", ratio));
            else
                lines.Add(string.Format(Inv, "Test RMSLE is only {0:F2}x the training RMSLE: little overfitting.", ratio));
            return lines;
        }

        public static string TuningReport(List<ExperimentRow> rows, ExperimentRow best, PreparedData d, int seeds)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine();
            sb.AppendLine(string.Format(Inv, "=== HYPERPARAMETER EXPERIMENTS (validation set, mean of {0} seeds per cell, lower RMSLE = better) ===", seeds));
            sb.AppendLine(string.Format("{0,7}{1,8}{2,8}{3,13}{4,13}{5,11}{6,13}", "Hidden", "LR", "Epochs", "TrainRMSLE", "ValRMSLE", "ValR2log", "ValMedAPE%"));
            foreach (ExperimentRow r in rows)
            {
                if (r.Diverged)
                {
                    sb.AppendLine(string.Format("{0,7}{1,8}{2,8}   diverged (NaN)", r.HiddenNodes, r.LearningRate.ToString("0.###", Inv), r.Epochs));
                    continue;
                }
                string mark = (r == best) ? "  <-- best" : string.Empty;
                sb.AppendLine(string.Format("{0,7}{1,8}{2,8}{3,13}{4,13}{5,11}{6,13}{7}", r.HiddenNodes, r.LearningRate.ToString("0.###", Inv), r.Epochs,
                    r.TrainRmsle.ToString("0.000", Inv), r.ValRmsle.ToString("0.000", Inv), r.ValR2Log.ToString("0.000", Inv), r.ValMedianApe.ToString("0.0", Inv), mark));
            }

            RegressionMetrics baseline = RegressionMetrics.Baseline(d.ValPrice, d.BaselinePrice);
            sb.AppendLine();
            sb.AppendLine(string.Format(Inv, "Reference: the predict-the-typical-price baseline has validation RMSLE {0:F3}.", baseline.Rmsle));

            List<ExperimentRow> ok = rows.Where(x => !x.Diverged).ToList();
            if (ok.Count > 0)
            {
                sb.AppendLine("Average validation RMSLE per setting (marginal effect of each hyperparameter):");
                sb.AppendLine("  hidden nodes : " + Marginal(ok, x => x.HiddenNodes.ToString(Inv)));
                sb.AppendLine("  learning rate: " + Marginal(ok, x => x.LearningRate.ToString("0.###", Inv)));
                sb.AppendLine("  epochs       : " + Marginal(ok, x => x.Epochs.ToString(Inv)));
            }
            if (best != null)
            {
                sb.AppendLine();
                sb.AppendLine(string.Format(Inv, "BEST (lowest validation RMSLE): hidden={0}, learning rate={1}, epochs={2} -> validation RMSLE {3:F3}, R2(ln) {4:F3}, median error {5:F1}%.",
                    best.HiddenNodes, best.LearningRate, best.Epochs, best.ValRmsle, best.ValR2Log, best.ValMedianApe));
                sb.AppendLine("The settings were copied into the controls. Click 'Train model' for the final run (the test set is used only there).");
            }
            return sb.ToString();
        }

        private static string Marginal(List<ExperimentRow> rows, Func<ExperimentRow, string> key)
        {
            List<string> parts = new List<string>();
            foreach (var group in rows.GroupBy(key))
                parts.Add(group.Key + " -> " + group.Average(x => x.ValRmsle).ToString("0.000", Inv));
            return string.Join("   ", parts.ToArray());
        }
    }
}
