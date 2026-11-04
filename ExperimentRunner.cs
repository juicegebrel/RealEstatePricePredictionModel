using System;
using System.Collections.Generic;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Systematic grid search over hidden-layer size x learning rate x epoch count.
    /// Each (hidden, rate) pair is trained once per seed up to the largest epoch checkpoint; metrics are
    /// recorded at every checkpoint, which is identical to separate runs of that length (same seed =
    /// same weight initialisation and shuffling) but 3x cheaper. Results are averaged over several seeds
    /// to reduce the effect of random weight initialisation. Selection uses VALIDATION data only (RMSLE, because peso RMSE is dominated by a few
    /// extremely expensive listings).
    /// </summary>
    public static class ExperimentRunner
    {
        public static List<ExperimentRow> RunGrid(PreparedData data, int[] hiddenSizes, double[] rates,
                                                  int[] checkpoints, int seeds,
                                                  Action<int, int, string> progress, Func<bool> isCancelled)
        {
            List<ExperimentRow> rows = new List<ExperimentRow>();
            int maxEpochs = 0;
            foreach (int c in checkpoints) if (c > maxEpochs) maxEpochs = c;

            int totalRuns = hiddenSizes.Length * rates.Length * seeds;
            int doneRuns = 0;

            foreach (int hidden in hiddenSizes)
            {
                foreach (double rate in rates)
                {
                    int cpCount = checkpoints.Length;
                    double[] trainRmsle = new double[cpCount];
                    double[] valRmsle = new double[cpCount];
                    double[] valR2Log = new double[cpCount];
                    double[] valMedApe = new double[cpCount];
                    int[] valid = new int[cpCount];

                    for (int s = 0; s < seeds; s++)
                    {
                        TrainingConfig cfg = new TrainingConfig();
                        cfg.HiddenNodes = hidden;
                        cfg.LearningRate = rate;
                        cfg.Epochs = maxEpochs;
                        cfg.Seed = 1000 + s;
                        cfg.CheckpointEpochs = checkpoints;

                        TrainingResult res = Trainer.Train(data, cfg, null, isCancelled);
                        if (res.Cancelled) return rows;

                        foreach (Checkpoint cp in res.Checkpoints)
                        {
                            int idx = Array.IndexOf(checkpoints, cp.Epoch);
                            trainRmsle[idx] += cp.Train.Rmsle;
                            valRmsle[idx] += cp.Validation.Rmsle;
                            valR2Log[idx] += cp.Validation.R2Log;
                            valMedApe[idx] += cp.Validation.MedianApe;
                            valid[idx]++;
                        }

                        doneRuns++;
                        if (progress != null)
                            progress(doneRuns, totalRuns, string.Format("hidden={0}, lr={1}, seed {2}/{3}", hidden, rate, s + 1, seeds));
                    }

                    for (int i = 0; i < cpCount; i++)
                    {
                        ExperimentRow row = new ExperimentRow();
                        row.HiddenNodes = hidden;
                        row.LearningRate = rate;
                        row.Epochs = checkpoints[i];
                        row.RunsAveraged = valid[i];
                        if (valid[i] == 0) row.Diverged = true;
                        else
                        {
                            row.TrainRmsle = trainRmsle[i] / valid[i];
                            row.ValRmsle = valRmsle[i] / valid[i];
                            row.ValR2Log = valR2Log[i] / valid[i];
                            row.ValMedianApe = valMedApe[i] / valid[i];
                        }
                        rows.Add(row);
                    }
                }
            }
            return rows;
        }

        /// <summary>Row with the lowest mean validation RMSLE (error of ln price) (null if everything diverged).</summary>
        public static ExperimentRow Best(List<ExperimentRow> rows)
        {
            ExperimentRow best = null;
            foreach (ExperimentRow r in rows)
            {
                if (r.Diverged || double.IsNaN(r.ValRmsle)) continue;
                if (best == null || r.ValRmsle < best.ValRmsle) best = r;
            }
            return best;
        }
    }
}
