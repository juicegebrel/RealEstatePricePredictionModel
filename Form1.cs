using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Windows.Forms;
using Backprop;

namespace RealEstatePricePredictionModel
{
    /// <summary>
    /// Main window. Trains the custom MLP on the house-price data (regression), runs the hyperparameter
    /// grid search, shows the loss curve / actual-vs-predicted plot / metrics and lets the user
    /// estimate the price of a single house. Heavy work runs on a BackgroundWorker so the UI stays responsive.
    /// </summary>
    public partial class Form1 : Form
    {
        // ---- fixed experiment settings ----
        private const int SplitSeed = 42;
        private const double TrainFraction = 0.70;
        private const double ValidationFraction = 0.15;   // remaining 15% = test set
        private const int LogInterval = 50;               // print the loss every N epochs
        private const int TuningSeeds = 3;
        private static readonly int[] TuningHiddenSizes = new int[] { 4, 8, 16, 32 };
        private static readonly double[] TuningLearningRates = new double[] { 0.05, 0.1, 0.3, 0.5 };
        private static readonly int[] TuningEpochCheckpoints = new int[] { 100, 300, 500 };

        private enum JobKind { Train, Tune }

        private class WorkerJob
        {
            public JobKind Kind;
            public TrainingConfig Config;
        }

        private class DatasetItem
        {
            public string Name;
            public string Path;
            public override string ToString() { return Name; }
        }

        // ---- toolbar controls ----
        private ComboBox cboDataset;
        private Button btnBrowse, btnTrain, btnTune, btnLoadWeights, btnCancel, btnOpenResults;
        private NumericUpDown nudHidden, nudLearning, nudEpochs, nudSeed;

        // ---- output controls ----
        private LossChartControl lossChart;
        private ScatterChartControl scatterChart;
        private TextBox txtLog;
        private ProgressBar progressBar;
        private Label lblStatus;

        // ---- prediction controls ----
        private NumericUpDown nudFloorArea, nudLandArea, nudBedrooms, nudBathrooms;
        private Button btnPredict;
        private Label lblPrediction;

        private BackgroundWorker worker;

        // ---- state ----
        private PreparedData data;            // data of the currently loaded dataset
        private string loadedPath;
        private TrainingConfig activeConfig;
        private NeuralNet trainedNet;         // last trained model (for single predictions)
        private PreparedData trainedData;     // scalers that belong to trainedNet

        public Form1()
        {
            InitializeComponent();
            BuildUi();

            worker = new BackgroundWorker();
            worker.WorkerReportsProgress = true;
            worker.WorkerSupportsCancellation = true;
            worker.DoWork += Worker_DoWork;
            worker.ProgressChanged += Worker_ProgressChanged;
            worker.RunWorkerCompleted += Worker_RunWorkerCompleted;

            FillDatasetList();
            Log("Ready. Choose a dataset, then click 'Train model' (or 'Tune hyperparameters' first).");
        }

        // =====================================================================
        //  UI construction
        // =====================================================================

        private void BuildUi()
        {
            MinimumSize = new Size(1000, 680);

            TableLayoutPanel root = new TableLayoutPanel();
            root.Dock = DockStyle.Fill;
            root.ColumnCount = 1;
            root.RowCount = 3;
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            Controls.Add(root);

            root.Controls.Add(BuildToolbar(), 0, 0);
            root.Controls.Add(BuildMainArea(), 0, 1);
            root.Controls.Add(BuildStatusBar(), 0, 2);
        }

        private Control BuildToolbar()
        {
            FlowLayoutPanel bar = new FlowLayoutPanel();
            bar.Dock = DockStyle.Fill;
            bar.AutoSize = true;
            bar.WrapContents = true;
            bar.Padding = new Padding(6, 4, 6, 4);

            cboDataset = new ComboBox();
            cboDataset.DropDownStyle = ComboBoxStyle.DropDownList;
            cboDataset.Width = 250;
            cboDataset.SelectedIndexChanged += delegate { data = null; };   // force reload on next run
            AddLabeled(bar, "Dataset:", cboDataset);

            btnBrowse = MakeButton("Browse...", 80);
            btnBrowse.Click += BrowseClicked;
            bar.Controls.Add(btnBrowse);

            nudHidden = MakeNumeric(1, 256, 16, 0, 1, 60);
            AddLabeled(bar, "Hidden nodes:", nudHidden);

            nudLearning = MakeNumeric(0.001M, 2M, 0.3M, 3, 0.01M, 70);
            AddLabeled(bar, "Learning rate:", nudLearning);

            nudEpochs = MakeNumeric(1, 100000, 500, 0, 50, 70);
            AddLabeled(bar, "Epochs:", nudEpochs);

            nudSeed = MakeNumeric(0, 1000000, 1, 0, 1, 70);
            AddLabeled(bar, "Seed:", nudSeed);

            btnTrain = MakeButton("Train model", 100);
            btnTrain.Click += TrainClicked;
            bar.Controls.Add(btnTrain);

            btnTune = MakeButton("Tune hyperparameters", 150);
            btnTune.Click += TuneClicked;
            bar.Controls.Add(btnTune);

            btnLoadWeights = MakeButton("Load weights...", 110);
            btnLoadWeights.Click += LoadWeightsClicked;
            bar.Controls.Add(btnLoadWeights);

            btnCancel = MakeButton("Cancel", 70);
            btnCancel.Enabled = false;
            btnCancel.Click += delegate { if (worker.IsBusy) worker.CancelAsync(); };
            bar.Controls.Add(btnCancel);

            btnOpenResults = MakeButton("Open results folder", 130);
            btnOpenResults.Click += delegate { Process.Start(AppPaths.ResultsDirectory); };
            bar.Controls.Add(btnOpenResults);

            return bar;
        }

        private Control BuildMainArea()
        {
            TableLayoutPanel main = new TableLayoutPanel();
            main.Dock = DockStyle.Fill;
            main.ColumnCount = 2;
            main.RowCount = 1;
            main.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 52F));
            main.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 48F));

            TableLayoutPanel charts = new TableLayoutPanel();
            charts.Dock = DockStyle.Fill;
            charts.ColumnCount = 1;
            charts.RowCount = 2;
            charts.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));
            charts.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));
            lossChart = new LossChartControl();
            lossChart.Dock = DockStyle.Fill;
            scatterChart = new ScatterChartControl();
            scatterChart.Dock = DockStyle.Fill;
            charts.Controls.Add(lossChart, 0, 0);
            charts.Controls.Add(scatterChart, 0, 1);

            TableLayoutPanel right = new TableLayoutPanel();
            right.Dock = DockStyle.Fill;
            right.ColumnCount = 1;
            right.RowCount = 2;
            right.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
            right.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            txtLog = new TextBox();
            txtLog.Multiline = true;
            txtLog.ReadOnly = true;
            txtLog.ScrollBars = ScrollBars.Both;
            txtLog.WordWrap = false;
            txtLog.Font = new Font("Consolas", 8.5F);
            txtLog.Dock = DockStyle.Fill;
            txtLog.BackColor = Color.White;
            right.Controls.Add(txtLog, 0, 0);
            right.Controls.Add(BuildPredictionPanel(), 0, 1);

            main.Controls.Add(charts, 0, 0);
            main.Controls.Add(right, 1, 0);
            return main;
        }

        private Control BuildPredictionPanel()
        {
            GroupBox box = new GroupBox();
            box.Text = "Predict the price of one house (uses the last trained model)";
            box.Dock = DockStyle.Fill;
            box.AutoSize = true;

            TableLayoutPanel grid = new TableLayoutPanel();
            grid.Dock = DockStyle.Fill;
            grid.AutoSize = true;
            grid.ColumnCount = 4;
            grid.Padding = new Padding(4);

            nudFloorArea = MakeNumeric(10, 5000, 100, 0, 5, 80);
            nudLandArea = MakeNumeric(10, 50000, 120, 0, 10, 80);
            nudBedrooms = MakeNumeric(1, 10, 3, 0, 1, 80);
            nudBathrooms = MakeNumeric(1, 10, 2, 0, 1, 80);

            AddGridRow(grid, "Floor area (sqm):", nudFloorArea, "Land area (sqm):", nudLandArea);
            AddGridRow(grid, "Bedrooms:", nudBedrooms, "Bathrooms:", nudBathrooms);

            btnPredict = MakeButton("Predict price", 110);
            btnPredict.Click += PredictClicked;
            lblPrediction = new Label();
            lblPrediction.AutoSize = true;
            lblPrediction.Font = new Font(Font, FontStyle.Bold);
            lblPrediction.Text = "Train a model first.";
            lblPrediction.Margin = new Padding(6, 9, 3, 3);
            grid.Controls.Add(btnPredict);
            grid.Controls.Add(lblPrediction);
            grid.SetColumnSpan(lblPrediction, 3);

            box.Controls.Add(grid);
            return box;
        }

        private Control BuildStatusBar()
        {
            TableLayoutPanel bar = new TableLayoutPanel();
            bar.Dock = DockStyle.Fill;
            bar.AutoSize = true;
            bar.ColumnCount = 2;
            bar.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            bar.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 260F));

            lblStatus = new Label();
            lblStatus.Text = "Idle";
            lblStatus.AutoSize = true;
            lblStatus.Margin = new Padding(6, 5, 3, 3);
            progressBar = new ProgressBar();
            progressBar.Dock = DockStyle.Fill;
            progressBar.Height = 18;

            bar.Controls.Add(lblStatus, 0, 0);
            bar.Controls.Add(progressBar, 1, 0);
            return bar;
        }

        private static Button MakeButton(string text, int width)
        {
            Button b = new Button();
            b.Text = text;
            b.Width = width;
            b.Height = 26;
            b.Margin = new Padding(6, 3, 3, 3);
            return b;
        }

        private static NumericUpDown MakeNumeric(decimal min, decimal max, decimal value, int decimals, decimal increment, int width)
        {
            NumericUpDown n = new NumericUpDown();
            n.Minimum = min;
            n.Maximum = max;
            n.DecimalPlaces = decimals;
            n.Increment = increment;
            n.Value = value;
            n.Width = width;
            return n;
        }

        private static void AddLabeled(FlowLayoutPanel panel, string text, Control control)
        {
            Label label = new Label();
            label.Text = text;
            label.AutoSize = true;
            label.Margin = new Padding(10, 8, 2, 0);
            control.Margin = new Padding(0, 4, 4, 0);
            panel.Controls.Add(label);
            panel.Controls.Add(control);
        }

        private static void AddGridRow(TableLayoutPanel grid, string label1, Control c1, string label2, Control c2)
        {
            grid.Controls.Add(MakeGridLabel(label1));
            grid.Controls.Add(c1);
            grid.Controls.Add(MakeGridLabel(label2));
            grid.Controls.Add(c2);
        }

        private static Label MakeGridLabel(string text)
        {
            Label l = new Label();
            l.Text = text;
            l.AutoSize = true;
            l.Margin = new Padding(3, 8, 3, 0);
            return l;
        }

        // =====================================================================
        //  Dataset handling
        // =====================================================================

        private void FillDatasetList()
        {
            string[] names = new string[] { AppPaths.DefaultDataset };
            foreach (string name in names)
            {
                string path = AppPaths.FindDataFile(name);
                if (path == null) continue;
                DatasetItem item = new DatasetItem();
                item.Name = name;
                item.Path = path;
                cboDataset.Items.Add(item);
            }
            if (cboDataset.Items.Count > 0) cboDataset.SelectedIndex = 0;
            else Log("No bundled CSV found - use 'Browse...' to pick a dataset.");
        }

        private void BrowseClicked(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*";
                if (dialog.ShowDialog() != DialogResult.OK) return;
                DatasetItem item = new DatasetItem();
                item.Name = System.IO.Path.GetFileName(dialog.FileName);
                item.Path = dialog.FileName;
                cboDataset.Items.Add(item);
                cboDataset.SelectedItem = item;
            }
        }

        /// <summary>Loads and prepares the selected dataset if that has not been done yet. Returns false on failure.</summary>
        private bool EnsureDataLoaded()
        {
            if (data != null) return true;
            DatasetItem item = cboDataset.SelectedItem as DatasetItem;
            if (item == null)
            {
                MessageBox.Show(this, "Please choose a dataset first.", Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                return false;
            }
            try
            {
                LoadResult load = CsvHouseLoader.Load(item.Path);
                data = DataPreparer.Prepare(load.Records, TrainFraction, ValidationFraction, SplitSeed);
                loadedPath = item.Path;
                Log(ReportBuilder.DatasetSummary(item.Path, load, data));
                return true;
            }
            catch (Exception ex)
            {
                data = null;
                MessageBox.Show(this, "Could not load the dataset:\r\n" + ex.Message, Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
                return false;
            }
        }

        // =====================================================================
        //  Button handlers
        // =====================================================================

        private void TrainClicked(object sender, EventArgs e)
        {
            if (worker.IsBusy || !EnsureDataLoaded()) return;

            TrainingConfig cfg = new TrainingConfig();
            cfg.HiddenNodes = (int)nudHidden.Value;
            cfg.LearningRate = (double)nudLearning.Value;
            cfg.Epochs = (int)nudEpochs.Value;
            cfg.Seed = (int)nudSeed.Value;
            cfg.EvaluateOnTest = true;
            activeConfig = cfg;

            lossChart.Clear();
            scatterChart.Clear();
            Log(string.Format(CultureInfo.InvariantCulture,
                "=== TRAINING: hidden={0}, learning rate={1}, epochs={2}, seed={3} ===\r\nLoss = MSE on scaled ln(price) (0.1-0.9). Train loss = running average over the epoch.",
                cfg.HiddenNodes, cfg.LearningRate, cfg.Epochs, cfg.Seed));

            WorkerJob job = new WorkerJob();
            job.Kind = JobKind.Train;
            job.Config = cfg;
            StartJob(job, "Training...");
        }

        private void TuneClicked(object sender, EventArgs e)
        {
            if (worker.IsBusy || !EnsureDataLoaded()) return;
            int runs = TuningHiddenSizes.Length * TuningLearningRates.Length * TuningSeeds;
            Log(string.Format(CultureInfo.InvariantCulture,
                "=== TUNING: {0} hidden sizes x {1} learning rates x {2} epoch checkpoints, {3} seeds each ({4} training runs; this can take a few minutes) ===",
                TuningHiddenSizes.Length, TuningLearningRates.Length, TuningEpochCheckpoints.Length, TuningSeeds, runs));

            WorkerJob job = new WorkerJob();
            job.Kind = JobKind.Tune;
            StartJob(job, "Tuning...");
        }

        private void StartJob(WorkerJob job, string status)
        {
            SetBusy(true);
            progressBar.Value = 0;
            lblStatus.Text = status;
            worker.RunWorkerAsync(job);
        }

        private void SetBusy(bool busy)
        {
            btnTrain.Enabled = !busy;
            btnTune.Enabled = !busy;
            btnLoadWeights.Enabled = !busy;
            btnBrowse.Enabled = !busy;
            cboDataset.Enabled = !busy;
            btnCancel.Enabled = busy;
        }

        /// <summary>
        /// Restores a model saved by NeuralNet.saveWeights (model_weights.txt). The file stores only weights and
        /// biases, so the hidden-layer size is derived from the line count
        /// (lines = hidden * (inputs + 2) + 1) and the imputer/scalers are rebuilt from the selected dataset
        /// (same CSV + fixed split seed = same preprocessing as in training).
        /// </summary>
        private void LoadWeightsClicked(object sender, EventArgs e)
        {
            if (worker.IsBusy || !EnsureDataLoaded()) return;

            string path;
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Open a weights file saved by this program";
                dialog.Filter = "Weights (*.txt)|*.txt|All files (*.*)|*.*";
                dialog.InitialDirectory = AppPaths.ResultsDirectory;
                dialog.FileName = "model_weights.txt";
                if (dialog.ShowDialog() != DialogResult.OK) return;
                path = dialog.FileName;
            }

            try
            {
                int inputs = data.InputCount;
                int lineCount = File.ReadAllLines(path).Length;
                int perHidden = inputs + 2;               // input->hidden weights + hidden->output weight + hidden bias
                if (lineCount < perHidden + 1 || (lineCount - 1) % perHidden != 0)
                    throw new InvalidDataException(string.Format(CultureInfo.InvariantCulture,
                        "The file has {0} values, which does not match a network with {1} inputs and 1 output.", lineCount, inputs));
                int hidden = (lineCount - 1) / perHidden;

                NeuralNet net = new NeuralNet(inputs, hidden, 1);
                net.loadWeights(path);

                RegressionMetrics train = Trainer.Evaluate(net, data.TrainX, data.TrainPrice, data.TargetScaler);
                RegressionMetrics val = Trainer.Evaluate(net, data.ValX, data.ValPrice, data.TargetScaler);
                double[] testPredictions = Trainer.PredictPrices(net, data.TestX, data.TargetScaler);
                RegressionMetrics test = RegressionMetrics.Compute(data.TestPrice, testPredictions);

                trainedNet = net;
                trainedData = data;
                nudHidden.Value = hidden;
                lossChart.Clear();
                scatterChart.SetPoints(data.TestPrice, testPredictions);
                lblPrediction.Text = "Model loaded - enter a house and click Predict.";
                Log(ReportBuilder.LoadedModelReport(path, hidden, train, val, test, data));
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, "Could not load the weights:\r\n" + ex.Message, Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void PredictClicked(object sender, EventArgs e)
        {
            if (trainedNet == null || trainedData == null)
            {
                lblPrediction.Text = "Train a model first.";
                return;
            }
            HouseRecord house = new HouseRecord();
            house.FloorArea = (double)nudFloorArea.Value;
            house.LandArea = (double)nudLandArea.Value;
            house.Bedrooms = (double)nudBedrooms.Value;
            house.Bathrooms = (double)nudBathrooms.Value;

            double[] scaled = trainedData.PrepareFeatures(FeatureEncoder.Encode(house));
            double price = trainedData.TargetScaler.Unscale(Trainer.PredictScaled(trainedNet, scaled));
            lblPrediction.Text = "Estimated price: PHP " + price.ToString("N0", CultureInfo.InvariantCulture);
        }

        // =====================================================================
        //  Background worker
        // =====================================================================

        private void Worker_DoWork(object sender, DoWorkEventArgs e)
        {
            WorkerJob job = (WorkerJob)e.Argument;
            if (job.Kind == JobKind.Train) e.Result = RunTraining(job.Config);
            else e.Result = RunTuning();
        }

        private TrainingResult RunTraining(TrainingConfig cfg)
        {
            int chartStep = Math.Max(1, cfg.Epochs / 150);
            return Trainer.Train(data, cfg,
                delegate(EpochRecord r)
                {
                    bool report = r.Epoch == 1 || r.Epoch == cfg.Epochs || r.Epoch % chartStep == 0 || r.Epoch % LogInterval == 0;
                    if (report) worker.ReportProgress(r.Epoch * 100 / cfg.Epochs, r);
                },
                delegate { return worker.CancellationPending; });
        }

        private List<ExperimentRow> RunTuning()
        {
            return ExperimentRunner.RunGrid(data, TuningHiddenSizes, TuningLearningRates, TuningEpochCheckpoints, TuningSeeds,
                delegate(int done, int total, string message)
                {
                    worker.ReportProgress(done * 100 / total, message);
                },
                delegate { return worker.CancellationPending; });
        }

        private void Worker_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            progressBar.Value = Math.Max(0, Math.Min(100, e.ProgressPercentage));

            EpochRecord rec = e.UserState as EpochRecord;
            if (rec != null)
            {
                lossChart.AddPoint(rec.Epoch, rec.TrainLoss, rec.ValLoss);
                lblStatus.Text = string.Format(CultureInfo.InvariantCulture, "Epoch {0}  train loss {1:F5}  validation loss {2:F5}", rec.Epoch, rec.TrainLoss, rec.ValLoss);
                if (activeConfig != null && (rec.Epoch == 1 || rec.Epoch % LogInterval == 0 || rec.Epoch == activeConfig.Epochs))
                    Log(string.Format(CultureInfo.InvariantCulture, "Epoch {0,5} | train loss {1:F6} | validation loss {2:F6}", rec.Epoch, rec.TrainLoss, rec.ValLoss));
                return;
            }

            string message = e.UserState as string;
            if (message != null) lblStatus.Text = "Tuning: " + message;
        }

        private void Worker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            SetBusy(false);
            progressBar.Value = 0;
            lblStatus.Text = "Idle";

            if (e.Error != null)
            {
                Log("ERROR: " + e.Error.Message);
                MessageBox.Show(this, e.Error.ToString(), Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            TrainingResult trained = e.Result as TrainingResult;
            if (trained != null) { FinishTraining(trained); return; }

            List<ExperimentRow> rows = e.Result as List<ExperimentRow>;
            if (rows != null) FinishTuning(rows);
        }

        private void FinishTraining(TrainingResult result)
        {
            if (result.Cancelled) { Log("Training cancelled."); return; }

            string report = ReportBuilder.TrainingReport(result, data);
            Log(report);
            if (result.Diverged) return;

            trainedNet = result.Network;
            trainedData = data;
            scatterChart.SetPoints(data.TestPrice, result.TestPredictions);
            lblPrediction.Text = "Model ready - enter a house and click Predict.";

            try
            {
                string dir = AppPaths.ResultsDirectory;
                ResultsWriter.WriteTrainingLog(System.IO.Path.Combine(dir, "training_log.csv"), result.History);
                ResultsWriter.WritePredictions(System.IO.Path.Combine(dir, "test_predictions.csv"), data.TestPrice, result.TestPredictions);
                ResultsWriter.WriteText(System.IO.Path.Combine(dir, "training_summary.txt"), "Dataset: " + System.IO.Path.GetFileName(loadedPath) + "\r\n" + report);
                result.Network.saveWeights(System.IO.Path.Combine(dir, "model_weights.txt"));
                Log("Saved training_log.csv, test_predictions.csv, training_summary.txt and model_weights.txt to: " + dir);
            }
            catch (Exception ex)
            {
                Log("Could not save result files: " + ex.Message);
            }
        }

        private void FinishTuning(List<ExperimentRow> rows)
        {
            ExperimentRow best = ExperimentRunner.Best(rows);
            string report = ReportBuilder.TuningReport(rows, best, data, TuningSeeds);
            Log(report);

            if (best != null)
            {
                nudHidden.Value = best.HiddenNodes;
                nudLearning.Value = (decimal)best.LearningRate;
                nudEpochs.Value = best.Epochs;
            }

            try
            {
                string dir = AppPaths.ResultsDirectory;
                ResultsWriter.WriteExperiments(System.IO.Path.Combine(dir, "experiment_results.csv"), rows);
                ResultsWriter.WriteText(System.IO.Path.Combine(dir, "tuning_summary.txt"), "Dataset: " + System.IO.Path.GetFileName(loadedPath) + "\r\n" + report);
                Log("Saved experiment_results.csv and tuning_summary.txt to: " + dir);
            }
            catch (Exception ex)
            {
                Log("Could not save result files: " + ex.Message);
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (worker != null && worker.IsBusy) worker.CancelAsync();
            base.OnFormClosing(e);
        }

        private void Log(string text)
        {
            txtLog.AppendText(text.Replace("\r\n", "\n").Replace("\n", "\r\n") + "\r\n");
        }
    }
}
