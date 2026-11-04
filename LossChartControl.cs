using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Windows.Forms;

namespace RealEstatePricePredictionModel
{
    /// <summary>Line chart of training loss and validation loss against the epoch number.</summary>
    public class LossChartControl : ChartControlBase
    {
        private readonly List<double> epochs = new List<double>();
        private readonly List<double> trainLoss = new List<double>();
        private readonly List<double> valLoss = new List<double>();

        public void Clear()
        {
            epochs.Clear(); trainLoss.Clear(); valLoss.Clear();
            Invalidate();
        }

        public void AddPoint(int epoch, double train, double validation)
        {
            epochs.Add(epoch); trainLoss.Add(train); valLoss.Add(validation);
            Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            DrawTitle(g, "Loss per epoch (MSE on scaled ln price)");

            if (epochs.Count < 2) { DrawMessage(g, "Train the model to see the loss curve."); return; }

            double xMin = epochs[0], xMax = epochs[epochs.Count - 1];
            double yMin = double.MaxValue, yMax = double.MinValue;
            for (int i = 0; i < epochs.Count; i++)
            {
                yMin = Math.Min(yMin, Math.Min(trainLoss[i], valLoss[i]));
                yMax = Math.Max(yMax, Math.Max(trainLoss[i], valLoss[i]));
            }
            double pad = Math.Max((yMax - yMin) * 0.05, 1e-9);
            yMin -= pad; yMax += pad;

            Rectangle plot = PlotArea;
            DrawAxes(g, plot, xMin, xMax, yMin, yMax,
                     delegate(double v) { return v.ToString("0", CultureInfo.InvariantCulture); },
                     delegate(double v) { return v.ToString("0.0000", CultureInfo.InvariantCulture); });

            DrawSeries(g, plot, trainLoss, Color.FromArgb(31, 119, 180), xMin, xMax, yMin, yMax);
            DrawSeries(g, plot, valLoss, Color.FromArgb(230, 120, 20), xMin, xMax, yMin, yMax);
            DrawLegend(g, plot);
        }

        private void DrawSeries(Graphics g, Rectangle plot, List<double> values, Color color,
                                double xMin, double xMax, double yMin, double yMax)
        {
            PointF[] points = new PointF[values.Count];
            for (int i = 0; i < values.Count; i++)
                points[i] = Map(plot, epochs[i], values[i], xMin, xMax, yMin, yMax);
            using (Pen pen = new Pen(color, 2f)) g.DrawLines(pen, points);
        }

        private void DrawLegend(Graphics g, Rectangle plot)
        {
            using (Brush text = new SolidBrush(ForeColor))
            using (Pen train = new Pen(Color.FromArgb(31, 119, 180), 3f))
            using (Pen val = new Pen(Color.FromArgb(230, 120, 20), 3f))
            {
                int x = plot.Right - 190;
                g.DrawLine(train, x, plot.Top + 12, x + 18, plot.Top + 12);
                g.DrawString("Training", Font, text, x + 22, plot.Top + 5);
                g.DrawLine(val, x + 90, plot.Top + 12, x + 108, plot.Top + 12);
                g.DrawString("Validation", Font, text, x + 112, plot.Top + 5);
            }
        }
    }
}
