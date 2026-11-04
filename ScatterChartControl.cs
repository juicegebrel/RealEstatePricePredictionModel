using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Windows.Forms;

namespace RealEstatePricePredictionModel
{
    /// <summary>Scatter plot of actual vs predicted price; the dashed diagonal is a perfect prediction.</summary>
    public class ScatterChartControl : ChartControlBase
    {
        private double[] actual;
        private double[] predicted;

        public void Clear()
        {
            actual = null; predicted = null;
            Invalidate();
        }

        public void SetPoints(double[] actualPrices, double[] predictedPrices)
        {
            actual = actualPrices; predicted = predictedPrices;
            Invalidate();
        }

        private static string FormatPesos(double value)
        {
            CultureInfo c = CultureInfo.InvariantCulture;
            if (value >= 1e9) return (value / 1e9).ToString("0.#", c) + "B";
            if (value >= 1e6) return (value / 1e6).ToString("0.#", c) + "M";
            return (value / 1e3).ToString("0", c) + "k";
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            DrawTitle(g, "Test set: actual vs predicted price (PHP, log scale)");

            if (actual == null || actual.Length == 0) { DrawMessage(g, "Train the model to see predictions."); return; }

            // Prices span several orders of magnitude, so both axes are drawn as log10(price).
            double lo = double.MaxValue, hi = double.MinValue;
            for (int i = 0; i < actual.Length; i++)
            {
                lo = Math.Min(lo, Math.Min(Math.Log10(actual[i]), Math.Log10(predicted[i])));
                hi = Math.Max(hi, Math.Max(Math.Log10(actual[i]), Math.Log10(predicted[i])));
            }
            double pad = Math.Max((hi - lo) * 0.03, 0.01);
            lo -= pad; hi += pad;

            Rectangle plot = PlotArea;
            Func<double, string> pesos = delegate(double v) { return FormatPesos(Math.Pow(10.0, v)); };
            DrawAxes(g, plot, lo, hi, lo, hi, pesos, pesos);

            using (Pen ideal = new Pen(Color.Firebrick, 1.5f))
            {
                ideal.DashStyle = DashStyle.Dash;
                g.DrawLine(ideal, Map(plot, lo, lo, lo, hi, lo, hi), Map(plot, hi, hi, lo, hi, lo, hi));
            }
            using (Brush dot = new SolidBrush(Color.FromArgb(110, 31, 119, 180)))
            {
                for (int i = 0; i < actual.Length; i++)
                {
                    PointF p = Map(plot, Math.Log10(actual[i]), Math.Log10(predicted[i]), lo, hi, lo, hi);
                    g.FillEllipse(dot, p.X - 2.5f, p.Y - 2.5f, 5f, 5f);
                }
            }
            using (Brush text = new SolidBrush(ForeColor))
            {
                g.DrawString("x: actual   y: predicted", Font, text, plot.Left + 6, plot.Top + 4);
            }
        }
    }
}
