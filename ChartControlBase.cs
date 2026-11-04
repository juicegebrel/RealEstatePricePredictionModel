using System;
using System.Drawing;
using System.Windows.Forms;

namespace RealEstatePricePredictionModel
{
    /// <summary>Shared GDI+ drawing helpers (frame, grid, tick labels) for the two simple charts.</summary>
    public abstract class ChartControlBase : Control
    {
        private const int MarginLeft = 70;
        private const int MarginRight = 16;
        private const int MarginTop = 30;
        private const int MarginBottom = 34;

        protected ChartControlBase()
        {
            DoubleBuffered = true;
            ResizeRedraw = true;
            BackColor = Color.White;
            ForeColor = Color.FromArgb(40, 40, 40);
        }

        protected Rectangle PlotArea
        {
            get
            {
                return new Rectangle(MarginLeft, MarginTop,
                    Math.Max(10, Width - MarginLeft - MarginRight),
                    Math.Max(10, Height - MarginTop - MarginBottom));
            }
        }

        protected static PointF Map(Rectangle plot, double x, double y, double xMin, double xMax, double yMin, double yMax)
        {
            float px = plot.Left + (float)((x - xMin) / (xMax - xMin) * plot.Width);
            float py = plot.Bottom - (float)((y - yMin) / (yMax - yMin) * plot.Height);
            return new PointF(px, py);
        }

        protected void DrawTitle(Graphics g, string title)
        {
            using (Font bold = new Font(Font, FontStyle.Bold))
            using (Brush brush = new SolidBrush(ForeColor))
            {
                g.DrawString(title, bold, brush, 6, 6);
            }
        }

        protected void DrawMessage(Graphics g, string message)
        {
            using (Brush brush = new SolidBrush(Color.Gray))
            {
                g.DrawString(message, Font, brush, MarginLeft + 10, MarginTop + 10);
            }
        }

        protected void DrawAxes(Graphics g, Rectangle plot, double xMin, double xMax, double yMin, double yMax,
                                Func<double, string> xFormat, Func<double, string> yFormat)
        {
            const int ticks = 4;
            using (Pen grid = new Pen(Color.FromArgb(225, 225, 225)))
            using (Pen border = new Pen(Color.FromArgb(150, 150, 150)))
            using (Brush text = new SolidBrush(ForeColor))
            using (StringFormat right = new StringFormat())
            using (StringFormat center = new StringFormat())
            {
                right.Alignment = StringAlignment.Far;
                center.Alignment = StringAlignment.Center;

                for (int i = 0; i <= ticks; i++)
                {
                    double fraction = (double)i / ticks;

                    float y = plot.Bottom - (float)(fraction * plot.Height);
                    g.DrawLine(grid, plot.Left, y, plot.Right, y);
                    g.DrawString(yFormat(yMin + fraction * (yMax - yMin)), Font, text, plot.Left - 4, y - 7, right);

                    float x = plot.Left + (float)(fraction * plot.Width);
                    g.DrawLine(grid, x, plot.Top, x, plot.Bottom);
                    g.DrawString(xFormat(xMin + fraction * (xMax - xMin)), Font, text, x, plot.Bottom + 4, center);
                }
                g.DrawRectangle(border, plot);
            }
        }
    }
}
