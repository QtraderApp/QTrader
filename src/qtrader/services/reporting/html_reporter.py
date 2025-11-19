"""HTML Report Generator for Backtest Results.

Generates a standalone, interactive HTML report with embedded charts
that users can open directly in their browser without any dependencies.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HTMLReportGenerator:
    """Generates standalone HTML reports for backtest results."""

    def __init__(self, output_dir: Path):
        """
        Initialize HTML report generator.

        Args:
            output_dir: Path to the backtest output directory containing:
                - performance.json
                - run_manifest.json (optional)
                - timeseries/*.parquet files
                - events.parquet
        """
        self.output_dir = Path(output_dir)
        self.timeseries_dir = self.output_dir / "timeseries"

    def generate(self) -> Path:
        """
        Generate standalone HTML report.

        Returns:
            Path to generated report.html file

        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If data is invalid or corrupted
        """
        # Load data
        performance = self._load_performance()
        manifest = self._load_manifest()
        equity_curve = self._load_timeseries("equity_curve.parquet")
        returns = self._load_timeseries("returns.parquet")
        drawdowns = self._load_timeseries("drawdowns.parquet")
        trades = self._load_trades()

        # Generate HTML sections
        html = self._build_html(
            performance=performance,
            manifest=manifest,
            equity_curve=equity_curve,
            returns=returns,
            drawdowns=drawdowns,
            trades=trades,
        )

        # Write to file
        report_path = self.output_dir / "report.html"
        report_path.write_text(html, encoding="utf-8")

        return report_path

    def _load_performance(self) -> dict[str, Any]:
        """Load performance.json."""
        perf_path = self.output_dir / "performance.json"
        if not perf_path.exists():
            raise FileNotFoundError(f"performance.json not found: {perf_path}")
        data: dict[str, Any] = json.loads(perf_path.read_text())
        return data

    def _load_manifest(self) -> dict[str, Any] | None:
        """Load run_manifest.json (optional)."""
        manifest_path = self.output_dir / "run_manifest.json"
        if manifest_path.exists():
            data: dict[str, Any] = json.loads(manifest_path.read_text())
            return data
        return None

    def _load_timeseries(self, filename: str) -> pd.DataFrame | None:
        """Load a timeseries parquet file (returns None if not found)."""
        filepath = self.timeseries_dir / filename
        if filepath.exists():
            df = pd.read_parquet(filepath)

            # Handle duplicate timestamps by keeping the last value for each timestamp
            # This can occur when multiple portfolio state events are emitted at the same time
            if "timestamp" in df.columns and df["timestamp"].duplicated().any():
                df = df.drop_duplicates(subset=["timestamp"], keep="last")

            return df
        return None

    def _load_trades(self) -> list[dict[str, Any]]:
        """Load trade events from events.parquet."""
        import json

        events_path = self.output_dir / "events.parquet"
        if not events_path.exists():
            return []

        try:
            events_df = pd.read_parquet(events_path)
            trade_events = events_df[events_df["event_type"] == "trade"]

            trades = []
            for _, row in trade_events.iterrows():
                payload = json.loads(row["payload"])
                trades.append(payload)

            return trades
        except Exception:
            return []

    def _build_html(
        self,
        performance: dict[str, Any],
        manifest: dict[str, Any] | None,
        equity_curve: pd.DataFrame | None,
        returns: pd.DataFrame | None,
        drawdowns: pd.DataFrame | None,
        trades: list[dict[str, Any]] | None,
    ) -> str:
        """Build complete HTML document."""
        # Generate charts
        combined_chart = self._create_combined_chart(equity_curve, drawdowns)

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {performance.get("backtest_id", "Unknown")}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f7;
            color: #1d1d1f;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        header h1 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        header p {{
            opacity: 0.9;
            font-size: 0.95rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 4px solid #667eea;
        }}
        .metric-card.positive {{
            border-left-color: #10b981;
        }}
        .metric-card.negative {{
            border-left-color: #ef4444;
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}
        .metric-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: #1d1d1f;
        }}
        .chart-container {{
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }}
        .chart-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1d1d1f;
        }}
        table {{
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        th {{
            background: #f9fafb;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
        }}
        td {{
            padding: 1rem;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background: #f9fafb;
        }}
        .heatmap-table td {{
            text-align: center;
            padding: 0.75rem;
            font-size: 0.875rem;
        }}
        .heatmap-table th {{
            text-align: center;
            padding: 0.75rem;
            font-size: 0.875rem;
        }}
        .heatmap-table td.positive {{
            background: #d1fae5;
            color: #065f46;
        }}
        .heatmap-table td.negative {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .heatmap-table td.neutral {{
            background: #f9fafb;
            color: #6b7280;
        }}
        .heatmap-table tr:hover td {{
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.875rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge.success {{
            background: #d1fae5;
            color: #065f46;
        }}
        .badge.warning {{
            background: #fef3c7;
            color: #92400e;
        }}
        .badge.danger {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .info-section {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .info-section h3 {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #374151;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}
        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #f3f4f6;
        }}
        .info-item:last-child {{
            border-bottom: none;
        }}
        .info-label {{
            color: #6b7280;
            font-size: 0.875rem;
        }}
        .info-value {{
            font-weight: 600;
            color: #1d1d1f;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 {performance.get("backtest_id", "Backtest Report")}</h1>
            <p>{performance.get("start_date")} to {performance.get("end_date")} ({performance.get("duration_days")} days)</p>
            {self._build_manifest_badges(manifest)}
        </header>

        {self._build_key_metrics(performance)}

        {self._build_run_info(performance, manifest)}

        {self._build_price_chart()}

        <div class="chart-container">
            <div class="chart-title">📈 Portfolio Performance</div>
            {combined_chart}
        </div>

        {self._build_trades_table(trades, equity_curve)}

        {self._build_performance_table(performance)}

        {self._build_monthly_breakdown_table(performance)}

        <div class="footer">
            <p>Generated by QTrader • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p style="margin-top: 0.5rem;">Raw data available in: {self.output_dir.name}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _build_manifest_badges(self, manifest: dict[str, Any] | None) -> str:
        """Build status badges from manifest."""
        if not manifest:
            return ""

        badges = []
        if git_info := manifest.get("git"):
            if commit := git_info.get("commit_hash"):
                badges.append(f'<span class="badge success">Git: {commit[:7]}</span>')
            if git_info.get("has_uncommitted_changes"):
                badges.append('<span class="badge warning">Uncommitted Changes</span>')

        if badges:
            return f'<p style="margin-top: 1rem;">{" ".join(badges)}</p>'
        return ""

    def _build_key_metrics(self, performance: dict[str, Any]) -> str:
        """Build key metrics cards."""
        # Determine metric card classes based on values
        total_return = float(performance.get("total_return_pct", 0))
        sharpe = float(performance.get("sharpe_ratio", 0))
        max_dd = float(performance.get("max_drawdown_pct", 0))

        return_class = "positive" if total_return > 0 else "negative" if total_return < 0 else ""
        sharpe_class = "positive" if sharpe > 1 else "negative" if sharpe < 0 else ""
        dd_class = "negative" if abs(max_dd) > 10 else ""

        return f"""
        <div class="metrics-grid">
            <div class="metric-card {return_class}">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{performance.get("total_return_pct", "0")}%</div>
            </div>
            <div class="metric-card {return_class}">
                <div class="metric-label">CAGR</div>
                <div class="metric-value">{performance.get("cagr", "0")}%</div>
            </div>
            <div class="metric-card {sharpe_class}">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{performance.get("sharpe_ratio", "0")}</div>
            </div>
            <div class="metric-card {dd_class}">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{performance.get("max_drawdown_pct", "0")}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volatility (Annual)</div>
                <div class="metric-value">{performance.get("volatility_annual_pct", "0")}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{performance.get("total_trades", 0)}</div>
            </div>
        </div>
        """

    def _build_run_info(self, performance: dict[str, Any], manifest: dict[str, Any] | None) -> str:
        """Build run information section."""
        info_items = []

        # Basic info with currency formatting
        initial_equity = float(performance.get("initial_equity", 0))
        final_equity = float(performance.get("final_equity", 0))

        info_items.append(
            f'<div class="info-item"><span class="info-label">Initial Equity</span><span class="info-value">${initial_equity:,.2f}</span></div>'
        )
        info_items.append(
            f'<div class="info-item"><span class="info-label">Final Equity</span><span class="info-value">${final_equity:,.2f}</span></div>'
        )
        info_items.append(
            f'<div class="info-item"><span class="info-label">Duration</span><span class="info-value">{performance.get("duration_days", 0)} days</span></div>'
        )

        # Manifest info
        if manifest:
            if timestamp := manifest.get("timestamp"):
                info_items.append(
                    f'<div class="info-item"><span class="info-label">Run Time</span><span class="info-value">{timestamp}</span></div>'
                )
            if git_info := manifest.get("git"):
                if branch := git_info.get("branch"):
                    info_items.append(
                        f'<div class="info-item"><span class="info-label">Git Branch</span><span class="info-value">{branch}</span></div>'
                    )

        return f"""
        <div class="info-section">
            <h3>Run Information</h3>
            <div class="info-grid">
                {"".join(info_items)}
            </div>
        </div>
        """

    def _create_combined_chart(self, equity_curve: pd.DataFrame | None, drawdowns: pd.DataFrame | None) -> str:
        """Create combined equity curve and drawdown chart."""
        if equity_curve is None:
            return "<p>Equity curve data not available</p>"

        # Reset index after deduplication to avoid gaps
        equity_curve = equity_curve.reset_index(drop=True)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=["Equity Curve", "Drawdown"],
            vertical_spacing=0.1,
            shared_xaxes=True,
        )

        # Equity curve - convert to lists to avoid pandas index issues
        fig.add_trace(
            go.Scatter(
                x=equity_curve["timestamp"].tolist(),
                y=equity_curve["equity"].tolist(),
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#667eea", width=2),
                fill="tozeroy",
                fillcolor="rgba(102, 126, 234, 0.1)",
            ),
            row=1,
            col=1,
        )

        # Drawdown - use drawdown_pct from equity_curve (not drawdowns DataFrame)
        if "drawdown_pct" in equity_curve.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity_curve["timestamp"].tolist(),
                    y=equity_curve["drawdown_pct"].tolist(),
                    mode="lines",
                    name="Drawdown %",
                    line=dict(color="#ef4444", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(239, 68, 68, 0.2)",
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=50, r=50, t=50, b=50),
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

        # Format axes
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        html: str = fig.to_html(include_plotlyjs=False, div_id="combined-chart")
        return html

    def _create_equity_chart(self, equity_curve: pd.DataFrame) -> str:
        """Create equity curve chart."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=equity_curve["timestamp"],
                y=equity_curve["equity"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#667eea", width=2),
                fill="tozeroy",
                fillcolor="rgba(102, 126, 234, 0.1)",
            )
        )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

        html: str = fig.to_html(include_plotlyjs=False, div_id="equity-chart")
        return html

    def _create_drawdown_chart(self, drawdowns: pd.DataFrame) -> str:
        """Create drawdown chart."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=drawdowns["timestamp"],
                y=drawdowns["drawdown_pct"],
                mode="lines",
                name="Drawdown",
                line=dict(color="#ef4444", width=2),
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.2)",
            )
        )

        fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

        html: str = fig.to_html(include_plotlyjs=False, div_id="drawdown-chart")
        return html

    def _create_monthly_returns_chart(self, performance: dict[str, Any]) -> str:
        """Create monthly returns bar chart."""
        monthly_returns = performance.get("monthly_returns", [])
        if not monthly_returns:
            return ""

        periods = [m["period"] for m in monthly_returns]
        returns = [float(m["return_pct"]) for m in monthly_returns]

        colors = ["#10b981" if r >= 0 else "#ef4444" for r in returns]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=periods,
                y=returns,
                marker_color=colors,
                name="Monthly Return",
                hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            height=350,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="#f0f0f0", zeroline=True, zerolinewidth=2, zerolinecolor="#9ca3af"
        )

        return f"""
        <div class="chart-container">
            <div class="chart-title">📅 Monthly Performance</div>
            {fig.to_html(include_plotlyjs=False, div_id="monthly-chart")}
        </div>
        """

    def _build_performance_table(self, performance: dict[str, Any]) -> str:
        """Build comprehensive performance metrics table with multi-column layout."""

        # Format returns with proper precision
        best_day = float(performance.get("best_day_return_pct", 0))
        worst_day = float(performance.get("worst_day_return_pct", 0))

        metrics = [
            (
                "Returns",
                [
                    ("Total Return", f"{performance.get('total_return_pct', '0')}%"),
                    ("CAGR", f"{performance.get('cagr', '0')}%"),
                    ("Best Day", f"{best_day:.2f}%"),
                    ("Worst Day", f"{worst_day:.2f}%"),
                ],
            ),
            (
                "Risk Metrics",
                [
                    ("Volatility (Annual)", f"{performance.get('volatility_annual_pct', '0')}%"),
                    ("Max Drawdown", f"{performance.get('max_drawdown_pct', '0')}%"),
                    ("Max DD Duration", f"{performance.get('max_drawdown_duration_days', 0)} days"),
                    ("Avg Drawdown", f"{performance.get('avg_drawdown_pct', '0')}%"),
                    ("Current Drawdown", f"{performance.get('current_drawdown_pct', '0')}%"),
                ],
            ),
            (
                "Risk-Adjusted Returns",
                [
                    ("Sharpe Ratio", performance.get("sharpe_ratio", "0")),
                    ("Sortino Ratio", performance.get("sortino_ratio", "0")),
                    ("Calmar Ratio", performance.get("calmar_ratio", "0")),
                    ("Risk-Free Rate", f"{float(performance.get('risk_free_rate', 0)) * 100:.2f}%"),
                ],
            ),
            (
                "Trade Statistics",
                [
                    ("Total Trades", performance.get("total_trades", 0)),
                    ("Winning Trades", performance.get("winning_trades", 0)),
                    ("Losing Trades", performance.get("losing_trades", 0)),
                    ("Win Rate", f"{performance.get('win_rate', '0')}%"),
                    (
                        "Profit Factor",
                        performance.get("profit_factor", "N/A") if performance.get("profit_factor") else "N/A",
                    ),
                    ("Avg Win", f"${performance.get('avg_win', '0')}"),
                    ("Avg Loss", f"${performance.get('avg_loss', '0')}"),
                    ("Expectancy", f"${performance.get('expectancy', '0')}"),
                ],
            ),
        ]

        # Build sections with 2-column grid layout
        sections_html = []
        for i in range(0, len(metrics), 2):
            section_pair = metrics[i : i + 2]
            pair_html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">'

            for section_title, items in section_pair:
                rows = "".join(
                    [f"<tr><td>{label}</td><td><strong>{value}</strong></td></tr>" for label, value in items]
                )
                pair_html += f"""
                    <div class="chart-container" style="margin-bottom: 0;">
                        <div class="chart-title">{section_title}</div>
                        <table>
                            <tbody>
                                {rows}
                            </tbody>
                        </table>
                    </div>
                """

            pair_html += "</div>"
            sections_html.append(pair_html)

        return "".join(sections_html)

    def _build_monthly_breakdown_table(self, performance: dict[str, Any]) -> str:
        """Build monthly returns heatmap table (years × months)."""
        monthly_returns = performance.get("monthly_returns", [])
        if not monthly_returns:
            return ""

        # Organize data by year and month
        from collections import defaultdict

        data_by_year: dict[str, dict[str, float]] = defaultdict(dict)

        for period in monthly_returns:
            # Parse period like "2020-03" into year and month
            year, month = period["period"].split("-")
            return_pct = float(period["return_pct"])
            data_by_year[year][month] = return_pct

        # Build table
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # Header row
        header = "<tr><th>Year</th>" + "".join([f"<th>{name}</th>" for name in month_names]) + "<th>YTD</th></tr>"

        # Data rows
        rows = []
        for year in sorted(data_by_year.keys()):
            row_cells = [f"<td><strong>{year}</strong></td>"]
            ytd_return = 1.0  # Multiplicative YTD

            for month in months:
                if month in data_by_year[year]:
                    ret = data_by_year[year][month]
                    ytd_return *= 1 + ret / 100

                    # Color coding
                    if ret > 0:
                        cell_class = "positive"
                    elif ret < 0:
                        cell_class = "negative"
                    else:
                        cell_class = "neutral"

                    row_cells.append(f'<td class="{cell_class}"><strong>{ret:.2f}%</strong></td>')
                else:
                    row_cells.append('<td class="neutral">—</td>')

            # YTD column
            ytd_pct = (ytd_return - 1) * 100
            ytd_class = "positive" if ytd_pct > 0 else "negative" if ytd_pct < 0 else "neutral"
            row_cells.append(f'<td class="{ytd_class}"><strong>{ytd_pct:.2f}%</strong></td>')

            rows.append("<tr>" + "".join(row_cells) + "</tr>")

        return f"""
        <div class="chart-container">
            <div class="chart-title">📊 Monthly Returns Heatmap</div>
            <table class="heatmap-table">
                <thead>
                    {header}
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _build_price_chart(self) -> str:
        """Build instrument price chart from bar events."""
        import json

        events_path = self.output_dir / "events.parquet"
        if not events_path.exists():
            return ""

        try:
            events_df = pd.read_parquet(events_path)
            bar_events = events_df[events_df["event_type"] == "bar"]

            if len(bar_events) == 0:
                return ""

            # Extract price data by symbol
            price_data: dict[str, list[dict[str, Any]]] = {}
            for _, row in bar_events.iterrows():
                payload = json.loads(row["payload"])
                symbol = payload.get("symbol", "")
                if symbol not in price_data:
                    price_data[symbol] = []
                price_data[symbol].append(
                    {
                        "timestamp": payload.get("timestamp", ""),
                        "close": float(payload.get("close", 0)),
                    }
                )

            if not price_data:
                return ""

            num_symbols = len(price_data)

            # Define color palette for multiple symbols
            colors = [
                "#667eea",
                "#764ba2",
                "#f093fb",
                "#4facfe",
                "#43e97b",
                "#fa709a",
                "#fee140",
                "#30cfd0",
            ]

            # For multiple symbols with very different price scales, use subplots
            # Check if price ranges differ significantly
            price_ranges = {}
            for symbol, data in price_data.items():
                prices = [d["close"] for d in data]
                price_ranges[symbol] = (min(prices), max(prices))

            # Calculate if we need separate subplots (price ranges differ by >10x)
            use_subplots = False
            if num_symbols > 1:
                max_range = max(r[1] for r in price_ranges.values())
                min_range = min(r[0] for r in price_ranges.values())
                if max_range / max(min_range, 0.01) > 10:
                    use_subplots = True

            if use_subplots and num_symbols > 1:
                # Create subplots for symbols with very different price scales
                fig = make_subplots(
                    rows=num_symbols,
                    cols=1,
                    subplot_titles=[symbol for symbol in sorted(price_data.keys())],
                    vertical_spacing=0.08,
                    shared_xaxes=True,
                )

                for idx, (symbol, data) in enumerate(sorted(price_data.items()), start=1):
                    data_sorted = sorted(data, key=lambda x: x["timestamp"])
                    timestamps = [d["timestamp"] for d in data_sorted]
                    prices = [d["close"] for d in data_sorted]

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=prices,
                            mode="lines",
                            name=symbol,
                            line=dict(width=2, color=colors[(idx - 1) % len(colors)]),
                            showlegend=False,
                        ),
                        row=idx,
                        col=1,
                    )

                    fig.update_yaxes(title_text="Price ($)", row=idx, col=1)

                fig.update_xaxes(title_text="Date", row=num_symbols, col=1)
                height = 250 * num_symbols
            else:
                # Single chart for one symbol or multiple symbols with similar scales
                fig = go.Figure()

                for idx, (symbol, data) in enumerate(sorted(price_data.items())):
                    data_sorted = sorted(data, key=lambda x: x["timestamp"])
                    timestamps = [d["timestamp"] for d in data_sorted]
                    prices = [d["close"] for d in data_sorted]

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=prices,
                            mode="lines",
                            name=symbol,
                            line=dict(width=2, color=colors[idx % len(colors)]),
                        )
                    )

                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Price ($)")
                height = 400

            # Update layout
            fig.update_layout(
                title="",
                hovermode="x unified",
                height=height,
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=num_symbols > 1 and not use_subplots,
                margin=dict(l=50, r=50, t=30 if not use_subplots else 50, b=50),
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

            html: str = fig.to_html(include_plotlyjs=False, div_id="price-chart")

            symbol_text = ", ".join(sorted(price_data.keys()))
            symbol_count_text = f"{num_symbols} Asset{'s' if num_symbols > 1 else ''}"
            return f"""
            <div class="chart-container">
                <div class="chart-title">📈 Instrument Prices · {symbol_count_text} ({symbol_text})</div>
                {html}
            </div>
            """
        except Exception:
            return ""

    def _build_trades_table(self, trades: list[dict[str, Any]] | None, equity_curve: pd.DataFrame | None) -> str:
        """Build trades table showing both closed and open trades."""
        if not trades:
            return ""

        # Sort trades by entry timestamp
        sorted_trades = sorted(trades, key=lambda t: t.get("entry_timestamp", ""))

        rows = []
        for trade in sorted_trades:
            trade_id = trade.get("trade_id", "")
            symbol = trade.get("symbol", "")
            strategy = trade.get("strategy_id", "")
            side = trade.get("side", "")
            status = trade.get("status", "")
            entry_price = float(trade.get("entry_price", 0))
            exit_price = trade.get("exit_price")
            quantity = int(trade.get("current_quantity", 0))
            commission = float(trade.get("commission_total", 0))
            entry_time = trade.get("entry_timestamp", "")[:10] if trade.get("entry_timestamp") else ""
            exit_time = trade.get("exit_timestamp", "")[:10] if trade.get("exit_timestamp") else ""

            # Format entry/exit prices
            entry_str = f"${entry_price:.2f}"
            exit_str = f"${float(exit_price):.2f}" if exit_price else "—"

            # Calculate PnL
            if status == "closed" and exit_price:
                exit_price_val = float(exit_price)
                if side == "long":
                    pnl = quantity * (exit_price_val - entry_price) - commission
                else:
                    pnl = quantity * (entry_price - exit_price_val) - commission
                pnl_pct = (pnl / (entry_price * quantity)) * 100 if quantity > 0 else 0
                pnl_str = f"${pnl:,.2f}"
                pnl_pct_str = f"({pnl_pct:+.2f}%)"
                pnl_class = "positive" if pnl >= 0 else "negative"
                status_badge = '<span class="badge success">Closed</span>'
            else:
                # Open trade - calculate unrealized PnL if we have current price
                if equity_curve is not None and len(equity_curve) > 0:
                    # Get last available price for this symbol from equity curve
                    # This is approximate; ideally we'd have current market price
                    current_price = entry_price  # Fallback

                    # Simple approximation: use equity change
                    if side == "long":
                        unrealized_pnl = quantity * (current_price - entry_price) - commission
                    else:
                        unrealized_pnl = quantity * (entry_price - current_price) - commission

                    unrealized_pct = (unrealized_pnl / (entry_price * quantity)) * 100 if quantity > 0 else 0
                    pnl_str = f"${unrealized_pnl:,.2f}"
                    pnl_pct_str = f"({unrealized_pct:+.2f}%)"
                    pnl_class = "positive" if unrealized_pnl >= 0 else "negative"
                else:
                    pnl_str = "—"
                    pnl_pct_str = ""
                    pnl_class = ""
                status_badge = '<span class="badge warning">Open</span>'

            row = f"""
            <tr>
                <td>{trade_id}</td>
                <td><strong>{symbol}</strong></td>
                <td>{strategy}</td>
                <td>{status_badge}</td>
                <td style="text-align: center;">{side.upper()}</td>
                <td>{entry_time}</td>
                <td style="text-align: right;">{entry_str}</td>
                <td>{exit_time}</td>
                <td style="text-align: right;">{exit_str}</td>
                <td style="text-align: right;">{quantity:,}</td>
                <td style="text-align: right;">${commission:.2f}</td>
                <td style="text-align: right;" class="{pnl_class}"><strong>{pnl_str}</strong> <small>{pnl_pct_str}</small></td>
            </tr>
            """
            rows.append(row)

        return f"""
        <div class="chart-container">
            <div class="chart-title">💼 Trades</div>
            <table>
                <thead>
                    <tr>
                        <th>Trade ID</th>
                        <th>Symbol</th>
                        <th>Strategy</th>
                        <th>Status</th>
                        <th style="text-align: center;">Side</th>
                        <th>Entry Date</th>
                        <th style="text-align: right;">Entry Price</th>
                        <th>Exit Date</th>
                        <th style="text-align: right;">Exit Price</th>
                        <th style="text-align: right;">Quantity</th>
                        <th style="text-align: right;">Commission</th>
                        <th style="text-align: right;">P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """
