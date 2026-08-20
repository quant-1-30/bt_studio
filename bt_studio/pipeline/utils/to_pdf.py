#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless servers
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Metric grouping: decides which metrics share a chart.
# Ordered dict so iteration order is deterministic across Python versions.
# ---------------------------------------------------------------------------
METRIC_GROUPS: Dict[str, Tuple[str, ...]] = {
    "Return": ("BenchmarkDret", "DailyReturn", "AnnualReturn", "CumReturn"),
    "Drawdown": (
        "MaxDrawdown", "drawDown", "maxDrawdown",
        "drawDownLength", "maxDrawdownLength",
    ),
    "Risk Ratio": ("Calmar", "SharpeRatio", "SQN"),
    "Portfolio Value": ("Portfolio", "Cash"),
    "PnL": ("NetPnL", "Pnl"),
    "Trades": ("OrdersCnt", "TradesCnt", "DailyTradesCnt"),
    "Win Rate": ("DailyWinRate", "TotalWinRate"),
    "Period Stats": (
        "PeriodStats AvgRet", "PeriodStats RetStd",
        "PeriodStats PosCnt", "PeriodStats NegCnt",
    ),
    "Holding": ("DailyAvgHold",),
}


def _categorize(metrics: Iterable[str]) -> Dict[str, List[str]]:
    """Map each metric to its group; unmapped go into ``Other``."""
    sets = {k: set(v) for k, v in METRIC_GROUPS.items()}
    result: Dict[str, List[str]] = {k: [] for k in METRIC_GROUPS}
    result["Other"] = []
    other_keys = set()
    for grp, keys in sets.items():
        other_keys |= keys
    for m in metrics:
        placed = False
        for grp, keys in sets.items():
            if m in keys:
                result[grp].append(m)
                placed = True
                break
        if not placed:
            result["Other"].append(m)
    # deterministic ordering
    for v in result.values():
        v.sort()
    return result


class ParquetToPDFConverter:
    """
    Convert a bt_core log parquet file into a multi-page PDF report

    Parquet -> PDF report converter for bt_core log files.

    Reads a parquet file produced by ``LogConsumerThread`` (long-format: each row
    is a ``(datetime, value, metric_name)`` triple) and renders a multi-page PDF
    report with cover, summary table and grouped time-series charts.

    Usage (library)::

        from plugins.to_pdf import convert_parquet_to_pdf
        convert_parquet_to_pdf("logs/log_cerebro_0.parquet", "logs/report.pdf")

    Usage (CLI)::

        python -m plugins.to_pdf.converter logs/log_cerebro_0.parquet logs/report.pdf
    """


    def __init__(self, parquet_path: str, pdf_path: Optional[str] = None):
        self.parquet_path = parquet_path
        if pdf_path is None:
            base, _ = os.path.splitext(parquet_path)
            pdf_path = base + ".pdf"
        self.pdf_path = pdf_path

        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ load
    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        # pyarrow reads the binary column as bytes; decode to str
        df = pd.read_parquet(self.parquet_path)
        if "metrics" in df.columns:
            df["metrics"] = df["metrics"].apply(
                lambda b: b.decode("utf-8", errors="ignore").rstrip("\x00")
                if isinstance(b, (bytes, bytearray)) else str(b)
            )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
        # Drop sentinel / uninitialized rows whose timestamp is the Unix epoch
        # (0). They are produced when the SHM channel publishes a sentinel
        # before the first real tick and would otherwise stretch the x-axis
        # back to 1970.
        df = df[df["datetime"] != pd.Timestamp(0, unit="s", tz="UTC")].copy()
        df = df.sort_values(["datetime", "metrics"]).reset_index(drop=True)
        self._df = df
        return df

    # ------------------------------------------------------------------ pages
    def _render_cover(self, pdf: PdfPages, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.axis("off")

        ts_min = df["datetime"].min()
        ts_max = df["datetime"].max()
        metrics = sorted(df["metrics"].unique())
        n_metrics = len(metrics)

        lines = [
            ("bt_core Backtest Report", "title"),
            ("", "blank"),
            (f"Source: {os.path.basename(self.parquet_path)}", "meta"),
            (f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", "meta"),
            ("", "blank"),
            ("Data Overview", "h2"),
            (f"  Total rows:        {len(df):,}", "body"),
            (f"  Unique metrics:    {n_metrics}", "body"),
            (f"  Time range:        {ts_min:%Y-%m-%d %H:%M}  ->  {ts_max:%Y-%m-%d %H:%M}", "body"),
            (f"  Unique timestamps: {df['datetime'].nunique():,}", "body"),
            ("", "blank"),
            ("Metrics Included", "h2"),
        ]
        # column layout for metric list
        cols = 3
        rows_per_col = (n_metrics + cols - 1) // cols
        for i in range(rows_per_col):
            row = []
            for c in range(cols):
                idx = i + c * rows_per_col
                row.append(f"  {metrics[idx]}" if idx < n_metrics else "")
            lines.append(("   |   ".join(row), "body"))

        y = 0.95
        for text, style in lines:
            if style == "title":
                fig.text(0.5, y, text, ha="center", va="top", fontsize=22, fontweight="bold", color="#1a3a5c")
                y -= 0.08
            elif style == "h2":
                fig.text(0.08, y, text, fontsize=14, fontweight="bold", color="#2c5f8a")
                y -= 0.04
            elif style == "blank":
                y -= 0.03
            elif style == "meta":
                fig.text(0.5, y, text, ha="center", va="top", fontsize=10, color="#555555")
                y -= 0.035
            else:  # body
                fig.text(0.08, y, text, fontsize=9, family="monospace", color="#333333")
                y -= 0.025

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _render_summary(self, pdf: PdfPages, wide: pd.DataFrame) -> None:
        cols = list(wide.columns)
        stats_rows = []
        for c in cols:
            s = wide[c].astype(float)
            stats_rows.append({
                "Metric": c,
                "Count": int(s.count()),
                "Min": float(np.nanmin(s)) if s.count() else float("nan"),
                "Max": float(np.nanmax(s)) if s.count() else float("nan"),
                "Mean": float(np.nanmean(s)) if s.count() else float("nan"),
                "Std": float(np.nanstd(s)) if s.count() else float("nan"),
                "Last": float(s.iloc[-1]) if s.count() else float("nan"),
            })
        stats_df = pd.DataFrame(stats_rows)

        n = len(stats_df)
        per_page = 18
        pages = (n + per_page - 1) // per_page
        for p in range(pages):
            chunk = stats_df.iloc[p * per_page:(p + 1) * per_page]
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(
                f"Metric Summary (page {p + 1}/{pages})",
                fontsize=16, fontweight="bold", color="#1a3a5c", pad=14,
            )
            tbl = ax.table(
                cellText=chunk.round(4).values,
                colLabels=stats_df.columns,
                cellLoc="center",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.4)
            # header styling
            for j in range(len(stats_df.columns)):
                cell = tbl[(0, j)]
                cell.set_facecolor("#2c5f8a")
                cell.set_text_props(color="white", fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def _format_date_axis(ax, index: pd.Series) -> None:
        """Anchor the x-axis to the parquet's actual time range and pick a
        locator/formatter appropriate for the span (avoids matplotlib falling
        back to the 1970 epoch when the series is sparse / has NaNs)."""
        ts = pd.to_datetime(index)
        tmin = ts.min()
        tmax = ts.max()
        ax.set_xlim(tmin, tmax)

        span_days = (tmax - tmin).days if hasattr(tmax - tmin, "days") else 0
        if span_days <= 1:
            loc = mdates.HourLocator(interval=1)
            fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")
        elif span_days <= 31:
            loc = mdates.DayLocator()
            fmt = mdates.DateFormatter("%Y-%m-%d")
        elif span_days <= 180:
            loc = mdates.WeekdayLocator(byweekday=mdates.MO)
            fmt = mdates.DateFormatter("%Y-%m-%d")
        elif span_days <= 730:
            loc = mdates.MonthLocator()
            fmt = mdates.DateFormatter("%Y-%m")
        else:
            loc = mdates.YearLocator()
            fmt = mdates.DateFormatter("%Y")
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)

    def _render_group_chart(
        self,
        pdf: PdfPages,
        title: str,
        index: pd.Series,
        data: Dict[str, pd.Series],
    ) -> None:
        if not data:
            return
        fig, ax = plt.subplots(figsize=(11.69, 5.5))
        for name, series in data.items():
            valid = series.dropna()
            if valid.empty:
                continue
            ax.plot(index.values, series.values, label=name, linewidth=0.9, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1a3a5c")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", fontsize=8, framealpha=0.8)
        self._format_date_axis(ax, index)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _render_single_metric(self, pdf: PdfPages, name: str, index: pd.Series, series: pd.Series) -> None:
        valid = series.dropna()
        if valid.empty:
            return
        fig, ax = plt.subplots(figsize=(11.69, 4.8))
        ax.plot(index.values, series.values, color="#2c5f8a", linewidth=0.9)
        ax.fill_between(index.values, series.values, alpha=0.12, color="#2c5f8a")
        ax.set_title(name, fontsize=13, fontweight="bold", color="#1a3a5c")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.35)
        self._format_date_axis(ax, index)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ main
    def convert(self) -> str:
        """Run the full conversion; returns the output PDF path."""
        df = self._load()

        # pivot to wide format: index=datetime, columns=metric name
        wide = df.pivot_table(index="datetime", columns="metrics", values="value", aggfunc="last")
        wide = wide.sort_index()
        index = wide.index

        with PdfPages(self.pdf_path) as pdf:
            self._render_cover(pdf, df)
            self._render_summary(pdf, wide)

            groups = _categorize(wide.columns.tolist())
            for group_name, members in groups.items():
                if not members:
                    continue
                # present members only
                present = [m for m in members if m in wide.columns]
                if not present:
                    continue
                if len(present) == 1:
                    m = present[0]
                    self._render_single_metric(pdf, f"{group_name} — {m}", index, wide[m])
                else:
                    data = {m: wide[m] for m in present}
                    self._render_group_chart(pdf, group_name, index, data)

            # metadata footer
            d = pdf.infodict()
            d["Title"] = "bt_core Backtest Report"
            d["Author"] = "plugins.to_pdf"
            d["Subject"] = "Parquet log report"
            d["CreationDate"] = datetime.now(timezone.utc)
            d["ModDate"] = datetime.now(timezone.utc)

        return self.pdf_path


def convert_parquet_to_pdf(parquet_path: str, pdf_path: Optional[str] = None) -> str:
    """Convenience wrapper: convert one parquet file to PDF."""
    return ParquetToPDFConverter(parquet_path, pdf_path).convert()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _cli(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m plugins.to_pdf.converter <input.parquet> [output.pdf]")
        return 1
    src = argv[0]
    dst = argv[1] if len(argv) > 1 else None
    if not os.path.isfile(src):
        print(f"ERROR: input file not found: {src}")
        return 2
    out = convert_parquet_to_pdf(src, dst)
    print(f"PDF report generated: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())