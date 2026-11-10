#!/usr/bin/env python3

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

METRIC_GROUPS: Dict[str, Tuple[str, ...]] = {
    "Return": ("BenchmarkDret", "DailyReturn", "AnnualReturn", "CumReturn"),
    "Drawdown": ("MaxDrawdown", "drawDown", "maxDrawdown", "drawDownLength", "maxDrawdownLength"),
    "Risk Ratio": ("Calmar", "SharpeRatio", "SQN"),
    "Portfolio Value": ("Portfolio", "Cash"),
    "PnL": ("NetPnL", "Pnl"),
    "Trades": ("OrdersCnt", "TradesCnt", "DailyTradesCnt"),
    "Win Rate": ("DailyWinRate", "TotalWinRate"),
    "Period Stats": ("PeriodStats AvgRet", "PeriodStats RetStd", "PeriodStats PosCnt", "PeriodStats NegCnt"),
    "Holding": ("DailyAvgHold",),
}


def _categorize(metrics: Iterable[str]) -> Dict[str, List[str]]:
    sets = {k: set(v) for k, v in METRIC_GROUPS.items()}
    result: Dict[str, List[str]] = {k: [] for k in METRIC_GROUPS}
    result["Other"] = []
    for m in metrics:
        placed = False
        for grp, keys in sets.items():
            if m in keys:
                result[grp].append(m)
                placed = True
                break
        if not placed:
            result["Other"].append(m)
    for v in result.values():
        v.sort()
    return result


class ParquetToPDFConverter:
    """Convert a long-format bt_core log parquet file into a multi-page PDF report."""

    def __init__(self, parquet_path: str, pdf_path: Optional[str] = None):
        self.parquet_path = parquet_path
        if pdf_path is None:
            base, _ = os.path.splitext(parquet_path)
            pdf_path = base + ".pdf"
        self.pdf_path = pdf_path
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        df = pd.read_parquet(self.parquet_path)
        if "metrics" in df.columns:
            df["metrics"] = df["metrics"].apply(
                lambda b: b.decode("utf-8", errors="ignore").rstrip("\x00")
                if isinstance(b, (bytes, bytearray)) else str(b)
            )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
        df = df[df["datetime"] != pd.Timestamp(0, unit="s", tz="UTC")].copy()
        df = df.sort_values(["datetime", "metrics"]).reset_index(drop=True)
        self._df = df
        return df

    def _render_cover(self, pdf: PdfPages, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.axis("off")

        ts_min, ts_max = df["datetime"].min(), df["datetime"].max()
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
        cols = 3
        rows_per_col = (n_metrics + cols - 1) // cols
        for i in range(rows_per_col):
            row = [f"  {metrics[i + c * rows_per_col]}" if (i + c * rows_per_col) < n_metrics else "" for c in range(cols)]
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
            else:
                fig.text(0.08, y, text, fontsize=9, family="monospace", color="#333333")
                y -= 0.025

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _render_summary(self, pdf: PdfPages, wide: pd.DataFrame) -> None:
        stats_rows = []
        for c in wide.columns:
            s = wide[c].astype(float)
            cnt = int(s.count())
            stats_rows.append({
                "Metric": c,
                "Count": cnt,
                "Min": float(np.nanmin(s)) if cnt else float("nan"),
                "Max": float(np.nanmax(s)) if cnt else float("nan"),
                "Mean": float(np.nanmean(s)) if cnt else float("nan"),
                "Std": float(np.nanstd(s)) if cnt else float("nan"),
                "Last": float(s.iloc[-1]) if cnt else float("nan"),
            })
        stats_df = pd.DataFrame(stats_rows)

        n = len(stats_df)
        per_page = 18
        pages = max(1, (n + per_page - 1) // per_page)
        for p in range(pages):
            chunk = stats_df.iloc[p * per_page : (p + 1) * per_page]
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(f"Metric Summary (page {p + 1}/{pages})", fontsize=16, fontweight="bold", color="#1a3a5c", pad=14)
            tbl = ax.table(cellText=chunk.round(4).values, colLabels=stats_df.columns, cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.4)
            for j in range(len(stats_df.columns)):
                cell = tbl[(0, j)]
                cell.set_facecolor("#2c5f8a")
                cell.set_text_props(color="white", fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def _format_date_axis(ax, index: pd.Series) -> None:
        ts = pd.to_datetime(index)
        tmin, tmax = ts.min(), ts.max()
        ax.set_xlim(tmin, tmax)
        span_days = (tmax - tmin).days if hasattr(tmax - tmin, "days") else 0
        if span_days <= 1:
            loc, fmt = mdates.HourLocator(interval=1), mdates.DateFormatter("%Y-%m-%d %H:%M")
        elif span_days <= 31:
            loc, fmt = mdates.DayLocator(), mdates.DateFormatter("%Y-%m-%d")
        elif span_days <= 180:
            loc, fmt = mdates.WeekdayLocator(byweekday=mdates.MO), mdates.DateFormatter("%Y-%m-%d")
        elif span_days <= 730:
            loc, fmt = mdates.MonthLocator(), mdates.DateFormatter("%Y-%m")
        else:
            loc, fmt = mdates.YearLocator(), mdates.DateFormatter("%Y")
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)

    def _render_group_chart(self, pdf: PdfPages, title: str, index: pd.Series, data: Dict[str, pd.Series]) -> None:
        if not data:
            return
        fig, ax = plt.subplots(figsize=(11.69, 5.5))
        for name, series in data.items():
            if not series.dropna().empty:
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
        if series.dropna().empty:
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

    def convert(self) -> str:
        df = self._load()
        wide = df.pivot_table(index="datetime", columns="metrics", values="value", aggfunc="last").sort_index()
        index = wide.index

        with PdfPages(self.pdf_path) as pdf:
            self._render_cover(pdf, df)
            self._render_summary(pdf, wide)
            for group_name, members in _categorize(wide.columns.tolist()).items():
                present = [m for m in members if m in wide.columns]
                if not present:
                    continue
                if len(present) == 1:
                    self._render_single_metric(pdf, f"{group_name} — {present[0]}", index, wide[present[0]])
                else:
                    self._render_group_chart(pdf, group_name, index, {m: wide[m] for m in present})

            d = pdf.infodict()
            d["Title"] = "bt_core Backtest Report"
            d["Author"] = "bt_studio.visual.to_pdf"
            d["Subject"] = "Parquet log report"
            d["CreationDate"] = datetime.now(timezone.utc)
            d["ModDate"] = datetime.now(timezone.utc)

        return self.pdf_path


def convert_parquet_to_pdf(parquet_path: str, pdf_path: Optional[str] = None) -> str:
    """Convert one bt_core parquet log file to a multi-page PDF report."""
    return ParquetToPDFConverter(parquet_path, pdf_path).convert()
