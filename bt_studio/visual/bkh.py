#!/usr/bin/env python3
"""Bokeh interactive backtest log visualizer with linked X-axis and synchronized crosshairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    CrosshairTool,
    CustomJS,
    Div,
    HoverTool,
    NumeralTickFormatter,
    Span,
    WheelZoomTool,
)
from bokeh.plotting import figure, show

tableau20 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


class PlotScheme:
    def __init__(self):
        self.figure_width = 1200
        self.main_height = 400
        self.ind_height = 250
        self.ana_height = 250
        self.vbar_width = 0.6 * 24 * 60 * 60 * 1000
        self.line_width = 1.5
        self.scaling_factor = 0.15
        self.location = "top_left"


def load_and_align(file_path: str, file_type: str = "parquet", tick_unit: str = "s") -> pd.DataFrame:
    """Load long-format bt_core log (datetime, metrics, value) and pivot into an aligned wide DataFrame."""
    if file_type == "parquet":
        df = pd.read_parquet(file_path)
    elif file_type == "csv":
        df = pd.read_csv(file_path)
    elif file_type == "jsonl":
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    if not {"metrics", "value", "datetime"}.issubset(df.columns):
        raise ValueError("Data must contain 'metrics', 'value', 'datetime' columns")

    df["datetime"] = pd.to_datetime(df["datetime"], unit=tick_unit)
    align_df = df.pivot_table(index="datetime", columns="metrics", values="value", aggfunc="last").ffill()
    align_df = align_df[align_df.index != pd.Timestamp(0)]
    if "close" in align_df.columns:
        align_df = align_df.dropna(subset=["close"])
    return align_df.reset_index()


class Plot:
    """Vertical-stacked Bokeh chart builder with linked X-range and synchronized hover crosshairs."""

    def __init__(self, scheme: PlotScheme | None = None):
        self.scheme = scheme or PlotScheme()
        self.fig_main = None
        self.all_figures: list = []
        self.bt_tooltips: dict = {}
        self.bt_renderers: dict = {}
        self.datasource = None

    def plot_from_wide_df(self, df: pd.DataFrame, candle: bool = True, auto_show: bool = True):
        df = df.rename(columns=lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x))
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])
        self.datasource = ColumnDataSource(df)
        available_cols = set(df.columns)
        feed_cols = {"datetime", "open", "high", "low", "close", "volume"}
        df_feed_cols = list(feed_cols.intersection(available_cols))
        ind_cols = [c for c in df.columns if c.startswith("ind_")]
        analyzer_cols = [
            c for c in df.columns
            if c not in df_feed_cols and not c.startswith("ind_") and c != "datetime"
        ]

        if "close" in available_cols:
            self._plot_main(candle)
        elif analyzer_cols:
            self._plot_main_fallback(analyzer_cols.pop(0))
        elif ind_cols:
            self._plot_main_fallback(ind_cols.pop(0))

        if ind_cols:
            self.layout_indicators = self._plot_series_stacked(
                ind_cols, "Technical Indicators", "Indicator", self.scheme.ind_height, color_offset=0, strip_prefix="ind_"
            )
        if analyzer_cols:
            self.layout_analyzers = self._plot_series_stacked(
                analyzer_cols, "Strategy Analyzers", "Analyzer", self.scheme.ana_height, color_offset=5
            )

        layout = self._build_layout()
        if auto_show:
            show(layout)
        return layout

    def plot_from_btcore_log(self, file_path: str, candle: bool = True, tick_unit: str = "s", auto_show: bool = True):
        df = load_and_align(file_path, tick_unit=tick_unit)
        return self.plot_from_wide_df(df, candle=candle and {"open", "high", "low"}.issubset(df.columns), auto_show=auto_show)

    def plot_from_integrated_df(self, df: pd.DataFrame, candle: bool = True, auto_show: bool = True):
        return self.plot_from_wide_df(df, candle=candle and {"open", "high", "low", "close"}.issubset(df.columns), auto_show=auto_show)

    def _plot_main(self, candle: bool) -> None:
        dmaster = self.datasource
        self.fig_main = figure(
            width=self.scheme.figure_width, height=self.scheme.main_height,
            title="OHLCV", x_axis_type="datetime",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        self.all_figures.append(self.fig_main)
        tooltip = [("Date", "@datetime{%F %T}")]
        renderers = []

        if "volume" in dmaster.data:
            vol_data = np.array(dmaster.data["volume"], dtype=float)
            v_max = np.nanmax(vol_data) if not np.all(np.isnan(vol_data)) else 1.0
            p_min = np.nanmin(dmaster.data["low"]) if "low" in dmaster.data else 0.0
            p_max = np.nanmax(dmaster.data["high"]) if "high" in dmaster.data else 1.0
            p_range = p_max - p_min if p_max > p_min else 1.0
            dmaster.data["volume_scaled"] = p_min + (vol_data / v_max) * (p_range * self.scheme.scaling_factor)
            self.fig_main.vbar(
                x="datetime", top="volume_scaled", bottom=p_min,
                width=self.scheme.vbar_width, source=dmaster,
                fill_alpha=0.3, line_alpha=0, color="gray", legend_label="Volume",
            )
            tooltip.append(("Volume", "@volume{0.00}"))

        close_line = self.fig_main.line(
            "datetime", "close", source=dmaster,
            line_width=self.scheme.line_width, color="#1f77b4", legend_label="Close",
        )
        renderers.append(close_line)
        tooltip.append(("Close", "@close{0.00}"))

        if candle and {"open", "high", "low"}.issubset(dmaster.data.keys()):
            tooltip[1:1] = [("Open", "@open{0.00}"), ("High", "@high{0.00}"), ("Low", "@low{0.00}")]
            op, cl = np.array(dmaster.data["open"]), np.array(dmaster.data["close"])
            dmaster.data["top_body"] = np.maximum(op, cl)
            dmaster.data["bottom_body"] = np.minimum(op, cl)
            dmaster.data["line_color"] = np.where(cl >= op, "#D32F2F", "#009624")
            dmaster.data["fill_color"] = np.where(cl >= op, "#FF5252", "#00C853")
            self.fig_main.segment(x0="datetime", y0="top_body", x1="datetime", y1="high", color="line_color", source=dmaster)
            self.fig_main.segment(x0="datetime", y0="bottom_body", x1="datetime", y1="low", color="line_color", source=dmaster)
            self.fig_main.vbar(
                x="datetime", width=self.scheme.vbar_width, top="top_body", bottom="bottom_body",
                source=dmaster, fill_color="fill_color", line_color="line_color", legend_label="Candle",
            )

        self.fig_main.legend.location = self.scheme.location
        self.bt_tooltips[self.fig_main] = tooltip
        self.bt_renderers[self.fig_main] = renderers
        self.layout_main = column(
            Div(text="<h2>Market Execution Feed</h2>", margin=(10, 0, 10, 0), sizing_mode="stretch_width"),
            self.fig_main,
            sizing_mode="stretch_width",
        )

    def _plot_main_fallback(self, col: str) -> None:
        self.fig_main = figure(
            width=self.scheme.figure_width, height=self.scheme.main_height,
            title=f"Main: {col}", x_axis_type="datetime",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        self.all_figures.append(self.fig_main)
        line = self.fig_main.line("datetime", col, source=self.datasource, line_width=self.scheme.line_width, color=tableau20[0])
        self.bt_tooltips[self.fig_main] = [("Date", "@datetime{%F %T}"), (col, f"@{{{col}}}{{0.0000}}")]
        self.bt_renderers[self.fig_main] = [line]
        self.layout_main = column(
            Div(text="<h2>Backtest Analyzers</h2>", margin=(10, 0, 10, 0), sizing_mode="stretch_width"),
            self.fig_main,
            sizing_mode="stretch_width",
        )

    def _plot_series_stacked(
        self, cols: list[str], section_title: str, prefix_label: str, height: int, color_offset: int = 0, strip_prefix: str = ""
    ):
        figs = []
        for i, col in enumerate(cols):
            label = col.replace(strip_prefix, "") if strip_prefix else col
            p = figure(
                width=self.scheme.figure_width, height=height,
                title=f"{prefix_label}: {label}",
                x_axis_type="datetime", x_range=self.fig_main.x_range,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            line = p.line(
                "datetime", col, source=self.datasource,
                line_width=self.scheme.line_width, color=tableau20[(i + color_offset) % len(tableau20)],
            )
            self.all_figures.append(p)
            self.bt_tooltips[p] = [("Date", "@datetime{%F %T}"), (col, f"@{{{col}}}{{0.0000}}")]
            self.bt_renderers[p] = [line]
            figs.append(p)
        return column(Div(text=f"<h3>{section_title}</h3>", margin=(10, 0, 10, 0)), *figs, sizing_mode="stretch_width")

    def _build_layout(self):
        vlines = []
        for plt in self.all_figures:
            vline = Span(location=0, dimension="height", line_color="red", line_width=1, line_alpha=0)
            plt.add_layout(vline)
            vlines.append(vline)

        hover_callback = CustomJS(args=dict(vlines=vlines), code="""
            for (let i = 0; i < vlines.length; i++) {
                vlines[i].line_alpha = 0.5;
                vlines[i].location = cb_data.geometry.x;
            }
        """)

        for plt in self.all_figures:
            hover = HoverTool(
                tooltips=self.bt_tooltips[plt],
                formatters={"@datetime": "datetime"},
                mode="vline", callback=hover_callback,
                renderers=self.bt_renderers[plt],
            )
            plt.add_tools(hover, CrosshairTool(dimensions="both", line_alpha=0.5))
            plt.yaxis.formatter = NumeralTickFormatter(format="0,0.00")
            plt.toolbar.active_scroll = plt.select_one(WheelZoomTool)

        items = [
            getattr(self, attr)
            for attr in ("layout_main", "layout_indicators", "layout_analyzers")
            if hasattr(self, attr)
        ]
        return column(items, sizing_mode="stretch_width")
