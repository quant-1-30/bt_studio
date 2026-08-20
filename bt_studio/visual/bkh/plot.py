import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import (ColumnDataSource, HoverTool, CrosshairTool,
                          Span, CustomJS, PanTool, WheelZoomTool, NumeralTickFormatter,
                          Div)

from .scheme import tableau20, PlotScheme
from .utils import load_and_align


class Plot(object):
    """

    vertical stacking + X-axis linking
        +------------------------------------------+
        |  1. OHLCV Main (line/candle + volume)    |  <- main_*
        +------------------------------------------+
        |  2. Indicators                           |  <- ind_* 
        +------------------------------------------+
        |  3. Analyzers                            |  <- bt_core metrics
        +------------------------------------------+

    Link Mechanism:
        1. X-axis range linking: All subplots share fig_main.x_range
        2. Crosshair linking: CustomJS hover callback synchronizes red vertical lines across all subplots
        3. Hover tooltip linking: Each subplot has its own HoverTool, with consistent datetime formatting
    """

    def __init__(self, scheme=None):
        self.scheme = scheme or PlotScheme()
        self.fig_main = None
        self.all_figures = []
        self.bt_tooltips = {}
        self.bt_renderers = {}
        self.datasource = None

    def plot_from_wide_df(self, df, candle=True, auto_show=True):
        df = df.rename(columns=lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x))
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])
        self.datasource = ColumnDataSource(df)
        available_cols = set(df.columns)
        feed_cols = {"datetime", "open", "high", "low", "close", "volume"}
        df_feed_cols = list(feed_cols.intersection(available_cols))
        ind_cols = [c for c in df.columns if c.startswith("ind_")]
        analyzer_cols = [c for c in df.columns if c not in df_feed_cols and not c.startswith("ind_") and c != "datetime"]
        has_ohlcv = {"close"}.issubset(available_cols)
        if has_ohlcv:
            self._plot_main(candle)
        elif analyzer_cols:
            self._plot_main_fallback(analyzer_cols[0])
            analyzer_cols = analyzer_cols[1:]
        elif ind_cols:
            self._plot_main_fallback(ind_cols[0])
            ind_cols = ind_cols[1:]
        if ind_cols:
            self._plot_indicators_stacked(ind_cols)
        if analyzer_cols:
            self._plot_analyzers_stacked(analyzer_cols)
        layout = self._build_layout()
        if auto_show:
            show(layout)
        return layout

    def plot_from_btcore_log(self, file_path, candle=True, tick_unit="s", auto_show=True):
        df = load_and_align(file_path, tick_unit=tick_unit)
        ohlcv_present = {"open", "high", "low"}.issubset(df.columns)
        if not ohlcv_present:
            candle = False
        return self.plot_from_wide_df(df, candle=candle, auto_show=auto_show)

    def plot_from_integrated_df(self, df, candle=True, auto_show=True):
        ohlcv_present = {"open", "high", "low", "close"}.issubset(df.columns)
        if not ohlcv_present:
            candle = False
        return self.plot_from_wide_df(df, candle=candle, auto_show=auto_show)

    def _plot_main(self, candle):
        dmaster = self.datasource
        self.fig_main = figure(
            width=self.scheme.figure_width, height=self.scheme.main_height,
            title="OHLCV", x_axis_type="datetime",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        self.all_figures.append(self.fig_main)
        _tooltip = [("Date", "@datetime{%F %T}")]
        renderers = []

        if "volume" in dmaster.data:
            vol_data = np.array(dmaster.data["volume"], dtype=float)
            v_max = np.nanmax(vol_data) if not np.all(np.isnan(vol_data)) else 1
            p_min = np.nanmin(dmaster.data["low"]) if "low" in dmaster.data else 0
            p_max = np.nanmax(dmaster.data["high"]) if "high" in dmaster.data else 1
            p_range = p_max - p_min if p_max > p_min else 1
            dmaster.data["volume_scaled"] = p_min + (vol_data / v_max) * (p_range * self.scheme.scaling_factor)
            self.fig_main.vbar(x="datetime", top="volume_scaled", bottom=p_min,
                               width=self.scheme.vbar_width, source=dmaster,
                               fill_alpha=0.3, line_alpha=0, color="gray", legend_label="Volume")
            _tooltip.append(("Volume", "@volume{0.00}"))

        close_line = self.fig_main.line("datetime", "close", source=dmaster,
                                        line_width=self.scheme.line_width, color="#1f77b4", legend_label="Close")
        renderers.append(close_line)
        _tooltip.append(("Close", "@close{0.00}"))

        if candle and {"open", "high", "low"}.issubset(dmaster.data.keys()):
            _tooltip.insert(1, ("Open", "@open{0.00}"))
            _tooltip.insert(2, ("High", "@high{0.00}"))
            _tooltip.insert(3, ("Low", "@low{0.00}"))
            op, cl = np.array(dmaster.data["open"]), np.array(dmaster.data["close"])
            dmaster.data["top_body"] = np.maximum(op, cl)
            dmaster.data["bottom_body"] = np.minimum(op, cl)
            dmaster.data["line_color"] = np.where(cl >= op, "#D32F2F", "#009624")
            dmaster.data["fill_color"] = np.where(cl >= op, "#FF5252", "#00C853")
            self.fig_main.segment(x0="datetime", y0="top_body", x1="datetime", y1="high",
                                  color="line_color", source=dmaster)
            self.fig_main.segment(x0="datetime", y0="bottom_body", x1="datetime", y1="low",
                                  color="line_color", source=dmaster)
            self.fig_main.vbar(x="datetime", width=self.scheme.vbar_width, top="top_body", bottom="bottom_body",
                               source=dmaster, fill_color="fill_color", line_color="line_color", legend_label="Candle")

        self.fig_main.legend.location = self.scheme.location
        self.bt_tooltips[self.fig_main] = _tooltip
        self.bt_renderers[self.fig_main] = renderers
        title_div = Div(text="<h2>Market Execution Feed</h2>", margin=(10, 0, 10, 0), sizing_mode="stretch_width")
        self.layout_main = column(title_div, self.fig_main, sizing_mode="stretch_width")

    def _plot_main_fallback(self, col):
        """Fallback main figure for bt_core analyzer-only logs (no OHLCV)."""
        self.fig_main = figure(
            width=self.scheme.figure_width, height=self.scheme.main_height,
            title=f"Main: {col}", x_axis_type="datetime",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        self.all_figures.append(self.fig_main)
        line = self.fig_main.line("datetime", col, source=self.datasource,
                                  line_width=self.scheme.line_width, color=tableau20[0])
        self.bt_tooltips[self.fig_main] = [("Date", "@datetime{%F %T}"), (col, f"@{{{col}}}{{0.0000}}")]
        self.bt_renderers[self.fig_main] = [line]
        title_div = Div(text="<h2>Backtest Analyzers</h2>", margin=(10, 0, 10, 0), sizing_mode="stretch_width")
        self.layout_main = column(title_div, self.fig_main, sizing_mode="stretch_width")

    def _plot_indicators_stacked(self, ind_cols):
        """每个 ind_* 列渲染为独立子图,垂直堆叠,共享 X 轴。"""
        ind_figs = []
        for i, col in enumerate(ind_cols):
            p_ind = figure(
                width=self.scheme.figure_width, height=self.scheme.ind_height,
                title=f"Indicator: {col.replace('ind_', '')}",
                x_axis_type="datetime", x_range=self.fig_main.x_range,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            color = tableau20[i % len(tableau20)]
            ind_line = p_ind.line("datetime", col, source=self.datasource,
                                  line_width=self.scheme.line_width, color=color)
            self.all_figures.append(p_ind)
            self.bt_tooltips[p_ind] = [("Date", "@datetime{%F %T}"), (col, f"@{{{col}}}{{0.0000}}")]
            self.bt_renderers[p_ind] = [ind_line]
            ind_figs.append(p_ind)
        title_div = Div(text="<h3>Technical Indicators</h3>", margin=(10, 0, 10, 0))
        self.layout_indicators = column(title_div, *ind_figs, sizing_mode="stretch_width")

    def _plot_analyzers_stacked(self, analyzer_cols):
        ana_figs = []
        for i, col in enumerate(analyzer_cols):
            p_ana = figure(
                width=self.scheme.figure_width, height=self.scheme.ana_height,
                title=f"Analyzer: {col}",
                x_axis_type="datetime", x_range=self.fig_main.x_range,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            color = tableau20[(i + 5) % len(tableau20)]
            ana_line = p_ana.line("datetime", col, source=self.datasource,
                                  line_width=self.scheme.line_width, color=color)
            self.all_figures.append(p_ana)
            self.bt_tooltips[p_ana] = [("Date", "@datetime{%F %T}"), (col, f"@{{{col}}}{{0.0000}}")]
            self.bt_renderers[p_ana] = [ana_line]
            ana_figs.append(p_ana)
        title_div = Div(text="<h3>Strategy Analyzers</h3>", margin=(10, 0, 10, 0))
        self.layout_analyzers = column(title_div, *ana_figs, sizing_mode="stretch_width")

    def _build_layout(self):
        _vlines = []
        for _plt in self.all_figures:
            vline = Span(location=0, dimension="height", line_color="red", line_width=1, line_alpha=0)
            _plt.add_layout(vline)
            _vlines.append(vline)

        hover_callback = CustomJS(args=dict(vlines=_vlines), code="""
            for (let i = 0; i < vlines.length; i++) {
                vlines[i].line_alpha = 0.5;
                vlines[i].location = cb_data.geometry.x;
            }
        """)

        for _plt in self.all_figures:
            hover = HoverTool(
                tooltips=self.bt_tooltips[_plt],
                formatters={"@datetime": "datetime"},
                mode="vline", callback=hover_callback,
                renderers=self.bt_renderers[_plt],
            )
            _plt.add_tools(hover, CrosshairTool(dimensions="both", line_alpha=0.5))
            _plt.yaxis.formatter = NumeralTickFormatter(format="0,0.00")
            _plt.toolbar.active_scroll = _plt.select_one(WheelZoomTool)

        layout_items = []
        if hasattr(self, "layout_main"):
            layout_items.append(self.layout_main)
        if hasattr(self, "layout_indicators"):
            layout_items.append(self.layout_indicators)
        if hasattr(self, "layout_analyzers"):
            layout_items.append(self.layout_analyzers)
        return column(layout_items, sizing_mode="stretch_width")


if __name__ == "__main__":
    _p = Plot()
    p_str = "/Users/hengxinliu/startup/bt_core/tests/logs/log_cerebro_0.parquet"
    _p.plot_from_btcore_log(p_str, candle=False)
