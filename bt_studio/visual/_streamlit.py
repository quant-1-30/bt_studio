#!/usr/bin/env python3
"""Unified multi-channel online quant desk (Screen A: Research & HPO, Screen B: K-Line & Motif, Screen C: Market Screener)."""

from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import streamlit as st
from bokeh.embed import file_html
from bokeh.resources import CDN

from bt_studio.visual.bkh import Plot, load_and_align
from bt_studio.visual.collapse import (
    detect_space_collapse,
    load_real_ray_results as load_ray_results,
    plot_collapse_dashboard,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_LOG_PATH = os.path.join(_PROJECT_ROOT, "scripts", "logs", "log_cerebro_0.parquet")

VIEW_MODES = [
    "All-in-one desk",
    "Screen A: Research & HPO",
    "Screen B: K-line & motif",
    "Screen C: Market screener",
]

st.set_page_config(page_title="bt_studio quant desk", layout="wide")


@st.cache_resource
def _get_monitor_index():
    from bt_studio.plugins.api_server.indexer import MonitorIndex

    return MonitorIndex()


@st.cache_data(ttl=5, max_entries=16)
def _cached_load_backtest(log_path: str) -> pd.DataFrame:
    if not log_path or not os.path.exists(log_path):
        return pd.DataFrame()
    return load_and_align(log_path, tick_unit="s")


@st.cache_data(ttl=5, max_entries=16)
def _cached_load_ray(exp_dir: str) -> pd.DataFrame:
    if not exp_dir or not os.path.exists(exp_dir):
        return pd.DataFrame()
    return load_ray_results(exp_dir)


# ---------------------------------------------------------------------------
# Top Control Bar: Workspace / Multi-Screen Router & Live Stream Toggle
# ---------------------------------------------------------------------------
with st.container(horizontal=True, vertical_alignment="bottom"):
    view_mode = st.segmented_control(
        "Workspace channel (shareable URL for multi-monitor display)",
        options=VIEW_MODES,
        default=VIEW_MODES[0],
        key="view_mode",
        bind="query-params",
    )
    live_stream = st.toggle(":material/sensors: Live stream (3s)", value=False, key="live_stream")

run_every = "3s" if live_stream else None


# ---------------------------------------------------------------------------
# Channel C: Cross-Sectional Market Screener & Cross-Screen Symbol Linker
# ---------------------------------------------------------------------------
@st.fragment(run_every=run_every)
def render_screen_c_screener() -> None:
    idx = _get_monitor_index()
    idx.scan_if_stale(min_interval=2.0)
    focus = idx.get_focus()
    screener_data = idx.screener(model_id=focus.get("model_id"), limit=200)
    items = screener_data.get("items") or []

    with st.container(border=True):
        st.subheader(":material/radar: Screen C — Cross-sectional trigger radar")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            focus_sid = st.text_input(
                "Linked symbol (sid)",
                value=str(focus.get("sid") or "600000"),
                key="input_focus_sid",
            )
            focus_model = st.number_input(
                "Model cutoff (YYYYMM)",
                value=int(screener_data.get("model_id") or focus.get("model_id") or 202412),
                step=1,
                key="input_focus_model",
            )
            if st.button(":material/sync: Broadcast focus to all screens", key="btn_sync_focus"):
                idx.set_focus(sid=focus_sid, model_id=int(focus_model))
                st.toast(f"Broadcasted focus: sid={focus_sid}, model={focus_model}")

        if not items:
            st.info(
                "No OOS score parquet indexed yet. Run `wfo_production` or `wfo_hpo` to populate real-time triggers.",
                title="Waiting for OOS trigger stream",
            )
        else:
            df_scr = pd.DataFrame(items)
            st.caption(
                f"Active model: `{screener_data.get('model_id')}` | Triggered stocks: `{len(df_scr)}` | "
                f"Current linked symbol: `{focus.get('sid')}` (click a row to broadcast across screens)"
            )
            sel_event = st.dataframe(
                df_scr,
                width="stretch",
                height=320,
                on_select="rerun",
                selection_mode="single-row",
                key="screener_table",
                alt="Cross-sectional FSMPredictor trigger radar",
            )
            if sel_event and sel_event.selection and sel_event.selection.rows:
                row_idx = sel_event.selection.rows[0]
                if 0 <= row_idx < len(df_scr) and "sid" in df_scr.columns:
                    clicked_sid = str(df_scr.iloc[row_idx]["sid"])
                    if clicked_sid != str(focus.get("sid")):
                        idx.set_focus(sid=clicked_sid, model_id=int(focus_model))
                        st.toast(f"Synced focus to clicked symbol: {clicked_sid}")


# ---------------------------------------------------------------------------
# Channel B: Intraday K-Line, Indicators, Execution & Motif View
# ---------------------------------------------------------------------------
@st.fragment(run_every=run_every)
def render_screen_b_kline() -> None:
    idx = _get_monitor_index()
    focus = idx.get_focus()

    with st.container(border=True):
        st.subheader(f":material/candlestick_chart: Screen B — Microstructure & execution feed (focus: `{focus.get('sid')}`)")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            log_file = st.text_input(
                "bt_core parquet log path",
                value=_DEFAULT_LOG_PATH,
                key="log_file_path",
            )
            show_candle = st.toggle("Candlestick", value=False, key="toggle_candle")
            render_now = st.button(":material/refresh: Load feed", key="btn_load_bt")

        if render_now or live_stream or os.path.exists(log_file):
            if not os.path.exists(log_file):
                st.warning(f"Parquet log not found at `{log_file}`.", title="Feed file unavailable")
            else:
                df = _cached_load_backtest(log_file)
                if df.empty:
                    st.warning("Log file contains no valid rows.", title="Empty feed")
                else:
                    plotter = Plot()
                    layout = plotter.plot_from_wide_df(df, candle=show_candle, auto_show=False)
                    html = file_html(layout, CDN, "bt_studio_feed")
                    st.html(html, unsafe_allow_javascript=True)
                    with st.expander("Inspect aligned feed table"):
                        st.dataframe(df.tail(200), width="stretch", alt="Aligned backtest feed rows")


# ---------------------------------------------------------------------------
# Channel A: Research Flywheel Monitor & HPO Parameter Landscape
# ---------------------------------------------------------------------------
@st.fragment(run_every=run_every)
def render_screen_a_research() -> None:
    idx = _get_monitor_index()
    idx.scan_if_stale(min_interval=2.0)
    ov = idx.overview()
    usage = ov.get("usage") or {}
    cov = ov.get("raw_pit") or {}

    with st.container(border=True):
        st.subheader(":material/science: Screen A — RL flywheel & HPO parameter space")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active runs (24h)", int(ov.get("active_runs_24h") or 0))
        c2.metric("Viable features", int(ov.get("viable_features") or 0))
        c3.metric("Today LLM tokens", f"{int(usage.get('total_tokens') or 0):,}")
        c4.metric("Raw PIT coverage", f"{float(cov.get('coverage') or 0.0):.1%}")

        feats = (idx.list_features(limit=20) or {}).get("items") or []
        if feats:
            st.caption("Top viable features (Stage 2 survivors)")
            st.dataframe(pd.DataFrame(feats), width="stretch", height=220, alt="Top viable Stage 2 features")

        st.divider()
        with st.form("hpo_landscape_form"):
            exp_path = st.text_input(
                "Ray Tune experiment directory",
                value="/tmp/ray_results/fsm_hpo_201508",
                key="exp_path",
            )
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                target = st.selectbox("Objective metric", ["metrics_score", "u_pval", "trigger_count"], key="hpo_target")
            with fcol2:
                method = st.selectbox("Surface interpolation", ["cubic", "linear", "rbf"], key="hpo_interp")
            bounds_str = st.text_input(
                "Search bounds (JSON)",
                value='{"downsample": [3,4,5], "motif_minutes": [30,45,60,90], "threshold_r": [0.65,0.85]}',
                key="hpo_bounds",
            )
            submitted = st.form_submit_button(":material/analytics: Analyze parameter landscape")

        if submitted:
            if not os.path.exists(exp_path):
                st.error(f"Directory not found: `{exp_path}`", title="Missing experiment path")
            else:
                rdf = _cached_load_ray(exp_path)
                if rdf.empty:
                    st.warning("No valid trials found in directory.", title="Empty experiment")
                else:
                    try:
                        bounds = json.loads(bounds_str)
                    except Exception:
                        bounds = None
                    diag = detect_space_collapse(rdf, target, bounds)
                    verdict = diag["overall_verdict"]
                    st.markdown(f"**Parameter space verdict:** `{verdict.upper()}` (`n_trials={diag['_n_trials']}`)")
                    rows = []
                    for col in [c for c in rdf.columns if c.startswith("config/")]:
                        name = col.replace("config/", "")
                        d = diag.get(name)
                        if isinstance(d, dict):
                            rows.append({
                                "param": name,
                                "type": "discrete" if d["is_discrete"] else "continuous",
                                "spread": round(d["spread_ratio"], 3) if d["spread_ratio"] == d["spread_ratio"] else None,
                                "entropy": round(d["kde_entropy"], 3),
                                "land_var": round(d["landscape_variance_ratio"], 3),
                                "worst_drop_iqr": round(d.get("worst_drop_ratio", 0.0), 2),
                                "status": (
                                    "spike" if d.get("is_isolated_spike")
                                    else "collapse" if d.get("is_collapsed")
                                    else "flat" if d.get("is_flat")
                                    else "ok"
                                ),
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch", alt="Parameter space diagnostics summary")
                    fig, _ = plot_collapse_dashboard(rdf, target, bounds, method=method)
                    st.pyplot(fig, width="stretch", alt="HPO parameter space contour and 3D landscape")


# ---------------------------------------------------------------------------
# Route Dispatch: Single-Screen Switch or Multi-Monitor Dedicated Views
# ---------------------------------------------------------------------------
active_mode = view_mode or VIEW_MODES[0]

if active_mode == "Screen A: Research & HPO":
    render_screen_a_research()
elif active_mode == "Screen B: K-line & motif":
    render_screen_b_kline()
elif active_mode == "Screen C: Market screener":
    render_screen_c_screener()
else:
    col_left, col_right = st.columns([1.0, 1.4])
    with col_left:
        render_screen_c_screener()
    with col_right:
        render_screen_b_kline()
    render_screen_a_research()
