"""bt_studio Streamlit app: backtest log viz + param space analysis."""

import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from bokeh.resources import CDN
from bokeh.embed import file_html

from bt_studio.visual.bkh import Plot
from bt_studio.visual.bkh.utils import load_and_align
from bt_studio.visual.landscape_contour import (
    load_ray_results,
    detect_space_collapse,
    plot_collapse_dashboard,
)

st.set_page_config(page_title="bt_studio", layout="wide")

tab_bt, tab_hpo = st.tabs(["Backtest Analyzer", "Tune Param Space"])

with tab_bt:
    st.header("Backtest Log")
    log_file = st.text_input(
        "bt_core parquet log", value="/Users/hengxinliu/startup/bt_studio/tests/logs/log_cerebro_0.parquet", key="log_file"
    )

    if st.button("Render", key="btn_bt"):
        if not os.path.exists(log_file):
            st.error(f"File not found: {log_file}")
        else:
            with st.spinner("Loading..."):
                df = load_and_align(log_file, tick_unit="s")

            if df.empty:
                st.warning("Empty data")
            else:
                st.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
                plotter = Plot()
                layout = plotter.plot_from_wide_df(df, candle=False, auto_show=False)
                html = file_html(layout, CDN, "backtest")
                components.html(html, height=800, scrolling=True)
                with st.expander("Raw data"):
                    st.dataframe(df)

with tab_hpo:
    st.header("Param Space")
    exp_path = st.text_input(
        "Ray Tune experiment dir",
        value="/tmp/ray_results/fsm_hpo_201508",
        key="exp_path",
    )

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Target", ["metrics_score", "u_pval", "trigger_count"])
    with col2:
        method = st.selectbox("Interp", ["cubic", "linear", "rbf"])

    bounds_str = st.text_input(
        "Search bounds (JSON)",
        value='{"downsample": [3,4,5], "motif_minutes": [30,45,60,90], "threshold_r": [0.6,0.8]}',
        key="bounds",
    )

    if st.button("Analyze", key="btn_hpo"):
        if not os.path.exists(exp_path):
            st.error(f"Dir not found: {exp_path}")
        else:
            with st.spinner("Loading Ray results..."):
                rdf = load_ray_results(exp_path)
            if rdf.empty:
                st.warning("No trials")
            else:
                st.success(f"Loaded {len(rdf)} trials")
                try:
                    bounds = json.loads(bounds_str)
                except Exception:
                    bounds = None

                with st.spinner("Collapse detection..."):
                    diag = detect_space_collapse(rdf, target, bounds)

                verdict = diag["overall_verdict"]
                tag = {"healthy": "[OK]", "collapsed": "[!]", "flat_landscape": "[--]"}.get(verdict, "?")
                st.subheader(f"{tag} Verdict: {verdict.upper()}")

                rows = []
                for col in [c for c in rdf.columns if c.startswith("config/")]:
                    name = col.replace("config/", "")
                    d = diag.get(name)
                    if d is None:
                        continue
                    sr = round(d["spread_ratio"], 3) if d["spread_ratio"] == d["spread_ratio"] else None
                    rows.append({
                        "param": name,
                        "type": "disc" if d["is_discrete"] else "cont",
                        "spread": sr,
                        "entropy": round(d["kde_entropy"], 3),
                        "land_var": round(d["landscape_variance_ratio"], 3),
                        "status": "flat" if d["is_flat"] else ("collapse" if d["is_collapsed"] else "ok"),
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch")

                with st.spinner("Generating dashboard..."):
                    fig, _ = plot_collapse_dashboard(rdf, target, bounds, method=method)
                    st.pyplot(fig, width="stretch")