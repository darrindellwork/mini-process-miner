import pandas as pd
import shutil, importlib, streamlit as st
st.write("Python OK. Checking deps…")
st.write("pm4py import:", bool(importlib.util.find_spec("pm4py")))
st.write("graphviz (pip) import:", bool(importlib.util.find_spec("graphviz")))
st.write("dot in PATH:", shutil.which("dot"))


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Mini Process Miner", layout="wide")
st.title("Mini Process Miner (vibe-coded)")

# Uploader with clear instructions
uploaded = st.file_uploader(
    "Upload your event log (CSV)",
    type=["csv"],
    help="Use EXACT headers (lowercase): required → case_id, activity, timestamp; optional → column1, column2, column3."
)
st.caption("**Required columns: (Please make sure the headers in your uploaded event log have the EXACT same names as the required here: ** case_id, activity, timestamp  •  **Optional:** column1, column2, column3 (e.g., resource, team, location)  •  **Disclaimer:** This demo tool offers no guarantees regarding data security or accuracy; use at your own risk.")

# ----------------------------
# Helpers
# ----------------------------
def ensure_parsed(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and parse timestamp."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def compute_ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["case_id", "timestamp"])

def apply_case_level_exclusion(df: pd.DataFrame, activities_to_drop: list) -> pd.DataFrame:
    """Remove entire cases that contain any of the selected activities."""
    if not activities_to_drop:
        return df
    cases_with_forbidden = df.loc[df["activity"].isin(activities_to_drop), "case_id"].unique()
    return df.loc[~df["case_id"].isin(cases_with_forbidden)].copy()

def apply_event_level_exclusion(df: pd.DataFrame, activities_to_remove: list) -> pd.DataFrame:
    """Remove only those activity events, keep the rest of the case."""
    if not activities_to_remove:
        return df
    out = df.loc[~df["activity"].isin(activities_to_remove)].copy()
    valid_cases = out["case_id"].value_counts()
    keep_cases = valid_cases[valid_cases > 0].index
    return out.loc[out["case_id"].isin(keep_cases)].copy()

def apply_activity_threshold(df: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    """Drop events whose activity total frequency < min_freq."""
    if min_freq <= 1 or df.empty:
        return df
    counts = df["activity"].value_counts()
    keep_acts = counts[counts >= min_freq].index
    return df.loc[df["activity"].isin(keep_acts)].copy()

def build_edges(ordered_df: pd.DataFrame) -> pd.DataFrame:
    """Build directly-follows edges with counts."""
    if ordered_df.empty:
        return pd.DataFrame(columns=["edge", "count"])
    tmp = ordered_df.copy()
    tmp["next_activity"] = tmp.groupby("case_id")["activity"].shift(-1)
    edges = tmp.dropna(subset=["next_activity"])[["activity", "next_activity"]]
    if edges.empty:
        return pd.DataFrame(columns=["edge", "count"])
    edges["edge"] = edges["activity"] + " → " + edges["next_activity"]
    edge_counts = edges["edge"].value_counts().rename_axis("edge").reset_index(name="count")
    return edge_counts

def apply_optional_column_includes(df: pd.DataFrame, colname: str, selected: list) -> pd.DataFrame:
    """If selections provided for a column, keep only rows where column ∈ selected."""
    if colname in df.columns and selected:
        return df[df[colname].astype(str).isin([str(x) for x in selected])]
    return df

# ----------------------------
# Main
# ----------------------------
if uploaded:
    raw_df = pd.read_csv(uploaded)

    # Validate columns early (we normalize to lowercase)
    required = {"case_id", "activity", "timestamp"}
    if not required.issubset(set([c.strip().lower() for c in raw_df.columns])):
        st.error("CSV must include required columns: case_id, activity, timestamp. Optional: column1, column2, column3.")
        st.stop()

    df = ensure_parsed(raw_df)

    # ----------------------------
    # Sidebar filters (case/event + optional column1/2/3) FIRST
    # ----------------------------
    st.sidebar.header("Filters")

    # Optional extra columns (exact names after normalization): column1, column2, column3
    extra_cols_present = [c for c in ["column1", "column2", "column3"] if c in df.columns]

    # Case-level exclusion
    all_activities = sorted(df["activity"].astype(str).unique().tolist())
    case_exclude = st.sidebar.multiselect(
        "Remove all CASES containing these activities",
        options=all_activities,
        help="If a case contains one of these activities, the entire case is removed."
    )

    # Event-level exclusion
    event_exclude = st.sidebar.multiselect(
        "Remove only EVENTS with these activities (keep cases)",
        options=all_activities,
        help="Events with these activities are dropped, but the case remains if other events exist."
    )

    # Optional include filters for extra columns
    if extra_cols_present:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Optional column filters")
        selections = {}
        for col in extra_cols_present:
            options = sorted(df[col].dropna().astype(str).unique().tolist())
            selections[col] = st.sidebar.multiselect(
                f"Include only {col} values",
                options=options,
                help=f"Leave empty to include all {col} values."
            )
    else:
        selections = {}

    # Apply case/event filters
    df_filt = apply_case_level_exclusion(df, case_exclude)
    df_filt = apply_event_level_exclusion(df_filt, event_exclude)

    # Apply optional column includes
    for col, sel in selections.items():
        df_filt = apply_optional_column_includes(df_filt, col, sel)

    if df_filt.empty:
        st.warning("All data filtered out. Adjust filters to see results.")
        st.stop()

    ordered = compute_ordered(df_filt)

    # ----------------------------
    # Sidebar sliders (activity & connection thresholds)
    # ----------------------------
    act_counts_for_slider = ordered["activity"].value_counts()
    max_act_allowed = int(act_counts_for_slider.max()) if not act_counts_for_slider.empty else 1
    if max_act_allowed < 1:
        max_act_allowed = 1

    apply_act_thresh_to_model = st.sidebar.checkbox(
        "Apply activity frequency threshold to the model",
        value=True,
        help="If enabled, activities below the threshold are removed before discovery/visualization."
    )
    min_act = st.sidebar.slider(
        "Min activity frequency to KEEP",
        min_value=1, max_value=max_act_allowed, value=1,
        help="Drops activities whose total frequency is below this value (if enabled above)."
    )

    # Create df_model after activity slider decision
    if apply_act_thresh_to_model:
        df_model = apply_activity_threshold(ordered, min_act)
    else:
        df_model = ordered

    df_model = compute_ordered(df_model)
    if df_model.empty:
        st.warning("All events dropped by the activity frequency threshold. Lower the threshold.")
        st.stop()

    # Connection frequency slider (visual-only)
    edge_counts_for_slider = build_edges(df_model)
    max_edge_allowed = int(edge_counts_for_slider["count"].max()) if not edge_counts_for_slider.empty else 1
    if max_edge_allowed < 1:
        max_edge_allowed = 1
    min_edge = st.sidebar.slider(
        "Min connection frequency to SHOW",
        min_value=1, max_value=max_edge_allowed, value=1,
        help="Hides low-frequency connections in the Connections/DFG views (visual-only)."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Activity threshold may modify the model; connection threshold only affects visuals.")

    # ----------------------------
    # Metrics
    # ----------------------------
    total_cases = df_model["case_id"].nunique()
    total_events = len(df_model)
    unique_acts = df_model["activity"].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total cases", total_cases)
    c2.metric("Total events", total_events)
    c3.metric("Unique activities", unique_acts)

    # ----------------------------
    # Activity frequency (reflects min_act)
    # ----------------------------
    st.subheader("Activity frequency")
    act_counts = df_model["activity"].value_counts().rename_axis("activity").reset_index(name="count")
    st.dataframe(act_counts[act_counts["count"] >= min_act], use_container_width=True)
    st.bar_chart(act_counts.set_index("activity")["count"])

    # ----------------------------
    # Variants (quick & dirty)
    # ----------------------------
    try:
        variants = (
            df_model.groupby("case_id")["activity"]
            .apply(lambda s: " → ".join(s))
            .value_counts()
        )
        st.subheader("Top variants (quick & dirty)")
        st.dataframe(
            variants.rename("count").reset_index().rename(columns={"index": "variant"}).head(20),
            use_container_width=True
        )
    except Exception:
        st.info("Could not compute variants; check your timestamp and activity values.")

    # ----------------------------
    # Connections (transitions) — respects min_edge (visual-only)
    # ----------------------------
    st.subheader("Connections (transitions)")
    edge_counts = build_edges(df_model)
    if edge_counts.empty:
        st.info("No transitions found after current filters.")
    else:
        st.dataframe(edge_counts[edge_counts["count"] >= min_edge], use_container_width=True)

    # ----------------------------
    # PM4Py visualizations (clean, frequency, performance, DFG)
    # ----------------------------
    st.subheader("Discovered Process Map")
    try:
        # Lazy imports so app still loads without pm4py
        from pm4py.objects.log.util import dataframe_utils
        from pm4py.objects.conversion.log import converter as log_converter
        from pm4py.algo.discovery.inductive import algorithm as inductive_miner
        from pm4py.visualization.petri_net import visualizer as pn_visualizer
        from pm4py.visualization.process_tree import visualizer as pt_visualizer
        from pm4py.objects.conversion.process_tree import converter as pt_converter
        from pm4py.objects.process_tree import obj as pt_obj
        from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
        from pm4py.visualization.dfg import visualizer as dfg_visualization

        # Prepare dataframe for PM4Py
        pm_df = df_model.rename(columns={
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp"
        }).copy()
        pm_df["time:timestamp"] = pd.to_datetime(pm_df["time:timestamp"], errors="coerce")
        pm_df = pm_df.dropna(subset=["time:timestamp"])
        pm_df = dataframe_utils.convert_timestamp_columns_in_df(pm_df)

        # Convert to event log
        event_log = log_converter.apply(pm_df)

        # Discover model
        model = inductive_miner.apply(event_log)
        if isinstance(model, pt_obj.ProcessTree):
            tree = model
            net, im, fm = pt_converter.apply(tree)
            tree_gviz = pt_visualizer.apply(tree)
        else:
            net, im, fm = model
            tree_gviz = None

        tabs = st.tabs(["Clean Petri Net", "Frequency", "Performance", "DFG (with numbers)"])

        # --- Clean Petri net ---
        with tabs[0]:
            gviz_pn = pn_visualizer.apply(net, im, fm)
            st.graphviz_chart(gviz_pn.source, use_container_width=True)
            if tree_gviz is not None:
                st.caption("Process Tree (discovered)")
                st.graphviz_chart(tree_gviz.source, use_container_width=True)

        # --- Frequency-decorated Petri net ---
        with tabs[1]:
            try:
                gviz_freq = pn_visualizer.apply(
                    net, im, fm,
                    variant=pn_visualizer.Variants.FREQUENCY,
                    log=event_log
                )
                st.graphviz_chart(gviz_freq.source, use_container_width=True)
                st.caption("Numbers reflect frequencies from the filtered log.")
            except Exception as e:
                st.info(f"Frequency decoration not available: {e}")

        # --- Performance-decorated Petri net ---
        with tabs[2]:
            try:
                gviz_perf = pn_visualizer.apply(
                    net, im, fm,
                    variant=pn_visualizer.Variants.PERFORMANCE,
                    log=event_log
                )
                st.graphviz_chart(gviz_perf.source, use_container_width=True)
                st.caption("Numbers reflect performance (e.g., average durations) computed from timestamps.")
            except Exception as e:
                st.info(f"Performance decoration not available: {e}")

        # --- DFG with numbers (respects min_edge visually) ---
        with tabs[3]:
            try:
                dfg_freq = dfg_discovery.apply(event_log)  # {(a,b): count}
                dfg_freq_filtered = {k: v for k, v in dfg_freq.items() if v >= min_edge}
                dfg_freq_gviz = dfg_visualization.apply(
                    dfg_freq_filtered if dfg_freq_filtered else dfg_freq,
                    log=event_log,
                    variant=dfg_visualization.Variants.FREQUENCY
                )
                st.graphviz_chart(dfg_freq_gviz.source, use_container_width=True)
                st.caption("DFG (Frequency): edge labels show counts. Low-frequency edges hidden per slider.")

                dfg_perf_gviz = dfg_visualization.apply(
                    dfg_freq_filtered if dfg_freq_filtered else dfg_freq,
                    log=event_log,
                    variant=dfg_visualization.Variants.PERFORMANCE
                )
                st.graphviz_chart(dfg_perf_gviz.source, use_container_width=True)
                st.caption("DFG (Performance): edge labels show avg durations. Low-frequency edges hidden per slider.")
            except Exception as e:
                st.info(f"DFG visualization not available: {e}")

    except ModuleNotFoundError:
        st.error("PM4Py not found. In Anaconda Prompt run: pip install pm4py graphviz and restart.")
    except Exception as e:
        st.warning(f"Could not render process map: {e}")

    # ----------------------------
    # Credits
    # ----------------------------
    st.markdown("---")
    with st.expander("Credits", expanded=False):
        st.markdown(
            """
**Credits**  
Created by **Dennis Arrindell** — creator of the best selling online course about Process Mining on Udemy.

100% Vibe coded using ChatGPT

Inspired by the pioneering work of **Wil van der Aalst**, the “godfather of process mining.”  

Powered by the **PM4Py** process mining library, created by **Sebastiaan J. van Zelst** and contributors.  

Built with Python and other open-source libraries (pandas, Streamlit, Graphviz, etc.).  

Full technical information, installation steps, and source code available in the **GitHub repository**.
            """
        )
