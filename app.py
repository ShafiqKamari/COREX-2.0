
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="COREX Weighting - 9 Step App", layout="wide")

CORE_TYPES = ["Benefit", "Cost", "Target"]

# -----------------------------
# COREX helpers
# -----------------------------
def normalize_column(x: pd.Series, ctype: str, target: float = None) -> pd.Series:
    xmin, xmax = x.min(), x.max()
    if ctype == "Benefit":
        r = (x - xmin) / (xmax - xmin)
    elif ctype == "Cost":
        r = (xmax - x) / (xmax - xmin)
    elif ctype == "Target":
        d = (x - target).abs()
        Dmax = d.max()
        r = 1.0 - d / Dmax
    else:
        raise ValueError("Unknown criterion type")
    return r.astype(float)

def corex_9step(df_vals: pd.DataFrame, crit_types: dict, targets: dict, alpha: float = 0.5):
    cols = list(df_vals.columns)
    n = len(cols)

    # Step 1: Normalize to R
    R = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        t = crit_types[c]
        if t == "Target":
            R[c] = normalize_column(df_vals[c], t, targets.get(c, 0.0))
        else:
            R[c] = normalize_column(df_vals[c], t)

    # Step 2: Overall performance P with 1/n
    P = R.sum(axis=1) / n

    # Step 3: Leave one out performance P_minus with 1/n
    row_sums = R.sum(axis=1)
    P_minus = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        P_minus[c] = (row_sums - R[c]) / n  # divisor n by design

    # Step 4: Absolute drops D and removal impact Rj
    D = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        D[c] = (P - P_minus[c]).abs()
    Rj = D.sum(axis=0).rename("RemovalImpact")

    # Step 5: Per criterion standard deviation sigma on R
    sigma = R.std(axis=0, ddof=1).rename("Sigma")

    # Step 6: Pearson correlation on R
    corr = R.corr(method="pearson").fillna(0.0)

    # Step 7: Sum of absolute correlations and Vj
    sum_abs_corr = corr.abs().sum(axis=1).rename("SumAbsCorr")  # includes self = 1
    Vj = (sigma / sum_abs_corr).rename("RedundancyAdjVar")

    # Step 8: Normalized components Rbar and Vbar
    Rbar = (Rj / Rj.sum()).rename("Rbar")
    Vbar = (Vj / Vj.sum()).rename("Vbar")

    # Step 9: Final weights W
    W = (alpha * Rbar + (1.0 - alpha) * Vbar).rename("Weight")
    W = W / W.sum()

    summary = pd.concat([Rj, Vj, Rbar, Vbar, W], axis=1)
    summary.index.name = "Criterion"

    artifacts = {
        "R": R,
        "P": P,
        "P_minus": P_minus,
        "D": D,
        "Rj": Rj,
        "sigma": sigma,
        "corr": corr,
        "sum_abs_corr": sum_abs_corr,
        "Vj": Vj,
        "Rbar": Rbar,
        "Vbar": Vbar,
        "W": W,
        "summary": summary
    }
    return artifacts

# -----------------------------
# Sample dataset helper
# -----------------------------
def make_sample_dataset():
    data = {
        "Benefit1": [70, 85, 90, 60, 75],
        "Benefit2": [150, 140, 160, 155, 145],
        "Cost1": [200, 180, 220, 210, 190],
        "Cost2": [15, 12, 18, 14, 13],
        "Target1": [50, 55, 52, 48, 60],
    }
    df = pd.DataFrame(data, index=[f"A{i+1}" for i in range(5)])
    return df

# -----------------------------
# UI
# -----------------------------
st.title("COREX Weighting - 9 Step App")

with st.sidebar:
    st.header("Configuration")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    alpha = st.slider("Blend parameter alpha", 0.0, 1.0, 0.5, 0.05)
    st.caption("alpha = 1 removal only, alpha = 0 redundancy only")
    use_sample = st.checkbox("Use sample dataset", value=False)

raw_df = None
if use_sample:
    raw_df = make_sample_dataset()
else:
    if file is not None:
        if file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)

if raw_df is None:
    st.info("Upload a file or use the sample dataset.")
    st.stop()

st.subheader("Raw data")
st.dataframe(raw_df, use_container_width=True)

with st.expander("Row identifiers"):
    idx_col = st.selectbox("Use this column as alternative names", ["<row number>"] + list(raw_df.columns), index=0)
if idx_col != "<row number>":
    raw_df = raw_df.set_index(idx_col)

num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.error("No numeric columns found.")
    st.stop()

with st.expander("Select criteria columns"):
    selected_cols = st.multiselect("Criteria", options=num_cols, default=num_cols)
if not selected_cols:
    st.error("Select at least one criterion.")
    st.stop()

df_vals = raw_df[selected_cols].copy()

st.subheader("Criterion types and targets")
if "crit_meta_full9" not in st.session_state or set(st.session_state["crit_meta_full9"].index) != set(selected_cols):
    st.session_state["crit_meta_full9"] = pd.DataFrame(
        {"Type": ["Benefit"]*len(selected_cols), "Target": [0.0]*len(selected_cols)},
        index=selected_cols
    )

meta = st.data_editor(
    st.session_state["crit_meta_full9"],
    column_config={
        "Type": st.column_config.SelectboxColumn(options=CORE_TYPES),
        "Target": st.column_config.NumberColumn(format="%.6f"),
    },
    use_container_width=True
)
st.session_state["crit_meta_full9"] = meta
crit_types = meta["Type"].to_dict()
targets = meta["Target"].astype(float).to_dict()

# Assumption checks
for c in df_vals.columns:
    x = df_vals[c]
    if crit_types[c] in ["Benefit", "Cost"]:
        if np.isclose(x.min(), x.max()):
            st.error(f"Criterion {c} has max equal to min. Adjust your data.")
            st.stop()
    else:
        d = (x - targets.get(c, 0.0)).abs()
        if np.isclose(d.max(), 0.0):
            st.error(f"All alternatives hit the exact target for {c}. Adjust your target.")
            st.stop()

if st.button("Compute COREX and show all 9 steps"):
    A = corex_9step(df_vals, crit_types, targets, alpha=alpha)
    st.success("COREX computed. See steps below.")

    with st.expander("Step 1 - Normalized matrix R", expanded=True):
        st.dataframe(A["R"].style.format("{:.6f}"), use_container_width=True)
        st.download_button("Download R (CSV)", data=A["R"].to_csv().encode("utf-8"),
                           file_name="step1_R.csv", mime="text/csv")

    with st.expander("Step 2 - Overall performance P", expanded=True):
        st.dataframe(A["P"].to_frame("P").style.format("{:.6f}"), use_container_width=True)
        st.download_button("Download P (CSV)", data=A["P"].to_frame("P").to_csv().encode("utf-8"),
                           file_name="step2_P.csv", mime="text/csv")

    with st.expander("Step 3 - Leave one out performance P^(−j) using 1/n", expanded=False):
        st.dataframe(A["P_minus"].style.format("{:.6f}"), use_container_width=True)
        st.download_button("Download P_minus (CSV)", data=A["P_minus"].to_csv().encode("utf-8"),
                           file_name="step3_Pminus.csv", mime="text/csv")

    with st.expander("Step 4 - Absolute drops D and removal impact Rj", expanded=True):
        st.write("D_ij = |P_i - P_i^(−j)|. Rj = sum_i D_ij.")
        st.dataframe(A["D"].style.format("{:.6f}"), use_container_width=True)
        st.dataframe(A["Rj"].to_frame().style.format("{:.6f}"), use_container_width=True)
        st.download_button("Download D (CSV)", data=A["D"].to_csv().encode("utf-8"),
                           file_name="step4_D.csv", mime="text/csv")
        st.download_button("Download Rj (CSV)", data=A["Rj"].to_frame().to_csv().encode("utf-8"),
                           file_name="step4_Rj.csv", mime="text/csv")

    with st.expander("Step 5 - Per criterion standard deviation sigma", expanded=True):
        st.dataframe(A["sigma"].to_frame().style.format("{:.6f}"), use_container_width=True)
        st.download_button("Download sigma (CSV)", data=A["sigma"].to_frame().to_csv().encode("utf-8"),
                           file_name="step5_sigma.csv", mime="text/csv")

    with st.expander("Step 6 - Pearson correlation on R", expanded=False):
        st.dataframe(A["corr"].style.format("{:.6f}"), use_container_width=True)
        st.download_button("Download correlation matrix (CSV)", data=A["corr"].to_csv().encode("utf-8"),
                           file_name="step6_corr.csv", mime="text/csv")

    with st.expander("Step 7 - Sum of absolute correlations and Vj", expanded=False):
        st.dataframe(A["sum_abs_corr"].to_frame().style.format("{:.6f}"), use_container_width=True)
        st.write("Vj = sigma_j divided by sum_k |rho_jk|, including self term.")
        st.dataframe(A["Vj"].to_frame().style.format("{:.6f}"), use_container_width=True)
        st.download_button("Download sum|rho| (CSV)", data=A["sum_abs_corr"].to_frame().to_csv().encode("utf-8"),
                           file_name="step7_sum_abs_corr.csv", mime="text/csv")
        st.download_button("Download Vj (CSV)", data=A["Vj"].to_frame().to_csv().encode("utf-8"),
                           file_name="step7_Vj.csv", mime="text/csv")

    with st.expander("Step 8 - Normalized components Rbar and Vbar", expanded=True):
        st.dataframe(pd.concat([A["Rbar"].to_frame(), A["Vbar"].to_frame()], axis=1).style.format("{:.6f}"),
                     use_container_width=True)
        st.download_button("Download Rbar (CSV)", data=A["Rbar"].to_csv().encode("utf-8"),
                           file_name="step8_Rbar.csv", mime="text/csv")
        st.download_button("Download Vbar (CSV)", data=A["Vbar"].to_csv().encode("utf-8"),
                           file_name="step8_Vbar.csv", mime="text/csv")

    with st.expander("Step 9 - Final COREX weights W", expanded=True):
        st.dataframe(A["W"].to_frame("Weight").style.format("{:.6f}"), use_container_width=True)
        fig = px.bar(A["W"].reset_index(), x="index", y="Weight")
        fig.update_layout(xaxis_title="Criterion", yaxis_title="Weight", bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download weights (CSV)", data=A["W"].to_frame("Weight").to_csv().encode("utf-8"),
                           file_name="step9_weights.csv", mime="text/csv")

    st.subheader("Summary table")
    st.dataframe(A["summary"].style.format("{:.6f}"), use_container_width=True)
    st.download_button("Download summary (CSV)", data=A["summary"].to_csv().encode("utf-8"),
                       file_name="summary_corex.csv", mime="text/csv")
else:
    st.info("Set types and targets, then click Compute to view all 9 steps.")
