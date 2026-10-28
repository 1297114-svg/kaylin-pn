"""
Hispanic Heritage Historical Data Explorer
A self-contained Streamlit app for historical World Bank data and function analysis.

Important: This code is provided as a complete starting implementation. Network/API behavior
and deployment must be verified in the target environment before claiming production testing.
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📈", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": {"code": "SP.DYN.LE00.IN", "unit": "years", "definition": "Expected years a newborn would live if current age-specific mortality rates continued."},
    "Unemployment, national estimate": {"code": "SL.UEM.TOTL.NE.ZS", "unit": "% of labor force", "definition": "Share of the labor force without work but available for and seeking employment, using national estimates."},
    "GDP per capita": {"code": "NY.GDP.PCAP.CD", "unit": "current US$ per person", "definition": "Gross domestic product divided by midyear population; not average wealth or household income."},
    "Crude birth rate": {"code": "SP.DYN.CBRT.IN", "unit": "births per 1,000 people per year", "definition": "Annual live births per 1,000 people; it is a rate, not a birth count."},
    "Net migration": {"code": "SM.POP.NETM", "unit": "people (net estimate)", "definition": "Net number of migrants over a period; it is not an emigration count."},
    "Intentional homicide rate": {"code": "VC.IHR.PSRC.P5", "unit": "per 100,000 people", "definition": "Intentional homicide victims per 100,000 population."},
    "Consumer price inflation": {"code": "FP.CPI.TOTL.ZG", "unit": "% annual", "definition": "Annual percentage change in the consumer price index."},
    "Central government debt": {"code": "GC.DOD.TOTL.GD.ZS", "unit": "% of GDP", "definition": "Central government debt as a share of gross domestic product."},
    "Total reserves including gold": {"code": "FI.RES.TOTL.CD", "unit": "current US$", "definition": "Total reserves including gold, reported in current US dollars."},
    "External balance on goods and services": {"code": "NE.RSB.GNFS.ZS", "unit": "% of GDP", "definition": "External balance on goods and services as a share of GDP."},
}
CONTEXTS = {
    "Business": ("Revenue", "currency units", "Time", "years", "Revenue changes over time. The derivative is revenue change per year."),
    "Engineering": ("Position", "meters", "Time", "seconds", "The first derivative is velocity (m/s); the second is acceleration (m/s²)."),
    "Medical": ("Concentration", "mg/L", "Time", "hours", "The function can represent concentration; derivatives describe how quickly concentration changes."),
    "Sports science": ("Distance", "meters", "Time", "seconds", "The first derivative is speed/velocity; the second describes acceleration."),
    "Biology": ("Population", "organisms", "Time", "days", "The first derivative is population change per day; the second shows whether that rate is increasing or decreasing."),
    "Bioengineering": ("Signal", "signal units", "Time", "seconds", "Derivatives describe the rate of signal change and how that rate changes."),
    "Economic forecasting": ("Indicator", "index units", "Year", "years", "The derivative is annual change; the second derivative describes acceleration or slowdown in that change."),
    "Weather": ("Temperature", "°C", "Time", "hours", "The first derivative is temperature change per hour; the second is the change in that rate."),
    "Custom": ("Output", "units", "Input", "units", "Interpret the function and derivatives using the variable names and units you choose."),
}
SUGGESTIONS = {
    "Business": "a*(x-h)**2+k",
    "Engineering": "v*x + p",
    "Medical": "a*exp(-k*x)",
    "Sports science": "a*sin(w*x)",
    "Biology": "a*exp(k*x)",
    "Bioengineering": "a*sin(w*x)+b",
    "Economic forecasting": "a*x+b",
    "Weather": "a*sin(w*x)+b",
    "Custom": "a*x**2+b*x+c",
}

def world_bank_data(country_code: str, indicator_code: str) -> pd.DataFrame:
    """Fetch up to 70 prior years from the World Bank API, with clear failures."""
    end_year = date.today().year - 1
    start_year = date.today().year - 70
    url = (f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
           f"{indicator_code}?date={start_year}:{end_year}&format=json&per_page=1000")
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return pd.DataFrame(columns=["year", "value"])
    rows = []
    for item in payload[1]:
        if item.get("date") is not None and item.get("value") is not None:
            try:
                rows.append({"year": int(item["date"]), "value": float(item["value"])})
            except (TypeError, ValueError):
                continue
    frame = pd.DataFrame(rows, columns=["year", "value"])
    if not frame.empty:
        frame = frame.sort_values("year").drop_duplicates("year", keep="last").reset_index(drop=True)
    return frame

@st.cache_data(ttl=3600, show_spinner=False)
def cached_world_bank_data(country_code: str, indicator_code: str) -> pd.DataFrame:
    return world_bank_data(country_code, indicator_code)

def parse_expr(text: str, x_symbol):
    """Parse a student-entered expression with a restricted set of math functions."""
    if not text.strip():
        raise ValueError("Enter a function, such as a*x**2+b*x+c.")
    if len(text) > 180:
        raise ValueError("Function text is too long.")
    allowed = {
        "x": x_symbol, "a": sp.Symbol("a"), "b": sp.Symbol("b"), "c": sp.Symbol("c"),
        "d": sp.Symbol("d"), "k": sp.Symbol("k"), "h": sp.Symbol("h"),
        "w": sp.Symbol("w"), "p": sp.Symbol("p"),
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
        "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E,
        "abs": sp.Abs,
    }
    # Prevent attribute access, imports, indexing, and other Python expressions.
    if re.search(r"__|[{}\[\];]|import|lambda|open\(|exec\(|eval\(|\.|\b(?:os|sys|subprocess)\b", text):
        raise ValueError("Use mathematical notation only (for example sin(x), exp(x), or x**2).")
    try:
        expr = sp.sympify(text.replace("^", "**"), locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not read that function. Try notation like a*x**2+b*x+c.") from exc
    if not expr.free_symbols.issubset({x_symbol, *[sp.Symbol(k) for k in ["a","b","c","d","k","h","w","p"]]}):
        raise ValueError("The function contains an unsupported variable.")
    return expr

def numeric_function(expr, x_symbol, params):
    substitutions = {sp.Symbol(k): float(v) for k, v in params.items()}
    resolved = sp.simplify(expr.subs(substitutions))
    fn = sp.lambdify(x_symbol, resolved, modules=["numpy"])
    def evaluate(x):
        with np.errstate(all="ignore"):
            y = np.asarray(fn(np.asarray(x, dtype=float)), dtype=float)
        return y
    return resolved, evaluate

def safe_values(fn, xs):
    try:
        vals = np.asarray(fn(xs), dtype=float)
        if vals.ndim == 0:
            vals = np.full_like(np.asarray(xs, dtype=float), float(vals))
        return np.where(np.isfinite(vals), vals, np.nan)
    except Exception:
        return np.full_like(np.asarray(xs, dtype=float), np.nan)

def fit_polynomial(frame: pd.DataFrame, degree: int):
    clean = frame[["year", "value"]].dropna().copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna().sort_values("year")
    if clean["year"].duplicated().any():
        raise ValueError("Years must be unique.")
    if not np.all(np.equal(clean["year"], np.floor(clean["year"]))):
        raise ValueError("Years must be integers.")
    if len(clean) < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years; only {len(clean)} are available.")
    base = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base
    y = clean["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    poly = np.poly1d(coeff)
    predicted = poly(t)
    return {"frame": clean, "base": base, "coeff": coeff, "poly": poly,
            "predicted": predicted, "residuals": y-predicted,
            "r2": float(r2_score(y, predicted)), "mae": float(mean_absolute_error(y, predicted))}

def make_csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")

def make_excel_bytes(summary, working, results):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        working.to_excel(writer, sheet_name="Working data", index=False)
        results.to_excel(writer, sheet_name="Calculated years", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
            sheet.set_column(0, 10, 18)
    return output.getvalue()

def make_pdf_bytes(summary_text, chart_png, preview, results):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]), Spacer(1, 8)]
    for line in summary_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
            story.append(Spacer(1, 4))
    story += [Spacer(1, 8), RLImage(io.BytesIO(chart_png), width=480, height=280), Spacer(1, 8)]
    for title, frame in [("Data preview", preview), ("Calculated years", results)]:
        story.append(Paragraph(title, styles["Heading2"]))
        rows = [list(map(str, frame.columns))] + frame.head(15).astype(str).values.tolist()
        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID", (0,0), (-1,-1), .35, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP")
        ]))
        story.append(table)
        story.append(Spacer(1, 10))
    doc.build(story)
    return output.getvalue()

st.title("📈 Hispanic Heritage Historical Data Explorer")
st.caption("Explore official historical indicators, fit models, and investigate functions across school mathematics pathways.")
with st.expander("About the data and Brazil comparison"):
    st.write("Historical indicators come from the official World Bank API. Brazil is included as a Latin American comparison alongside the Spanish-speaking countries listed.")
    st.write("The app requests the previous 70 calendar years, ending last year. Missing years are normal for some indicators; the app reports actual coverage and never fills missing values with invented data.")
    st.markdown("[World Bank API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation)")

left, right = st.columns(2)
with left:
    country_label = st.selectbox("Country", list(COUNTRIES), index=0)
with right:
    indicator_label = st.selectbox("Historical indicator", list(INDICATORS), index=0)
country_code = COUNTRIES[country_label]
indicator = INDICATORS[indicator_label]
source_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator['code']}?format=json"

try:
    with st.spinner("Retrieving World Bank observations..."):
        source_frame = cached_world_bank_data(country_code, indicator["code"])
    retrieval_message = f"Retrieved on {date.today().isoformat()}."
    if source_frame.empty:
        st.warning("The API returned no usable observations for this country and indicator. Try another selection or check the World Bank API status.")
except requests.RequestException as exc:
    source_frame = pd.DataFrame(columns=["year", "value"])
    retrieval_message = f"Retrieval failed: {exc.__class__.__name__}. Check your connection or the World Bank API and retry."
    st.error(retrieval_message)
except Exception as exc:
    source_frame = pd.DataFrame(columns=["year", "value"])
    retrieval_message = f"Could not process the API response ({exc.__class__.__name__})."
    st.error(retrieval_message)

st.markdown(f"**Definition:** {indicator['definition']}  \n**Unit:** {indicator['unit']}  \n**Indicator code:** `{indicator['code']}`  \n**Source:** [World Bank indicator API]({source_url})  \n**{retrieval_message}**")
if not source_frame.empty:
    st.caption(f"Actual coverage: {int(source_frame.year.min())}–{int(source_frame.year.max())}; {len(source_frame)} non-missing annual observations.")
else:
    st.caption("Actual coverage: no non-missing annual observations available.")

if "working_data" not in st.session_state:
    st.session_state.working_data = source_frame.copy()
if st.session_state.get("data_key") != (country_code, indicator["code"]):
    st.session_state.working_data = source_frame.copy()
    st.session_state.data_key = (country_code, indicator["code"])
    st.session_state.pop("calculated_years", None)
working = st.session_state.working_data.copy()

st.subheader("1. Review and edit data")
st.write("Edit the year/value table if you want to test a scenario. Edited values are separate from the source data.")
edited = st.data_editor(working, num_rows="dynamic", use_container_width=True, key="data_editor")
valid_data = True
validation_errors = []
if not edited.empty:
    numeric_year = pd.to_numeric(edited.get("year"), errors="coerce")
    numeric_value = pd.to_numeric(edited.get("value"), errors="coerce")
    if numeric_year.isna().any() or not np.equal(numeric_year.dropna(), np.floor(numeric_year.dropna())).all():
        validation_errors.append("Every year must be a whole number.")
    if numeric_year.dropna().duplicated().any():
        validation_errors.append("Duplicate years are not allowed.")
    if numeric_value.isna().any():
        validation_errors.append("Every value must be numeric; remove blank rows or enter a number.")
    if indicator_label in ["Life expectancy at birth", "Unemployment, national estimate", "Intentional homicide rate", "Crude birth rate"] and (numeric_value.dropna() < 0).any():
        validation_errors.append("This indicator should not have negative values.")
    if indicator_label == "Unemployment, national estimate" and (numeric_value.dropna() > 100).any():
        validation_errors.append("Unemployment percentage cannot exceed 100.")
    if indicator_label == "Life expectancy at birth" and (numeric_value.dropna() > 125).any():
        validation_errors.append("Life expectancy value exceeds the app's plausibility check (125 years).")
if validation_errors:
    valid_data = False
    for error in validation_errors:
        st.error(error)
if st.button("Reset to source data"):
    st.session_state.working_data = source_frame.copy()
    st.session_state.pop("calculated_years", None)
    st.rerun()
if not edited.equals(working):
    st.session_state.working_data = edited.copy()
    st.session_state["data_was_edited"] = True
else:
    st.session_state.setdefault("data_was_edited", False)
data_status = "USER-MODIFIED SCENARIO" if st.session_state.get("data_was_edited", False) and not edited.equals(source_frame) else "Source data"
st.caption(f"Data status: **{data_status}**. Source data remains available separately for reset.")

st.subheader("2. Fit a historical polynomial")
c1, c2, c3 = st.columns(3)
with c1:
    degree = st.selectbox("Polynomial degree", list(range(3, 9)), index=3)
with c2:
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 3)
with c3:
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
comparison_label = st.selectbox("Comparison country (same indicator)", ["None"] + [c for c in COUNTRIES if c != country_label])
printer_mode = st.checkbox("Printer-friendly report view")

fit = None
if valid_data and not edited.empty:
    try:
        fit = fit_polynomial(edited, degree)
    except ValueError as exc:
        st.error(str(exc))
if fit:
    frame = fit["frame"]
    base = fit["base"]
    coeff = fit["coeff"]
    poly = fit["poly"]
    year_min, year_max = int(frame.year.min()), int(frame.year.max())
    st.markdown(f"**Model:** \(t = year - {base}\), degree {degree}; fit uses centered time.")
    terms = []
    for power, coefficient in enumerate(coeff[::-1]):
        if abs(coefficient) < 1e-12:
            continue
        if power == 0:
            terms.append(f"{coefficient:.8g}")
        elif power == 1:
            terms.append(f"{coefficient:.8g}t")
        else:
            terms.append(f"{coefficient:.8g}t^{power}")
    equation = " + ".join(terms).replace("+ -", "- ")
    st.code(f"y = {equation}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (in-sample)", f"{fit['r2']:.4f}")
    m2.metric("MAE (in-sample)", f"{fit['mae']:.4g} {indicator['unit']}")
    m3.metric("Observations", len(frame))
    m4.metric("Historical mean", f"{frame.value.mean():.5g} {indicator['unit']}")
    st.caption(f"Historical fitted range: {year_min}–{year_max}. Residuals are actual minus fitted values. These are in-sample metrics, not evidence that extrapolation is reliable.")
    residual_table = frame.copy()
    residual_table["fitted"] = fit["predicted"]
    residual_table["residual"] = fit["residuals"]
    with st.expander("Residuals and observation-level fitted values"):
        st.dataframe(residual_table, use_container_width=True)
    if len(frame) >= degree + 4:
        holdout_n = max(1, min(5, len(frame)//5))
        train = frame.iloc[:-holdout_n]
        test = frame.iloc[-holdout_n:]
        if len(train) >= degree + 1:
            hold_coeff = np.polyfit(train.year.to_numpy()-base, train.value.to_numpy(), degree)
            hold_pred = np.polyval(hold_coeff, test.year.to_numpy()-base)
            st.write(f"Chronological holdout MAE (last {holdout_n} observations; separately fitted on earlier years): **{mean_absolute_error(test.value, hold_pred):.5g}**. This score is not the same as the in-sample MAE.")

    # Single and batch prediction inputs
    st.subheader("3. Calculate years")
    p1, p2 = st.columns([1, 2])
    with p1:
        single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(year_max, 1950), 2100), step=1)
        calculate_single = st.button("Calculate Value")
    with p2:
        batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
        if batch_mode == "Year range":
            b1, b2, b3 = st.columns(3)
            with b1: batch_start = st.number_input("Start year", 1950, 2100, min(max(year_min,1950),2100), 1)
            with b2: batch_end = st.number_input("End year", 1950, 2100, min(max(year_max,1950),2100), 1)
            with b3: batch_step = st.selectbox("Range step", list(range(1,11)), index=0)
            custom_text = ""
        else:
            custom_text = st.text_input("Comma-separated years", value=f"{year_min},{year_max}")
            batch_start, batch_end, batch_step = 0, 0, 1
        calculate_batch = st.button("Calculate batch")
    if calculate_single:
        st.session_state.single_result = int(single_year)
    if calculate_batch:
        try:
            if batch_mode == "Year range":
                if batch_start > batch_end:
                    raise ValueError("Start year must be less than or equal to end year.")
                years = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
            else:
                tokens = [x.strip() for x in custom_text.split(",") if x.strip()]
                if not tokens:
                    raise ValueError("Enter at least one year.")
                if any(not re.fullmatch(r"\d{4}", token) for token in tokens):
                    raise ValueError("Use comma-separated whole years, such as 1990, 2000, 2010.")
                years = [int(token) for token in tokens]
                if len(set(years)) != len(years):
                    raise ValueError("Custom years must not contain duplicates.")
                if any(y < 1950 or y > 2100 for y in years):
                    raise ValueError("Every requested year must be between 1950 and 2100.")
            st.session_state.calculated_years = years
            st.session_state.calc_signature = (country_code, indicator["code"], degree, tuple(frame.year), tuple(frame.value))
        except ValueError as exc:
            st.error(str(exc))
    signature = (country_code, indicator["code"], degree, tuple(frame.year), tuple(frame.value))
    if st.session_state.get("calc_signature") != signature:
        st.session_state.pop("calculated_years", None)
        st.session_state.pop("single_result", None)
    def predict_years(years):
        values = poly(np.asarray(years, dtype=float)-base)
        rows = []
        for year, value in zip(years, np.atleast_1d(values)):
            actual_rows = frame.loc[frame.year == year, "value"]
            rows.append({"year": int(year), "estimate": float(value), "unit": indicator["unit"],
                         "classification": "within fitted range" if year_min <= year <= year_max else "outside fitted range",
                         "actual observation": float(actual_rows.iloc[0]) if len(actual_rows) else np.nan,
                         "observed year?": bool(len(actual_rows))})
        return pd.DataFrame(rows)
    if st.session_state.get("calc_signature") == signature and "single_result" in st.session_state:
        single_df = predict_years([st.session_state.single_result])
        st.write("**Single-year estimate**")
        st.dataframe(single_df, use_container_width=True)
        if st.session_state.single_result < year_min or st.session_state.single_result > year_max:
            st.warning("This is extrapolation outside the observed fitted range. Polynomial extrapolations can become implausible; values are not clipped.")
    result_years = st.session_state.get("calculated_years", [])
    result_df = predict_years(result_years) if result_years and st.session_state.get("calc_signature") == signature else pd.DataFrame(columns=["year","estimate","unit","classification","actual observation","observed year?"])
    if not result_df.empty:
        st.write("**Batch estimates**")
        st.dataframe(result_df, use_container_width=True)

    # Plot model, observed points, boundary and sample markers
    plot_min = min(year_min, int(min(result_years)) if result_years else year_min)
    plot_max = max(year_max + horizon, int(max(result_years)) if result_years else year_max)
    xs = np.linspace(plot_min, plot_max, max(300, (plot_max-plot_min)*4+1))
    ys = poly(xs-base)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(frame.year, frame.value, label="Observed / edited data", s=22)
    ax.plot(xs, ys, label="Smooth polynomial fit")
    sample_years = np.arange(year_min, year_max+1, sample_step)
    ax.scatter(sample_years, poly(sample_years-base), marker="x", label=f"Model markers every {sample_step} year(s)")
    ax.axvline(year_max, linestyle="--", linewidth=1, label="Historical boundary")
    if year_max < plot_max:
        ax.axvspan(year_max, plot_max, alpha=.08, label="Extrapolated display")
    if not result_df.empty:
        ax.scatter(result_df.year, result_df.estimate, marker="D", s=45, label="Batch estimates")
    if "single_result" in st.session_state and st.session_state.get("calc_signature") == signature:
        sy = int(st.session_state.single_result)
        ax.scatter([sy], [float(poly(sy-base))], marker="*", s=130, label="Single-year estimate")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_label} ({indicator['unit']})")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
    chart_png = chart_buffer.getvalue()
    plt.close(fig)

    if comparison_label != "None":
        try:
            compare_code = COUNTRIES[comparison_label]
            comp = cached_world_bank_data(compare_code, indicator["code"])
            if comp.empty:
                st.warning(f"No usable comparison observations for {comparison_label}.")
            else:
                joined = pd.merge(frame, comp, on="year", how="outer", suffixes=(" (selected)", " (comparison)")).sort_values("year")
                st.subheader(f"Country comparison: {country_label} and {comparison_label}")
                comp_fig, comp_ax = plt.subplots(figsize=(10, 4))
                comp_ax.plot(frame.year, frame.value, marker="o", label=country_label)
                comp_ax.plot(comp.year, comp.value, marker="o", label=comparison_label)
                comp_ax.set_xlabel("Year")
                comp_ax.set_ylabel(f"{indicator_label} ({indicator['unit']})")
                comp_ax.grid(True, alpha=.25)
                comp_ax.legend()
                st.pyplot(comp_fig, use_container_width=True)
                st.caption(f"Selected coverage: {year_min}–{year_max} ({len(frame)} observations); comparison coverage: {int(comp.year.min())}–{int(comp.year.max())} ({len(comp)} observations). Gaps mean the API did not provide a non-missing value for that year.")
                plt.close(comp_fig)
        except Exception as exc:
            st.warning(f"Comparison could not be loaded: {exc.__class__.__name__}.")

    st.subheader("4. Function lab")
    pathway = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)
    context = st.selectbox("Context", list(CONTEXTS), index=0)
    suggested_y, suggested_unit, suggested_x, suggested_xunit, default_description = CONTEXTS[context]
    cdesc = st.text_input("Context description", value=default_description)
    cvars = st.columns(4)
    with cvars[0]: x_name = st.text_input("Independent variable", value=suggested_x)
    with cvars[1]: x_unit = st.text_input("Input units", value=suggested_xunit)
    with cvars[2]: y_name = st.text_input("Dependent variable", value=suggested_y)
    with cvars[3]: y_unit = st.text_input("Output units", value=suggested_unit)
    function_source = st.radio("Function source", ["Historical curve of best fit", "Suggested function", "Build with parameters", "Enter my own function"], horizontal=True)
    param_cols = st.columns(4)
    params = {}
    for col, name, default in zip(param_cols, ["a","b","c","k"], [1.0, 0.0, 0.0, 0.1]):
        with col:
            params[name] = st.number_input(f"Parameter {name}", value=float(default), format="%.4f", key=f"param_{name}_{context}")
    if function_source == "Historical curve of best fit":
        # Re-express fitted polynomial in x as years from the model's base year.
        x = sp.Symbol("x", real=True)
        expr = sum(sp.Float(float(coef), 12) * (x-base)**(degree-i) for i, coef in enumerate(coeff))
        expr_label = f"Historical fitted polynomial where x is calendar year"
    elif function_source in ("Suggested function", "Build with parameters"):
        x = sp.Symbol("x", real=True)
        template = SUGGESTIONS[context]
        expr = parse_expr(template, x)
        expr_label = f"{context} suggested model"
    else:
        x = sp.Symbol("x", real=True)
        custom_expr = st.text_input("Your function in x", value="a*x**2+b*x+c", help="Allowed: + - * / **, parentheses, sin, cos, tan, exp, log, sqrt, pi. Use x for input and a,b,c,k,h,w,p for parameters.")
        expr = parse_expr(custom_expr, x)
        expr_label = "Student-entered function"
    try:
        resolved_expr, fn = numeric_function(expr, x, params)
        st.latex("f(x) = " + sp.latex(resolved_expr))
        if pathway == "Calculus":
            d1 = sp.diff(resolved_expr, x)
            d2 = sp.diff(d1, x)
            st.latex("f'(x) = " + sp.latex(d1))
            st.latex("f''(x) = " + sp.latex(d2))
        else:
            st.info(f"{pathway} focus: graph shape, domain/range, end behavior and rates of change. Calculus derivatives are available by selecting Calculus.")
        st.caption(f"Model: {expr_label}. {cdesc}")
        lo_default = float(year_min if function_source == "Historical curve of best fit" else 0)
        hi_default = float(min(year_max, lo_default+20) if function_source == "Historical curve of best fit" else 10)
        lo_default = min(lo_default, hi_default-0.1)
        interval_cols = st.columns(3)
        with interval_cols[0]:
            x_min = st.number_input("Analysis interval start", value=lo_default, key=f"xmin_{context}_{function_source}")
        with interval_cols[1]:
            x_max = st.number_input("Analysis interval end", value=max(hi_default, lo_default+1), key=f"xmax_{context}_{function_source}")
        with interval_cols[2]:
            x_point = st.number_input("Current input value", value=(lo_default+max(hi_default,lo_default+1))/2, key=f"xpoint_{context}_{function_source}")
        if x_min >= x_max:
            st.error("The analysis interval start must be less than its end.")
        else:
            grid = np.linspace(x_min, x_max, 700)
            ygrid = safe_values(fn, grid)
            show_f = st.checkbox("Show function f(x)", True)
            show_d1 = st.checkbox("Show first derivative", pathway == "Calculus")
            show_d2 = st.checkbox("Show second derivative", False)
            d1_fn = sp.lambdify(x, sp.diff(resolved_expr,x), modules=["numpy"])
            d2_fn = sp.lambdify(x, sp.diff(resolved_expr,x,2), modules=["numpy"])
            fig2, ax2 = plt.subplots(figsize=(10, 4.4))
            if show_f: ax2.plot(grid, ygrid, label="f(x)")
            if show_d1: ax2.plot(grid, safe_values(d1_fn, grid), label="f'(x)")
            if show_d2: ax2.plot(grid, safe_values(d2_fn, grid), label="f''(x)")
            f_point = float(safe_values(fn, np.array([x_point]))[0])
            slope = float(safe_values(d1_fn, np.array([x_point]))[0])
            if np.isfinite(f_point):
                ax2.scatter([x_point],[f_point], marker="o", s=55, label="Current point")
                tangent_x = np.linspace(x_min,x_max,150)
                if np.isfinite(slope):
                    ax2.plot(tangent_x, f_point+slope*(tangent_x-x_point), linestyle="--", label="Tangent line")
            f_lo = float(safe_values(fn,np.array([x_min]))[0])
            f_hi = float(safe_values(fn,np.array([x_max]))[0])
            if np.isfinite(f_lo) and np.isfinite(f_hi):
                secant_slope = (f_hi-f_lo)/(x_max-x_min)
                ax2.plot([x_min,x_max],[f_lo,f_hi], linestyle=":", linewidth=2, label="Secant line")
            else:
                secant_slope = np.nan
            ax2.set_xlabel(f"{x_name} ({x_unit})")
            ax2.set_ylabel(f"{y_name} ({y_unit})")
            ax2.grid(True, alpha=.25)
            ax2.legend(fontsize=8)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
            if np.isfinite(f_point):
                st.write(f"At {x_name} = {x_point:g} {x_unit}, **{y_name} ≈ {f_point:.6g} {y_unit}**.")
                st.write(f"Instantaneous rate: **f′(x) ≈ {slope:.6g} {y_unit}/{x_unit}**.")
            else:
                st.warning("The function is undefined or non-finite at the selected input.")
            if np.isfinite(secant_slope):
                st.write(f"Average rate of change on [{x_min:g}, {x_max:g}]: **{secant_slope:.6g} {y_unit}/{x_unit}**.")
            st.write(f"Second derivative at the selected point: **{float(safe_values(d2_fn,np.array([x_point]))[0]):.6g} {y_unit}/{x_unit}²** when defined.")
            st.caption("Function and derivative values are numerical approximations when evaluated numerically. Units are based on the labels you selected.")
            if pathway == "Calculus":
                # Numerical root discovery and sign checks on a finite interval.
                d1_vals = safe_values(d1_fn, grid)
                d2_vals = safe_values(d2_fn, grid)
                critical = []
                for i in range(len(grid)-1):
                    if np.isfinite(d1_vals[i]) and np.isfinite(d1_vals[i+1]) and d1_vals[i] == 0:
                        critical.append(float(grid[i]))
                    elif np.isfinite(d1_vals[i]) and np.isfinite(d1_vals[i+1]) and d1_vals[i]*d1_vals[i+1] < 0:
                        critical.append(float((grid[i]+grid[i+1])/2))
                critical = sorted({round(v,5) for v in critical})
                if critical:
                    rows=[]
                    for cp in critical:
                        delta=max((x_max-x_min)/1000,1e-5)
                        before=float(safe_values(d1_fn,np.array([cp-delta]))[0])
                        after=float(safe_values(d1_fn,np.array([cp+delta]))[0])
                        kind="local minimum (derivative changes − to +)" if before < 0 < after else "local maximum (derivative changes + to −)" if before > 0 > after else "stationary; no extremum classified"
                        rows.append({"x (approx.)":cp,"f(x) (approx.)":float(safe_values(fn,np.array([cp]))[0]),"classification":kind})
                    st.write("**Critical points (numerical sign-change search)**")
                    st.dataframe(pd.DataFrame(rows),use_container_width=True)
                else:
                    st.write("No derivative sign-change critical points detected on this interval; this numerical scan can miss repeated roots.")
                inflections=[]
                for i in range(len(grid)-1):
                    if np.isfinite(d2_vals[i]) and np.isfinite(d2_vals[i+1]) and d2_vals[i]*d2_vals[i+1] < 0:
                        mid=float((grid[i]+grid[i+1])/2)
                        inflections.append({"x (approx.)":mid,"f(x) (approx.)":float(safe_values(fn,np.array([mid]))[0]),"concavity changes":"likely; verify around point"})
                st.write("**Possible inflection points** (found by a numerical second-derivative sign-change scan)")
                if inflections: st.dataframe(pd.DataFrame(inflections),use_container_width=True)
                else: st.write("No concavity sign changes detected in the scan.")
                st.write("**Domain/discontinuity note:** the plot leaves non-finite values as gaps. For rational/logarithmic functions, inspect denominators and logarithm arguments; the numerical scan is not a formal proof.")
            else:
                finite = ygrid[np.isfinite(ygrid)]
                if finite.size:
                    st.write(f"On the displayed interval, approximate range: {np.nanmin(finite):.5g} to {np.nanmax(finite):.5g} {y_unit}.")
                    st.write("Average rates over shorter intervals can help describe whether the graph is increasing, decreasing, or changing its trend.")
    except Exception as exc:
        st.error(f"Function analysis could not be completed: {exc}. Check the function syntax and parameter values.")

    st.subheader("5. Export current results")
    summary = {
        "retrieval_date": date.today().isoformat(), "country": country_label, "country_code": country_code,
        "indicator": indicator_label, "indicator_code": indicator["code"], "unit": indicator["unit"],
        "data_status": data_status, "degree": degree, "base_year": base, "historical_start": year_min,
        "historical_end": year_max, "observations": len(frame), "r_squared_in_sample": fit["r2"],
        "mae_in_sample": fit["mae"], "historical_mean": float(frame.value.mean()), "equation": f"y = {equation}",
    }
    summary_text = (f"Country: {country_label}\nIndicator: {indicator_label}\nDefinition: {indicator['definition']}\n"
                    f"Unit: {indicator['unit']}\nRetrieved: {date.today().isoformat()}\nData status: {data_status}\n"
                    f"Coverage: {year_min}–{year_max}; observations: {len(frame)}\nEquation: y = {equation}, t = year - {base}\n"
                    f"Degree: {degree}; in-sample R²: {fit['r2']:.6f}; in-sample MAE: {fit['mae']:.6g}\n"
                    "Limitations: Polynomial fit describes the supplied data and can extrapolate implausibly. MAE is not a prediction interval.")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.download_button("Download working data CSV", make_csv_bytes(edited), file_name="working_data.csv", mime="text/csv")
    with ex2:
        excel = make_excel_bytes(summary, edited, result_df)
        st.download_button("Download Excel workbook", excel, file_name="hispanic_heritage_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with ex3:
        pdf = make_pdf_bytes(summary_text, chart_png, residual_table[["year","value","fitted","residual"]], result_df)
        st.download_button("Download PDF report", pdf, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
    if printer_mode:
        st.markdown("---")
        st.subheader("Printer-friendly summary")
        st.write(summary_text.replace("\n", "  \n"))
        st.dataframe(frame.tail(20), use_container_width=True)
else:
    st.info("A polynomial fit and exports will appear when the table contains enough valid distinct years for the selected degree.")

st.divider()
st.caption("Educational tool, not an official forecast. World Bank data can be revised, missing values are not imputed, and polynomial extrapolation is especially uncertain.")
