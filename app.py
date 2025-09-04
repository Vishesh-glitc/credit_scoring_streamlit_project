# app.py
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from backend import fetch_company_report, generate_pdf

st.set_page_config(page_title="Credit Score APP", layout="wide")
st.markdown("""
<style>
  .chips {display:flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 .6rem 0;}
  .chip {display:inline-block; background:#f9fbff; border:1px solid #E6EAF2; border-radius: 999px; padding: .25rem .6rem; font-size: .9rem;}
  .stDownloadButton>button {height: 2.4rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
PLACEHOLDER = "Yet to be fetched"
def _fmt_cell(v):
    if v is None:
        return PLACEHOLDER
    if isinstance(v, str):
        if v.strip() == "" or v.strip().lower() == "empty":
            return PLACEHOLDER
        return v.strip()
    if isinstance(v, list):
        return ", ".join([str(x) for x in v]) if v else PLACEHOLDER
    return v

def ensure_columns(rows: List[Dict[str, Any]] | None, columns: List[str]) -> pd.DataFrame:
    """
    Build a DataFrame with exactly the given columns.
    If rows is empty/None, return an empty df with those columns.
    Missing keys are filled with None; extra keys are ignored.
    """
    rows = rows or []
    normalized = []
    for r in rows:
        normalized.append({c: _fmt_cell(r.get(c)) for c in columns})
    if not normalized:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(normalized, columns=columns)

def single_row_frame(obj: Dict[str, Any] | None, columns: List[str]) -> pd.DataFrame:
    """
    For objects (like company/riskScore), render a single-row dataframe with all requested columns,
    filling missing values with None.
    """
    obj = obj or {}
    return pd.DataFrame([{c: _fmt_cell(obj.get(c)) for c in columns}], columns=columns)

def section_header(title: str):
    st.markdown(f"### {title}")

def render_table(title: str, rows: List[Dict[str, Any]] | None, columns: List[str]):
    st.markdown(f"**{title}**")
    df = ensure_columns(rows, columns)
    if df.empty:
        df = pd.DataFrame([{c: PLACEHOLDER for c in columns}], columns=columns)
    df = df.replace(
        to_replace=[None, r'^\s*$', r'(?i)^\s*empty\s*$'],
        value=PLACEHOLDER,
        regex=True
    )
    df = df.map(_fmt_cell)
    if len(df) == 1 and all(val == PLACEHOLDER for val in df.iloc[0]):
        merged_row = {columns[0]: PLACEHOLDER}
        for col in columns[1:]:
            merged_row[col] = ""
        df = pd.DataFrame([merged_row], columns=columns)
    st.dataframe(df, width="stretch", hide_index=True)

def render_card_df(title: str, obj: Dict[str, Any] | None, columns: List[str]):
    st.markdown(f"**{title}**")
    df = single_row_frame(obj, columns)
    df = df.replace(
        to_replace=[None, r'^\s*$', r'(?i)^\s*empty\s*$'],
        value=PLACEHOLDER,
        regex=True
    )
    df = df.map(_fmt_cell)
    st.dataframe(df, width="stretch", hide_index=True)

# ---------------------------
# Top Bar
# ---------------------------
st.title("Credit Score APP")
default_company = "Elite Zone Ltd"  # default search text
cols = st.columns([5, 1])
with cols[0]:
    company_name = st.text_input("🔎 Search company", value=default_company, placeholder="Enter a company name")
with cols[1]:
    st.markdown("<div style='height: 1.9rem'></div>", unsafe_allow_html=True)
    generate = st.button("Generate Report 🚀", type="primary", width="stretch")

if generate or company_name:
    data = fetch_company_report(company_name.strip())

    # ----- Report header: logo + name + single PDF download button -----
    comp_obj = data.get("company") or {}
    logo_url = data.get("logoUrl")
    display_name = comp_obj.get("name") or company_name.strip() or "Company"
    website = comp_obj.get("website")
    pdf_filename = f"{display_name.replace(' ', '_')}_report.pdf"

    # Pre-generate the PDF so we can offer a single download button (no extra click)
    try:
        _pdf_path = generate_pdf(data)
    except Exception:
        _pdf_path = None

    header_cols = st.columns([0.9, 6.5, 2.6])
    with header_cols[0]:
        if logo_url:
            st.image(logo_url, caption="", width=88)
    with header_cols[1]:
        title_md = f"## {display_name} {'🌐' if website else ''}"
        st.markdown(title_md)
        if website:
            st.markdown(f"[{website}]({website})")
        desc = (comp_obj.get('description') or '').strip()
        if desc:
            st.markdown(f"_{desc}_")
        else:
            st.caption(PLACEHOLDER)
    with header_cols[2]:
        if _pdf_path:
            with open(_pdf_path, "rb") as _pf:
                st.download_button("Download PDF 📄", _pf, file_name=pdf_filename, width="stretch")
        else:
            st.caption("PDF unavailable")

    # -----------------------------------------
    # 📌 Company & Core Info
    # -----------------------------------------
    st.divider()
    section_header("📌 Company & Core Info")
    # Single chip row for all Company & Core Info (no tables)
    comp_obj = data.get("company") or {}
    chips = [
        ("🏷️ Industry", comp_obj.get("industry")),
        ("📍 Country", comp_obj.get("country")),
        ("🗺️ HQ", comp_obj.get("headquarters")),
        ("📅 Founded", comp_obj.get("foundedYear")),
        ("🧾 VAT ID", comp_obj.get("vatId")),
        ("🏛️ Legal Name", comp_obj.get("legalName")),
        ("🆔 Registration ID", comp_obj.get("registrationId")),
        ("👥 Employees", comp_obj.get("employeeCount")),
        ("🌐 Countries", comp_obj.get("countries")),
    ]
    chip_html = "".join([f"<span class='chip'><b>{k}</b>: {_fmt_cell(v)}</span>" for k, v in chips])
    st.markdown(f"<div class='chips'>{chip_html}</div>", unsafe_allow_html=True)

    # -----------------------------------------
    # 📌 Risk & Alerts
    # -----------------------------------------
    st.divider()
    section_header("📌 Risk & Alerts")
    risk_cols = ["scoredAt","score","level","financial","payments","news","industryRisk","geoRisk","delta","pd"]
    render_card_df("RiskScore", data.get("riskScore"), risk_cols)
    alert_cols = ["type","priority","createdAt","message","resolvedAt"]
    render_table("Alert", data.get("alerts"), alert_cols)
    alert_cfg_cols = ["name","minScore","maxScore","scoreDrop","newsFloor","dpdThreshold","priority","isActive","createdAt","updatedAt"]
    render_table("AlertConfig", data.get("alertConfigs"), alert_cfg_cols)

    # -----------------------------------------
    # 📌 Benchmarks
    # -----------------------------------------
    st.divider()
    section_header("📌 Benchmarks")
    ind_bm_cols = ["industry","country","avgRevenue","avgMargin","avgD2E","updatedAt"]
    comp_bm_cols = ["benchmarkId"]
    render_table("IndustryBenchmark", data.get("benchmarks"), ind_bm_cols)
    render_table("CompanyBenchmark", [data.get("companyBenchmark")] if data.get("companyBenchmark") else [], comp_bm_cols)

    # -----------------------------------------
    # 📌 News & Sentiment
    # -----------------------------------------
    st.divider()
    section_header("📌 News & Sentiment")

    # Overall sentiment chip (from risk.news mapped 0..10)
    _news_score = (data.get("riskScore") or {}).get("news")
    if isinstance(_news_score, (int, float)):
        if _news_score >= 7.0:
            chip = "<span class='chip' style='background:#e9f9ef;border-color:#b7ebc6'>📈 Overall sentiment: <b>Positive</b></span>"
        elif _news_score >= 4.0:
            chip = "<span class='chip' style='background:#f1f5f9;border-color:#E6EAF2'>😐 Overall sentiment: <b>Neutral</b></span>"
        else:
            chip = "<span class='chip' style='background:#fff1f0;border-color:#ffccc7'>⚠️ Overall sentiment: <b>Negative</b></span>"
        st.markdown(f"<div class='chips'>{chip}</div>", unsafe_allow_html=True)

    _rows = data.get("newsSentiment") or []

    def _art_row(r: Dict[str, Any]):
        return {
            "Date": r.get("observedAt"),
            "Headline": r.get("headline"),
            "Source": r.get("source"),
            "Link": r.get("url"),
        }

    def _sent_row(r: Dict[str, Any]):
        sc = r.get("score", 0)
        lbl = r.get("label") or ("Positive" if sc > 0 else ("Neutral" if sc == 0 else "Negative"))
        emj = r.get("emoji") or ("📈" if sc > 0 else ("😐" if sc == 0 else "⚠️"))
        return {
            "Date": r.get("observedAt"),
            "Sentiment": f"{emj} {lbl}",
            "Headline": r.get("headline"),
            "Source": r.get("source"),
            "Link": r.get("url"),
        }

    article_rows = [_art_row(r) for r in _rows] if _rows else []
    sentiment_rows = [_sent_row(r) for r in _rows] if _rows else []

    render_table("Latest Articles", article_rows, ["Date","Headline","Source","Link"])
    render_table("Sentiment (per headline)", sentiment_rows, ["Date","Sentiment","Headline","Source","Link"])

    # (Optional placeholder for social signals; still rendered via generic table for now)
    social_cols = ["platform","sentiment","content","observedAt","author","url","language","ingestedAt"]
    render_table("SocialSignal", data.get("socialSignals"), social_cols)

    # -----------------------------------------
    # 📌 Financials & Payments
    # -----------------------------------------
    st.divider()
    section_header("📌 Financials & Payments")
    fin_cols = ["periodStart","periodEnd","revenue","profitMargin","debtToEquity","currentRatio"]
    pay_cols = ["invoiceDate","dueDate","paidDate","amount","status","dpd"]
    render_table("FinancialMetric", data.get("financials"), fin_cols)
    render_table("Payment", data.get("payments"), pay_cols)

    # -----------------------------------------
    # 📌 Documents & Notes
    # -----------------------------------------
    st.divider()
    section_header("📌 Documents & Notes")
    doc_cols = ["title","url","kind","uploadedAt"]
    note_cols = ["author","text","createdAt"]
    render_table("Document", data.get("documents"), doc_cols)
    render_table("Note", data.get("notes"), note_cols)

    # -----------------------------------------
    # 📌 Data Sources
    # -----------------------------------------
    st.divider()
    section_header("📌 Data Sources")
    ds_cols = ["type","name","endpoint","createdAt"]
    job_cols = ["dataSourceId","startedAt","endedAt","status","message"]
    render_table("DataSource", data.get("dataSources"), ds_cols)
    render_table("IngestionJob", data.get("ingestionJobs"), job_cols)

    # -----------------------------------------
    # 📌 Credit Applications
    # -----------------------------------------
    st.divider()
    section_header("📌 Credit Applications")
    app_cols = ["applicant","email","amount","status","createdAt"]
    render_table("CreditApplication", data.get("creditApplications"), app_cols)

    # -----------------------------------------
    # 📌 Registries, Tax IDs & Filings
    # -----------------------------------------
    st.divider()
    section_header("📌 Registries, Tax IDs & Filings")
    tax_cols = ["type","value","country","verified","addedAt"]
    reg_cols = ["source","country","registryId","vatId","foundedAt","legalForm","status","address","url","fetchedAt"]
    filing_cols = ["source","periodEnd","currency","revenue","profit","ebitda","debt","equity","assets","liabilities","marginPct","url","ingestedAt"]
    render_table("CompanyTaxId", data.get("taxIds"), tax_cols)
    render_table("CompanyRegistry", data.get("registries"), reg_cols)
    render_table("FilingSnapshot", data.get("filings"), filing_cols)

    # -----------------------------------------
    # 📌 Public Records & Litigation
    # -----------------------------------------
    st.divider()
    section_header("📌 Public Records & Litigation")
    pr_cols = ["type","detail","recordedAt","source","country","url","ingestedAt"]
    lit_cols = ["court","caseNumber","filedAt","status","jurisdiction","parties","url","updatedAt","ingestedAt"]
    render_table("PublicRecord", data.get("publicRecords"), pr_cols)
    render_table("LitigationRecord", data.get("litigation"), lit_cols)

    # -----------------------------------------
    # 📌 Macro Indicators
    # -----------------------------------------
    st.divider()
    section_header("📌 Macro Indicators")
    macro_cols = ["country","indicator","period","value","unit","source","observedAt"]
    render_table("MacroIndicator", data.get("macro"), macro_cols)

    # -----------------------------------------
    # Actions
    # -----------------------------------------
    # (PDF actions UI removed; replaced by single top-level PDF download button)
else:
    st.info("Enter a company name and click **Generate Report** to build a report.")