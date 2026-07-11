"""Shared CSS injected into every page for a consistent dark UI."""
import streamlit as st

_CSS = """
<style>
/* ── Layout ─────────────────────────────────────────────────────────────── */
.block-container {
  padding-top: 1.1rem !important;
  padding-bottom: 2rem !important;
  max-width: 1280px !important;
}

/* ── Typography ──────────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 { color: #38bdf8 !important; letter-spacing: -0.01em; }
h1 { font-size: 1.85rem !important; }
h2 { font-size: 1.45rem !important; }
h3 { font-size: 1.15rem !important; }
.stMarkdown p, .stMarkdown li { color: #94a3b8; }
label, .stSelectbox label, .stSlider label,
.stRadio label, .stTextInput label, .stNumberInput label { color: #7dd3fc !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: #64748b !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(175deg, #08111f 0%, #0d1b30 55%, #0f172a 100%) !important;
  border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] * { color: #7dd3fc !important; }
[data-testid="stSidebarNav"] a {
  font-size: 1.05rem !important;
  padding: 0.52rem 1rem !important;
  font-weight: 600 !important;
  border-radius: 7px !important;
  margin: 1px 0 !important;
  transition: background 0.14s, color 0.14s !important;
}
[data-testid="stSidebarNav"] a:hover { background: rgba(56,189,248,0.10) !important; }
[data-testid="stSidebarNav"] [aria-current="page"] {
  background: rgba(56,189,248,0.14) !important;
  color: #38bdf8 !important;
  border-left: 3px solid #0ea5e9 !important;
}
[data-testid="stSidebarNavItems"] { gap: 0.15rem !important; }

/* ── Metric cards ─────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: #0f1f38 !important;
  border: 1px solid #1e3a5f !important;
  border-radius: 10px !important;
  padding: 14px 18px !important;
  transition: border-color 0.15s !important;
}
[data-testid="metric-container"]:hover { border-color: #2563eb !important; }
.stMetric label {
  color: #38bdf8 !important;
  font-size: 0.76rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  font-weight: 700 !important;
}
.stMetric [data-testid="metric-container"] > div { color: #e0f2fe !important; font-weight: 700 !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  transition: transform 0.12s, opacity 0.12s, box-shadow 0.12s !important;
}
.stButton > button:hover {
  opacity: 0.90 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(14,165,233,0.18) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
  border: none !important;
  color: #fff !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid #1e3a5f !important;
  border-radius: 10px !important;
  background: #0a1628 !important;
  margin-bottom: 8px !important;
}
.streamlit-expanderHeader {
  font-weight: 600 !important;
  color: #38bdf8 !important;
  padding: 12px 16px !important;
  border-radius: 9px !important;
}
.streamlit-expanderHeader:hover { background: rgba(56,189,248,0.06) !important; }
.streamlit-expanderContent { padding: 4px 16px 16px !important; }

/* ── Input fields ────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: #0f1f38 !important;
  border: 1px solid #1e3a5f !important;
  border-radius: 7px !important;
  color: #e0f2fe !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: #0ea5e9 !important;
  box-shadow: 0 0 0 2px rgba(14,165,233,0.15) !important;
}
[data-baseweb="select"] > div {
  background: #0f1f38 !important;
  border: 1px solid #1e3a5f !important;
  border-radius: 7px !important;
  color: #e0f2fe !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: #1e3a5f !important; opacity: 0.6 !important; margin: 1.2rem 0 !important; }

/* ── Alert / info / warning boxes ───────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 8px !important;
  border-left-width: 3px !important;
  border-left-style: solid !important;
}
div[data-testid="stInfo"]    { background: rgba(56,189,248,0.07) !important;  border-color: #38bdf8 !important; }
div[data-testid="stWarning"] { background: rgba(234,179,8,0.07) !important;   border-color: #eab308 !important; }
div[data-testid="stSuccess"] { background: rgba(34,197,94,0.07) !important;   border-color: #22c55e !important; }
div[data-testid="stError"]   { background: rgba(239,68,68,0.07) !important;   border-color: #ef4444 !important; }

/* ── Progress bar ────────────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, #0ea5e9, #38bdf8) !important;
  border-radius: 99px !important;
}
[data-testid="stProgressBar"] { border-radius: 99px !important; background: #1e293b !important; }

/* ── Native dataframe ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: 8px !important;
  border: 1px solid #1e3a5f !important;
  overflow: hidden !important;
}

/* ── Date input ──────────────────────────────────────────────────────────── */
[data-testid="stDateInput"] input {
  background: #0f1f38 !important;
  border: 1px solid #1e3a5f !important;
  border-radius: 7px !important;
  color: #e0f2fe !important;
}

/* ── Radio ───────────────────────────────────────────────────────────────── */
[data-testid="stRadio"] > div > div { gap: 0.6rem !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #2563eb; }

/* ── Responsive stat tiles (wrap on phone, one row on desktop) ──────────── */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
  margin: 4px 0 14px;
}
.stat-tile {
  background: #0f1f38;
  border: 1px solid #1e3a5f;
  border-radius: 10px;
  padding: 12px 16px;
  min-width: 0;
}
.stat-tile .lbl {
  color: #38bdf8; font-size: 0.70rem; text-transform: uppercase;
  letter-spacing: 0.05em; font-weight: 700; white-space: nowrap;
}
.stat-tile .val { color: #e0f2fe; font-size: 1.35rem; font-weight: 800; margin-top: 2px; }
.stat-tile .sub { color: #64748b; font-size: 0.72rem; margin-top: 1px; }

/* ── Responsive play-card grid (1-col phone, 2-3 col desktop) ───────────── */
.play-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
  gap: 10px;
}
.play-card {
  background: #0f1f38;
  border: 1px solid #1e3a5f;
  border-radius: 12px;
  padding: 12px 14px;
  min-width: 0;
}
.play-card.win  { border-left: 3px solid #22c55e; }
.play-card.loss { border-left: 3px solid #ef4444; }
.play-card.open { border-left: 3px solid #f59e0b; }

/* ── Mobile tuning ──────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .block-container {
    padding-left: 0.85rem !important;
    padding-right: 0.85rem !important;
    padding-top: 0.8rem !important;
  }
  h1 { font-size: 1.45rem !important; }
  h2 { font-size: 1.2rem !important; }
  /* Wide HTML tables scroll inside their own container instead of
     stretching the page — applies to every custom table on every page */
  .stMarkdown table {
    display: block;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    white-space: nowrap;
  }
}
</style>
"""


def inject_styles():
    """Inject the shared dark-UI stylesheet into the current page."""
    st.markdown(_CSS, unsafe_allow_html=True)
