
'''
Code or Keys to Reading this code written by Me Max Onoyima
Well incase you are seeing this there's no key except to read it. Be patient. Odigo ka o ga adi

Don't mind me I have carefully place the comments to help you read he codes
Each section of the code and functionality i have placed with a well decorated section heading
just dey read dey go
'''

import html
import colorsys
import inspect
import json
import logging
import os
import pickle
import textwrap
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
st.set_page_config(
    page_title="Pay Her Right",
    page_icon=r"C:\Users\valen\PycharmProjects\PayHerRight_System\PayHer Empowerment Logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# PAGE CONFIG (must be the first Streamlit call)



log = logging.getLogger("payherright")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[PayHerRight] %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

# =====================================================================
# PATHS
# =====================================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "salary_predictor_model.pkl"
GRAPH_PATH = APP_DIR / "negotiation_graph.gpickle"
BIAS_PATH = APP_DIR / "bias_artifacts.json"

# Background image.
BG_NAME = "\u2014Pngtree\u2014multi exposure of blockchain theme_15571193.png"
BG_CANDIDATES = [
    Path(r"C:\Users\valen\PycharmProjects\PayHerRight_System\—Pngtree—multi exposure of blockchain theme_15571193.png"),
    APP_DIR / BG_NAME,
    APP_DIR / "background.png",
    APP_DIR / "background.webp",
    APP_DIR / "digital-art-cryptocurrency-bitcoin-blockchain-high-resolution-artwork-featuring-single-glittering-coins-scattered-386399271.webp",
]

# =============================
# STREAMLIT VERSION COMPATIBILITY HELPERS
# ========================================================
fragment = (getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
            or (lambda f: f))


def _params(fn):
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}


def _wide(fn):
    """Full-width kwarg for this Streamlit version (new: width='stretch', old: use_container_width)."""
    return {"width": "stretch"} if "width" in _params(fn) else {"use_container_width": True}


def _icon(fn, name):
    return {"icon": name} if "icon" in _params(fn) else {}


PLOT_W = _wide(st.plotly_chart)
BTN_W = _wide(st.button)
SUBMIT_W = _wide(st.form_submit_button)
DL_W = _wide(st.download_button)
CONTAINER_HAS_KEY = "key" in _params(st.container)

PLOT_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
                               "hoverClosestCartesian", "hoverCompareCartesian"],
}


# THEME + ACCENT PALETTES
# ==============================
ACCENTS = {
    "Royal Purple":  {"hex": "#7E22CE", "hover": "#9333EA", "rgb": "126, 34, 206", "on": "#FFFFFF"},
    "Bitcoin Gold":  {"hex": "#F7931A", "hover": "#FBA83C", "rgb": "247, 147, 26", "on": "#1A1204"},
    "Magenta Bloom": {"hex": "#D6336C", "hover": "#E8508A", "rgb": "214, 51, 108", "on": "#FFFFFF"},
    "Teal Current":  {"hex": "#0B8176", "hover": "#11A396", "rgb": "11, 129, 118", "on": "#FFFFFF"},
    "Indigo Dusk":   {"hex": "#5B4FE0", "hover": "#7468F0", "rgb": "91, 79, 224", "on": "#FFFFFF"},
}

THEMES = {
    "Dark": {   # white / #E2E8F0 text on navy glass
        "glass": "rgba(23, 30, 45, 0.70)", "solid": "#171E2D", "border": "rgba(255, 255, 255, 0.10)",
        "text": "#FFFFFF", "soft": "#CBD5E1", "input_bg": "rgba(255, 255, 255, 0.05)",
        "input_border": "rgba(255, 255, 255, 0.20)", "pop_bg": "#131A2A", "pop_text": "#FFFFFF",
        "tint": "rgba(255, 255, 255, 0.08)", "up": "#10B981", "down": "#FB7185",
        "chart_text": "#E2E8F0", "chart_grid": "rgba(255, 255, 255, 0.08)",
    },
    "Light": {  # #1E293B text on frosted white glass
        "glass": "rgba(255, 255, 255, 0.74)", "solid": "#FFFFFF", "border": "rgba(30, 41, 59, 0.14)",
        "text": "#1E293B", "soft": "#475569", "input_bg": "rgba(255, 255, 255, 0.92)",
        "input_border": "rgba(30, 41, 59, 0.30)", "pop_bg": "#FFFFFF", "pop_text": "#1E293B",
        "tint": "rgba(30, 41, 59, 0.07)", "up": "#047857", "down": "#BE123C",
        "chart_text": "#1E293B", "chart_grid": "rgba(30, 41, 59, 0.10)",
    },
}


# DROPDOWN OPTIONS (must match the categories the model was trained on)

JOB_CATEGORIES = {
    "Software Engineering": ["Backend Developer", "Frontend Developer", "Full Stack Developer",
                             "Software Engineer", "Mobile Developer (Android/iOS)"],
    "Data Science": ["Data Analyst", "Data Scientist", "Machine Learning Engineer",
                     "Business Intelligence Analyst"],
    "Product Management": ["Product Manager", "Associate Product Manager", "Technical Product Manager"],
    "DevOps": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Platform Engineer"],
    "Web Development": ["Web Developer", "WordPress Developer", "E-commerce Developer"],
}
EDUCATION_OPTIONS = ["OND", "HND", "BSc", "MSc", "PhD", "Bootcamp/Certification"]
LOCATION_OPTIONS = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Enugu", "Kano",
                    "Remote (International Client)"]
COMPANY_SIZE_OPTIONS = ["Startup (<50 employees)", "Medium (50-250 employees)", "Large (250+ employees)"]
SKILL_POOL = {
    "Software Engineering": ["JavaScript", "Python", "Java", "React", "Node.js", "SQL", "Git",
                             "TypeScript", "Kotlin", "Swift", "Django", "Spring Boot"],
    "Data Science": ["Python", "SQL", "R", "Pandas", "Scikit-learn", "TensorFlow", "Power BI",
                     "Tableau", "Excel", "PyTorch", "Statistics"],
    "Product Management": ["Agile", "Scrum", "Jira", "Roadmapping", "SQL", "Figma", "A/B Testing",
                           "Stakeholder Management", "User Research"],
    "DevOps": ["AWS", "Docker", "Kubernetes", "CI/CD", "Terraform", "Linux", "Azure", "Jenkins",
               "Ansible", "Monitoring/Grafana"],
    "Web Development": ["HTML/CSS", "JavaScript", "WordPress", "PHP", "React", "Shopify",
                        "SEO Basics", "Git"],
}
PROFILE_COLS = ["gender", "age", "years_experience", "years_current_company", "education",
                "job_category", "job_title", "location", "company_size", "remote_ratio",
                "num_skills", "certifications"]
DIMENSIONS = {  # what-if dimension -> profile column
    "Years of experience": "years_experience", "Location": "location", "Education": "education",
    "Company size": "company_size", "Remote ratio": "remote_ratio", "Job title": "job_title",
    "Number of skills": "num_skills", "Certifications": "certifications",
}
NUMERIC_DIMS = {"Years of experience", "Number of skills", "Certifications"}
EDU_LEVEL = {"OND": 1, "HND": 2, "Bootcamp/Certification": 2, "BSc": 3, "MSc": 4, "PhD": 5}

# Pay-period conversion (full-time assumptions)
PAY_PERIODS = {"Annual": 1, "Monthly": 12, "Weekly": 52, "Daily": 260, "Hourly": 2080}
PAY_PERIOD_NOTE = ("Breakdown assumes full-time work: 12 months, 52 weeks, 260 working days "
                   "(5-day week) and 2,080 working hours (40 hours a week) per year.")


# SESSION STATE DEFAULTS (set BEFORE the widgets that use these keys)

for _k, _v in {"theme": "Dark", "accent": "Royal Purple", "profile_submitted": False,
               "prediction": None, "profile_key": None, "negotiation_result": None,
               "full_name": "Candidate", "current_salary": 0}.items():
    st.session_state.setdefault(_k, _v)

# =====================================================================
# CACHED LOADERS  (cache_resource for objects, cache_data for the image)
# Each logs only on a cache MISS, so the console proves caching works:
# you should see each "LOAD" line exactly once per server process.
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    t0 = time.perf_counter()
    if not MODEL_PATH.exists():
        return None
    mdl = joblib.load(MODEL_PATH)
    # Single-row predictions are faster without thread-pool start-up overhead.
    try:
        ens = mdl.named_steps.get("ensemble")
        for est in getattr(ens, "estimators_", []):
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
    except Exception:  # pragma: no cover - purely an optimisation
        pass
    log.info("LOAD model (cache miss) in %.0f ms", (time.perf_counter() - t0) * 1000)
    return mdl


@st.cache_resource(show_spinner=False)
def load_graph():
    t0 = time.perf_counter()
    if not GRAPH_PATH.exists():
        return None
    with open(GRAPH_PATH, "rb") as f:
        g = pickle.load(f)
    log.info("LOAD graph (cache miss) in %.0f ms", (time.perf_counter() - t0) * 1000)
    return g


@st.cache_resource(show_spinner=False)
def load_bias_artifacts():
    t0 = time.perf_counter()
    if not BIAS_PATH.exists():
        return None
    with open(BIAS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    log.info("LOAD bias artefacts (cache miss) in %.0f ms", (time.perf_counter() - t0) * 1000)
    return data


def _sniff_mime(raw):
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


@st.cache_data(show_spinner=False)
def load_background():
    """Find, verify and base64-encode the background ONCE (cached)."""
    import base64
    t0 = time.perf_counter()
    paths = list(BG_CANDIDATES)
    try:  # last resort: any png/webp beside app.py that looks like the background
        for p in APP_DIR.iterdir():
            if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg") and any(
                    w in p.name.lower() for w in ("blockchain", "bitcoin", "background")):
                paths.append(p)
    except OSError:
        pass
    for p in paths:
        try:
            if not p.is_file():
                continue
            raw = p.read_bytes()
        except OSError:
            continue
        mime = _sniff_mime(raw)
        if mime is None:
            continue
        orig_kb = len(raw) // 1024
        recompressed = False
        if len(raw) > 900_000:  # a multi-MB data-URI would be re-sent on every rerun: shrink it
            try:
                import io
                from PIL import Image
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im.thumbnail((1920, 1080))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=82, optimize=True)
                raw, mime, recompressed = buf.getvalue(), "image/jpeg", True
            except Exception:
                pass
        log.info("LOAD background (cache miss) %s, %s, %d KB -> %d KB in %.0f ms", p.name, mime,
                 orig_kb, len(raw) // 1024, (time.perf_counter() - t0) * 1000)
        return {"b64": base64.b64encode(raw).decode(), "mime": mime, "path": str(p),
                "orig_kb": orig_kb, "kb": len(raw) // 1024, "recompressed": recompressed}
    log.info("Background image NOT found. Looked in: %s", [str(p) for p in paths])
    return None


_BOOT = time.perf_counter()
model = load_model()
negotiation_graph = load_graph()
bias_artifacts = load_bias_artifacts()
background = load_background()
_LOAD_MS = (time.perf_counter() - _BOOT) * 1000


# CORE LOGIC (SalaryPredictor / BiasDetector / NegotiationAssistant)
# All model calls are batched and cached, so interactions stay fast.

def profile_to_key(df):
    row = df.iloc[0]
    out = []
    for c in PROFILE_COLS:
        v = row[c]
        out.append((c, v.item() if hasattr(v, "item") else v))
    return tuple(out)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_core(key):
    """Male / female / user prediction for one profile in ONE model call."""
    base = dict(key)
    preds = model.predict(pd.DataFrame([{**base, "gender": "Male"}, {**base, "gender": "Female"}]))
    pm, pf = float(preds[0]), float(preds[1])
    user = pm if base["gender"] == "Male" else pf
    return {"male": pm, "female": pf, "user": user,
            "gap_pct": (1 - pf / pm) * 100 if pm > 0 else 0.0}


def predict_salary(key):
    user = cached_core(key)["user"]
    mae = bias_artifacts["test_mae"]
    return {"predicted_salary": round(user, -3), "range_low": round(max(0, user - mae), -3),
            "range_high": round(user + mae, -3),
            "confidence_pct": round(max(50, min(95, 100 - (mae / user * 100))), 1)}


def pay_breakdown(annual):
    return {p: annual / d for p, d in PAY_PERIODS.items()}


def _variant_rows(base, dim):
    rows, labels = [], []
    if dim == "Years of experience":
        for e in range(0, 26):
            r = dict(base)
            r.update(years_experience=e, years_current_company=min(base["years_current_company"], e),
                     age=int(min(65, max(18, base["age"] + e - base["years_experience"]))))
            rows.append(r); labels.append(e)
    else:
        options = {"Location": LOCATION_OPTIONS, "Education": EDUCATION_OPTIONS,
                   "Company size": COMPANY_SIZE_OPTIONS, "Remote ratio": [0, 50, 100],
                   "Job title": JOB_CATEGORIES[base["job_category"]],
                   "Number of skills": list(range(1, 13)), "Certifications": list(range(0, 7))}[dim]
        for v in options:
            r = dict(base); r[DIMENSIONS[dim]] = v
            rows.append(r); labels.append(v)
    return rows, labels


@st.cache_data(show_spinner=False, max_entries=96)
def cached_sweep(key, dim):
    """Predict male AND female for every value of one dimension (single batched call)."""
    base = dict(key)
    rows, labels = _variant_rows(base, dim)
    n = len(rows)
    preds = model.predict(pd.DataFrame([{**r, "gender": "Male"} for r in rows]
                                       + [{**r, "gender": "Female"} for r in rows]))
    df = pd.DataFrame({"x": labels, "male": preds[:n], "female": preds[n:]})
    df["user"] = df["male"] if base["gender"] == "Male" else df["female"]
    df["gap_pct"] = (1 - df["female"] / df["male"]) * 100
    df["is_current"] = [lab == base[DIMENSIONS[dim]] for lab in labels]
    return df


@st.cache_data(show_spinner=False, max_entries=32)
def cached_gap_distribution(key):
    """Matched-pair gap (%) across many variations of THIS profile (model-based)."""
    gaps = []
    for dim in DIMENSIONS:
        gaps.extend(cached_sweep(key, dim)["gap_pct"].tolist())
    return gaps


@st.cache_data(show_spinner=False, max_entries=32)
def cached_attribution(key):
    """Sequential counterfactual build-up of the fair salary:
    baseline (0 yrs experience, no skills/certs, lowest-paying location) ->
    + experience -> + skills & certifications -> + location = fair salary."""
    base = dict(key)
    loc_df = cached_sweep(key, "Location")
    cheapest = loc_df.loc[loc_df["user"].idxmin(), "x"]
    p0 = dict(base)
    p0.update(years_experience=0, years_current_company=0, num_skills=0, certifications=0,
              age=int(max(18, base["age"] - base["years_experience"])), location=cheapest)
    p1 = dict(p0); p1.update(years_experience=base["years_experience"],
                             years_current_company=base["years_current_company"], age=base["age"])
    p2 = dict(p1); p2.update(num_skills=base["num_skills"], certifications=base["certifications"])
    p3 = dict(base)
    v = model.predict(pd.DataFrame([p0, p1, p2, p3]))
    fair = float(v[3])
    steps = [max(0.0, float(v[1] - v[0])), max(0.0, float(v[2] - v[1])), max(0.0, float(v[3] - v[2]))]
    return {"fair": fair, "experience": steps[0], "skills": steps[1], "location": steps[2],
            "base": max(0.0, fair - sum(steps)), "cheapest": cheapest}


def prob_array(pct, k=0.12448, mid=17.588):
    return np.clip(100 / (1 + np.exp(k * (np.asarray(pct, dtype=float) - mid))), 2, 98)


def calculate_negotiation_probability(fair, ask):
    pct = (ask / fair - 1) * 100
    return round(float(prob_array(pct)), 1), round(pct, 1)


def query_negotiation_graph(graph, role, location, ask_pct):
    if graph is None or role not in graph:
        return ["No graph entry found for this role. Using general negotiation guidance only."]
    facts = []
    for _, target, _d in graph.out_edges(role, data=True):
        nd = graph.nodes[target]
        if nd.get("type") == "market_rate":
            facts.append(f"Market rate for {role}: {nd['fact']}")
        elif nd.get("type") == "location" and target == location:
            facts.append(f"Location note ({location}): {nd['fact']}")
        elif nd.get("type") == "tactic":
            facts.append(f"Recommended tactic: {target} -- {nd['fact']}")
    if ask_pct > 12:
        for node, nd in graph.nodes(data=True):
            if nd.get("type") == "tactic" and "leverage" in node.lower():
                line = (f"Since your ask is aggressive ({ask_pct:.1f}% above market): "
                        f"{node} -- {nd['fact']}")
                if not any(node in f for f in facts):  # avoid retrieving the same tactic twice
                    facts.append(line)
    return facts


def generate_negotiation_advice(role, location, fair, ask, prob, facts):
    api_key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("openai_key_runtime")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
    context = "\n".join(f"- {f}" for f in facts)
    template = (f"Based on your profile as a {role} in {location}, approach the negotiation with a "
                f"clear target supported by your skills, experience, responsibilities and market "
                f"evidence. Be ready to explain the value you bring rather than presenting the request "
                f"as an arbitrary figure. Relevant context from the negotiation knowledge base:\n"
                + "\n".join(f"  - {f}" for f in facts))
    if not api_key:
        return template, False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (f"You are an expert salary negotiation coach helping a technology professional in "
                  f"Nigeria negotiate compensation.\n\nCandidate profile:\nRole: {role}\nLocation: "
                  f"{location}\nPredicted fair market salary: NGN {fair:,.0f}\nProposed asking amount: "
                  f"NGN {ask:,.0f}\nEstimated negotiation success probability: {prob}%\n\nRelevant facts "
                  f"retrieved from the negotiation knowledge graph:\n{context}\n\nWrite 3-4 concise, "
                  f"specific, encouraging sentences of negotiation advice. Use the retrieved facts where "
                  f"relevant. Do not repeat the exact salary figures verbatim. Do not invent statistics. "
                  f"Do not claim the probability is scientifically validated.")
        out = client.responses.create(model=model_name, input=prompt).output_text
        if out and out.strip():
            return out.strip(), True
    except Exception:
        pass
    return template, False



# FORMATTING / HTML HELPERS

def ngn(x):
    return f"NGN {x:,.0f}"


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"


def acc():
    return ACCENTS[st.session_state.accent]


def card(name):
    """A glass card. Keyed containers give a stable CSS hook (st-key-card_*)."""
    if CONTAINER_HAS_KEY:
        return st.container(key=f"card_{name}")
    return st.container(border=True)


def md(html_str):
    st.markdown(html_str, unsafe_allow_html=True)


def hero(kicker, title, subtitle, big=True):
    cls = "hero big" if big else "hero small"
    md(f'<div class="{cls}"><div class="hero-kicker">{html.escape(kicker)}</div>'
       f'<div class="hero-title">{html.escape(title)}</div>'
       f'<div class="hero-subtitle">{html.escape(subtitle)}</div></div>')


def ctitle(title, sub=""):
    md(f'<div class="card-title">{html.escape(title)}</div>'
       + (f'<div class="card-sub">{html.escape(sub)}</div>' if sub else ""))


def kpi(label, value, sub="", tone=""):
    return (f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}</div>'
            f'<div class="kpi-sub {tone}">{html.escape(str(sub))}</div></div>')


def kpi_row(items):
    md('<div class="kpi-grid">' + "".join(items) + "</div>")


def html_table(df):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                   for row in df.itertuples(index=False))
    md(f'<div class="table-wrap"><table class="glass-table"><thead><tr>{head}</tr></thead>'
       f"<tbody>{body}</tbody></table></div>")



# CSS -- ONE function, cached per (theme, accent)

CSS_STATIC = """
html, body, .stApp, button, input, textarea, select { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }
.hero-title, .card-title, .kpi-value, .welcome-title,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important; font-weight: 700 !important; }

/* ---------- Transparent shells so the fixed background shows on EVERY page ---------- */
[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, .main,
[data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background: transparent !important; }
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stHeader"] *, [data-testid="stToolbar"] * { color: #F8FAFC !important; }
.block-container { max-width: 1400px !important; padding: 2rem 2rem 4rem 2rem !important; }
.block-container > [data-testid="stVerticalBlock"], [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
[data-testid="column"] > [data-testid="stVerticalBlock"] { gap: 1.5rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .8rem !important; }

/* ---------- Glass surfaces ---------- */
[class*="st-key-card_"], [data-testid="stSidebar"], [data-testid="stForm"], .kpi, .tile,
[data-testid="stMetric"], .welcome, [data-testid="stAlert"], [data-testid="stExpander"] {
    background: var(--glass) !important;
    backdrop-filter: blur(16px) saturate(130%); -webkit-backdrop-filter: blur(16px) saturate(130%);
    border: 1px solid var(--border) !important; border-radius: 20px !important;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.35); }
[class*="st-key-card_"] { padding: 1.5rem !important; gap: 1rem !important; transition: box-shadow .25s ease, transform .25s ease; }
[class*="st-key-card_"]:hover { box-shadow: 0 14px 40px rgba(0,0,0,.42), 0 0 0 1px rgba(var(--accent-rgb), .35); }
[data-testid="stSidebar"] { border-radius: 0 20px 20px 0 !important; }
[data-testid="stSidebar"] > div:first-child { background: transparent !important; }
[data-testid="stForm"] { padding: 1rem !important; box-shadow: none; background: transparent !important; backdrop-filter: none; }
[data-testid="stExpander"] { box-shadow: none; border-radius: 14px !important; backdrop-filter: none; }
[class*="st-key-card_"] [data-testid="stExpander"], [data-testid="stSidebar"] [data-testid="stExpander"] { background: var(--tint) !important; }
[data-testid="stAlert"] { border-radius: 14px !important; box-shadow: none; }
hr { border-color: var(--border) !important; }

/* ---------- Typography / explicit text colours (contrast) ---------- */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, label, label p,
[data-testid="stRadio"] label, [data-testid="stRadio"] label *, [data-testid="stCheckbox"] label *,
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary *,
[data-testid="stAlert"], [data-testid="stAlert"] *, [data-testid="stSpinner"] *,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"],
[data-testid="stMetricValue"] *, [data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] *,
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"], [data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: var(--soft) !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { font-weight: 800 !important; }
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]), [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) * { color: var(--up) !important; }
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]), [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) * { color: var(--down) !important; }
[data-testid="stMetricDelta"] svg { fill: currentColor !important; }
[data-testid="stExpander"] summary svg, [data-baseweb="select"] svg { fill: var(--text) !important; color: var(--text) !important; }
a, a * { color: var(--accent-link) !important; }
code { background: var(--tint) !important; color: var(--text) !important; }

/* ---------- Hero ---------- */
.hero { text-align: center; margin: 0 auto 1.2rem auto; }
.hero.big { padding: 1.5rem 0 .5rem 0; }
.hero-kicker { display: inline-block; letter-spacing: .28em; text-transform: uppercase; font-size: .78rem; font-weight: 700;
    color: #FFFFFF !important; padding: .35rem 1rem; border-radius: 999px; background: rgba(0,0,0,.35);
    border: 1px solid rgba(255,255,255,.25); backdrop-filter: blur(8px); }
.hero-title { color: #FFFFFF !important; text-shadow: 0 4px 10px rgba(0,0,0,.4); line-height: 1.05; margin: .6rem 0 .3rem 0; }
.hero.big .hero-title { font-size: clamp(2.8rem, 6vw, 5.2rem); }
.hero.small .hero-title { font-size: clamp(1.7rem, 3vw, 2.4rem); }
.hero-subtitle { color: #E2E8F0 !important; text-shadow: 0 2px 8px rgba(0,0,0,.5); font-size: 1.1rem; max-width: 760px; margin: 0 auto; }

/* ---------- Cards, tiles, KPIs ---------- */
.card-title { font-size: 1.45rem; color: var(--text) !important; margin: 0; }
.card-sub { color: var(--soft) !important; font-size: .92rem; margin: .15rem 0 .4rem 0; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1.5rem; }
.kpi { padding: 1.1rem 1.3rem; }
.kpi-label { color: var(--soft) !important; font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.kpi-value { color: var(--text) !important; font-size: 1.85rem; line-height: 1.2; margin-top: .25rem; }
.kpi-sub { color: var(--soft) !important; font-size: .85rem; margin-top: .15rem; }
.kpi-sub.up { color: var(--up) !important; font-weight: 600; }
.kpi-sub.down { color: var(--down) !important; font-weight: 600; }
.tile-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; }
.tile { padding: 1.5rem; }
.tile-icon { font-size: 1.8rem; }
.tile-title { font-family: 'Playfair Display', Georgia, serif; font-weight: 700; font-size: 1.3rem; color: var(--text) !important; margin: .4rem 0 .25rem 0; }
.tile-body { color: var(--soft) !important; font-size: .95rem; line-height: 1.5; }
.welcome { max-width: 680px; margin: 0 auto; padding: 2rem; text-align: center; }
.welcome-title { font-size: 1.9rem; color: var(--text) !important; }
.welcome-sub { color: var(--soft) !important; margin: .3rem 0 1rem 0; }
.step { text-align: left; color: var(--text) !important; padding: .55rem .9rem; margin: .4rem 0; border-radius: 12px; background: var(--tint); }
.step b { color: var(--accent-link) !important; }
.salary-big { font-family: 'Playfair Display', Georgia, serif; font-weight: 800; font-size: clamp(2.2rem, 4vw, 3.2rem); color: var(--text) !important; line-height: 1.1; }
.salary-big small { font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600; color: var(--soft) !important; }
.chips { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: .5rem; }
.chip { padding: .35rem .8rem; border-radius: 999px; background: var(--tint); border: 1px solid var(--border); color: var(--text) !important; font-size: .88rem; }
.period-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .8rem; }
.period { padding: .8rem 1rem; border-radius: 14px; background: var(--tint); border: 1px solid var(--border); }
.period .p-label { color: var(--soft) !important; font-size: .75rem; font-weight: 700; letter-spacing: .08em; }
.period .p-val { color: var(--text) !important; font-size: 1.15rem; font-weight: 800; }
.pill { display: inline-block; padding: .4rem 1.1rem; border-radius: 999px; font-weight: 700; font-size: .95rem; }
.pill-pass { background: rgba(16,185,129,.18); color: var(--up) !important; border: 1px solid var(--up); box-shadow: 0 0 14px rgba(16,185,129,.3); }
.pill-fail { background: rgba(244,63,94,.16); color: var(--down) !important; border: 1px solid var(--down); box-shadow: 0 0 14px rgba(244,63,94,.3); }
.metric-row { display: flex; justify-content: space-between; gap: 1rem; padding: .6rem .2rem; border-bottom: 1px solid var(--border); color: var(--text) !important; }
.metric-row:last-child { border-bottom: none; }
.ok { color: var(--up) !important; font-weight: 700; } .bad { color: var(--down) !important; font-weight: 700; }
.fact { padding: .7rem 1rem; margin: .4rem 0; border-radius: 12px; background: var(--tint); border-left: 4px solid var(--accent); color: var(--text) !important; font-size: .93rem; }
.advice { padding: 1.1rem 1.3rem; border-radius: 14px; background: var(--tint); border: 1px solid var(--border); color: var(--text) !important; line-height: 1.65; white-space: pre-wrap; }
.table-wrap { overflow-x: auto; }
.glass-table { width: 100%; border-collapse: collapse; }
.glass-table th { text-align: left; padding: .6rem .8rem; color: var(--soft) !important; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; border-bottom: 1px solid var(--border); }
.glass-table td { padding: .6rem .8rem; color: var(--text) !important; border-bottom: 1px solid var(--border); }

/* ---------- Buttons: accent fill, lift + HALO GLOW on hover ---------- */
:is(.stButton, .stFormSubmitButton, .stDownloadButton) button {
    background: var(--accent) !important; color: var(--on-accent) !important; border: none !important;
    border-radius: 10px !important; padding: .9rem 1.5rem !important; font-weight: 600 !important; font-size: 1rem !important;
    min-height: 3rem; transition: transform .18s ease, box-shadow .22s ease, background .18s ease; }
:is(.stButton, .stFormSubmitButton, .stDownloadButton) button *, :is(.stButton, .stFormSubmitButton, .stDownloadButton) button p {
    color: var(--on-accent) !important; font-weight: 600 !important; }
:is(.stButton, .stFormSubmitButton, .stDownloadButton) button:hover {
    background: var(--accent-hover) !important; transform: translateY(-2px);
    box-shadow: 0 0 0 2px var(--solid), 0 0 0 5px rgba(var(--accent-rgb), .45), 0 0 28px 4px rgba(var(--accent-rgb), .40) !important; }
:is(.stButton, .stFormSubmitButton, .stDownloadButton) button:active { transform: translateY(0); }
:is(.stButton, .stFormSubmitButton, .stDownloadButton) button:focus-visible, [data-baseweb="tab"]:focus-visible {
    outline: 3px solid #FFFFFF !important; outline-offset: 3px !important;
    box-shadow: 0 0 0 5px rgba(var(--accent-rgb), .55) !important; }

/* ---------- Inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
    background: var(--input-bg) !important; border: 1px solid var(--input-border) !important; border-radius: 8px !important;
    transition: box-shadow .2s ease, border-color .2s ease; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input, textarea, [data-baseweb="select"] * { color: var(--text) !important; }
input::placeholder, textarea::placeholder { color: var(--soft) !important; opacity: 1; }
[data-baseweb="input"]:hover, [data-baseweb="select"] > div:hover { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .18); }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div, [data-baseweb="textarea"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--solid), 0 0 0 5px rgba(var(--accent-rgb), .45), 0 0 28px 4px rgba(var(--accent-rgb), .40) !important; }
[data-testid="stNumberInput"] button { background: transparent !important; color: var(--text) !important; box-shadow: none !important; }
[data-testid="stNumberInput"] button svg { fill: var(--text) !important; }
[data-baseweb="tag"] { background: var(--accent) !important; border-radius: 8px !important; }
[data-baseweb="tag"] *, [data-baseweb="tag"] span { color: var(--on-accent) !important; }
[data-baseweb="tag"] svg { fill: var(--on-accent) !important; }

/* ---------- Dropdown popovers: SOLID with matching text ---------- */
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul, [role="listbox"] {
    background: var(--pop-bg) !important; color: var(--pop-text) !important; opacity: 1 !important; }
[data-baseweb="popover"] > div { border: 1px solid var(--input-border) !important; border-radius: 12px !important; box-shadow: 0 18px 40px rgba(0,0,0,.45) !important; }
[data-baseweb="popover"] li, [role="option"], [role="option"] * { background: var(--pop-bg) !important; color: var(--pop-text) !important; }
[role="option"]:hover, [role="option"]:hover *, li[aria-selected="true"], li[aria-selected="true"] * { background: rgba(var(--accent-rgb), .28) !important; }

/* ---------- Tabs: glass pill ---------- */
[data-baseweb="tab-list"] { background: var(--glass) !important; backdrop-filter: blur(16px) saturate(130%); border: 1px solid var(--border);
    border-radius: 15px; padding: 6px; gap: 6px; box-shadow: 0 10px 35px rgba(0,0,0,.35); }
[data-baseweb="tab"] { background: transparent !important; color: var(--soft) !important; border-radius: 10px !important; padding: .65rem 1.3rem !important;
    font-weight: 600 !important; transition: background .18s ease, box-shadow .22s ease; }
[data-baseweb="tab"] p, [data-baseweb="tab"] span, [data-baseweb="tab"] div { color: inherit !important; font-weight: 600 !important; }
[data-baseweb="tab"]:hover { background: var(--tint) !important; color: var(--text) !important;
    box-shadow: 0 0 0 1px rgba(var(--accent-rgb), .4), 0 0 20px rgba(var(--accent-rgb), .38) !important; }
[data-baseweb="tab"][aria-selected="true"] { background: var(--accent) !important; color: var(--on-accent) !important;
    box-shadow: 0 0 18px rgba(var(--accent-rgb), .55) !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* ---------- Slider: accent track (hue-shifted), purple thumb with white ring + glow ---------- */
[data-baseweb="slider"] div[style*="linear-gradient"] { filter: hue-rotate(var(--hue)) saturate(1.15); }
[data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; border: 3px solid #FFFFFF !important;
    box-shadow: 0 0 12px rgba(var(--accent-rgb), .6) !important; transition: box-shadow .2s ease; }
[data-baseweb="slider"] [role="slider"]:hover, [data-baseweb="slider"] [role="slider"]:focus {
    box-shadow: 0 0 0 5px rgba(var(--accent-rgb), .35), 0 0 22px 4px rgba(var(--accent-rgb), .6) !important; }
[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent-hover)) !important; }

/* ---------- Metrics ---------- */
[data-testid="stMetric"] { padding: 1rem 1.2rem; box-shadow: none; }
</style>
"""


def _hue_shift(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return round(colorsys.rgb_to_hsv(r, g, b)[0] * 360)  # Streamlit's default red has hue 0


@st.cache_data(show_spinner=False)
def build_css(theme_name, accent_name):
    t, a = THEMES[theme_name], ACCENTS[accent_name]
    link = a["hover"] if theme_name == "Dark" else a["hex"]
    if background:
        img = f"url('data:{background['mime']};base64,{background['b64']}')"
        bg_rule = (f".stApp {{ background: linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.55)) fixed, {img} "
                   f"center / cover no-repeat fixed !important; }}")
    else:  # graceful fallback so the app is still beautiful without the image
        bg_rule = (".stApp { background: radial-gradient(1200px 700px at 15% 10%, #3B1A66 0%, transparent 60%), "
                   "radial-gradient(900px 600px at 90% 90%, #0B3A5B 0%, transparent 60%), #0B1020 !important; }")
    variables = (
        ":root {"
        f"--glass:{t['glass']}; --solid:{t['solid']}; --border:{t['border']}; --text:{t['text']}; --soft:{t['soft']};"
        f"--input-bg:{t['input_bg']}; --input-border:{t['input_border']}; --pop-bg:{t['pop_bg']}; --pop-text:{t['pop_text']};"
        f"--tint:{t['tint']}; --up:{t['up']}; --down:{t['down']};"
        f"--accent:{a['hex']}; --accent-hover:{a['hover']}; --accent-rgb:{a['rgb']}; --on-accent:{a['on']};"
        f"--accent-link:{link}; --hue:{_hue_shift(a['hex'])}deg; }}"
        f" .stApp {{ color-scheme: {'dark' if theme_name == 'Dark' else 'light'}; }}"
    )
    fonts = ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
             "&family=Playfair+Display:wght@700;800&display=swap');")
    return f"<style>{fonts}{variables}{bg_rule}{CSS_STATIC}"


def inject_css():
    md(build_css(st.session_state.theme, st.session_state.accent))


# CHART HELPERS (Plotly, transparent, theme-aware)

def style_fig(fig, height=340):
    t = THEMES[st.session_state.theme]
    fig.update_layout(
        template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(color=t["chart_text"], family="Inter, Segoe UI, sans-serif", size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor=acc()["hex"], font=dict(color="#FFFFFF", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t["chart_text"])),
    )
    ax = dict(gridcolor=t["chart_grid"], zerolinecolor=t["chart_grid"], automargin=True,
              title_font=dict(color=t["chart_text"]), tickfont=dict(color=t["chart_text"]))
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    return fig


def show(fig, key, height=340):
    st.plotly_chart(style_fig(fig, height), key=key, config=PLOT_CONFIG, **PLOT_W)


def fig_pay_breakdown(pred):
    periods = list(PAY_PERIODS)
    val = [pred["predicted_salary"] / d for d in PAY_PERIODS.values()]
    lo = [max(1.0, pred["range_low"] / d) for d in PAY_PERIODS.values()]
    hi = [pred["range_high"] / d for d in PAY_PERIODS.values()]
    t = THEMES[st.session_state.theme]
    fig = go.Figure(go.Bar(
        y=periods, x=val, orientation="h", marker=dict(color=acc()["hex"]),
        error_x=dict(type="data", symmetric=False, array=[h - v for h, v in zip(hi, val)],
                     arrayminus=[v - l for v, l in zip(val, lo)], color=t["chart_text"], thickness=2, width=6),
        customdata=np.column_stack([lo, hi]),
        hovertemplate="<b>%{y}</b><br>Predicted: NGN %{x:,.0f}<br>Range: NGN %{customdata[0]:,.0f} "
                      "to NGN %{customdata[1]:,.0f}<extra></extra>"))
    fig.update_xaxes(type="log", title="NGN (log scale, whiskers = estimated range)",
                     range=[np.log10(min(lo) * 0.5), np.log10(max(hi) * 1.6)])
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_importance(fi, n):
    d = fi.head(n)[::-1]
    fig = go.Figure(go.Bar(y=d["label"], x=d["importance"], orientation="h", marker=dict(color=acc()["hex"]),
                           hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Importance (share of model decisions)")
    return fig


def fig_salary_compare(current, pred):
    t = THEMES[st.session_state.theme]
    a = acc()["hex"]
    p, lo, hi = pred["predicted_salary"], pred["range_low"], pred["range_high"]
    gap = p - current
    fig = go.Figure()
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=-0.3, y1=0.3, fillcolor=rgba(a, 0.28),
                  line=dict(color=a, width=1))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="markers", showlegend=False,
                             marker=dict(size=1, color="rgba(0,0,0,0)"),
                             hovertemplate=["Range low: NGN %{x:,.0f}<extra></extra>",
                                            "Range high: NGN %{x:,.0f}<extra></extra>"]))
    fig.add_trace(go.Scatter(x=[p], y=[0], mode="markers", name="Predicted fair salary",
                             marker=dict(size=22, symbol="diamond", color=a, line=dict(color="#FFFFFF", width=2)),
                             hovertemplate=f"<b>Predicted fair salary</b><br>NGN %{{x:,.0f}}<br>"
                                           f"Range: {ngn(lo)} to {ngn(hi)}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[current], y=[0], mode="markers", name="Current salary",
                             marker=dict(size=20, color="#F59E0B", line=dict(color="#FFFFFF", width=2)),
                             hovertemplate=f"<b>Current salary</b><br>NGN %{{x:,.0f}}<br>"
                                           f"Gap to fair salary: {ngn(gap)} ({gap / current * 100:+.1f}%)<extra></extra>"))
    xs = [lo, hi, current]
    pad = (max(xs) - min(xs)) * 0.15 + 1
    fig.update_xaxes(title="NGN per year", range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_layout(legend=dict(orientation="h", y=1.25, x=0))
    return fig


def fig_radar(row):
    cats = ["Experience", "Skills", "Certifications", "Education", "Remote access", "Tenure"]
    vals = [min(row["years_experience"] / 25, 1), min(row["num_skills"] / 12, 1),
            min(row["certifications"] / 5, 1), EDU_LEVEL[row["education"]] / 5,
            row["remote_ratio"] / 100, min(row["years_current_company"] / 20, 1)]
    a = acc()["hex"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0.5] * 7, theta=cats + cats[:1], name="Scale midpoint (reference)",
                                  line=dict(color="#94A3B8", dash="dot"), hovertemplate="%{theta}: 50% of scale<extra>Reference</extra>"))
    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself", name="Your profile",
                                  line=dict(color=a), fillcolor=rgba(a, 0.30),
                                  hovertemplate="%{theta}: %{r:.0%} of scale<extra>You</extra>"))
    t = THEMES[st.session_state.theme]
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=t["chart_grid"], tickfont=dict(color=t["chart_text"])),
                                 angularaxis=dict(gridcolor=t["chart_grid"], tickfont=dict(color=t["chart_text"]))),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def fig_whatif(df, dim, gender):
    a = acc()["hex"]
    cur = df[df["is_current"]]
    if dim in NUMERIC_DIMS:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=df["x"], y=df["user"], mode="lines+markers", name=f"Predicted ({gender})",
                                   line=dict(color=a, width=3), marker=dict(size=6),
                                   hovertemplate=f"{dim}: %{{x}}<br>Predicted: NGN %{{y:,.0f}}<extra></extra>"))
        fig.add_trace(go.Scattergl(x=cur["x"], y=cur["user"], mode="markers", name="Your profile",
                                   marker=dict(size=15, color="#F59E0B", line=dict(color="#FFFFFF", width=2)),
                                   hovertemplate=f"<b>Your profile</b><br>{dim}: %{{x}}<br>NGN %{{y:,.0f}}<extra></extra>"))
    else:
        colors = [a if c else rgba(a, 0.45) for c in df["is_current"]]
        fig = go.Figure(go.Bar(x=df["x"].astype(str), y=df["user"], marker=dict(color=colors),
                               hovertemplate=f"{dim}: %{{x}}<br>Predicted: NGN %{{y:,.0f}}<extra></extra>"))
        fig.update_xaxes(tickangle=-20)
    fig.update_xaxes(title=dim)
    fig.update_yaxes(title="Predicted annual salary (NGN)")
    fig.update_layout(legend=dict(orientation="h", y=1.15))
    return fig


def fig_matched_pair(core, mae):
    fig = go.Figure(go.Bar(
        x=["If listed as Male", "If listed as Female"], y=[core["male"], core["female"]],
        marker=dict(color=["#38BDF8", "#F472B6"]),
        error_y=dict(type="data", array=[mae, mae], color=THEMES[st.session_state.theme]["chart_text"], thickness=2, width=8),
        customdata=[[0.0], [core["gap_pct"]]],
        hovertemplate="<b>%{x}</b><br>NGN %{y:,.0f}<br>Whisker = +/- model MAE<br>"
                      "Gap vs male: %{customdata[0]:.2f}%<extra></extra>"))
    fig.update_yaxes(title="Predicted annual salary (NGN)",
                     range=[0, max(core["male"], core["female"]) + mae * 1.3])
    fig.add_annotation(x=1, y=core["female"] + mae * 1.15, text=f"Gap {core['gap_pct']:.2f}%", showarrow=False,
                       font=dict(color="#F472B6", size=14))
    return fig


def fig_fairness(fm, targets):
    rows = [("Equalized Odds Diff.", fm["equalized_odds_difference"], 0.05, "< 0.05", targets["equalized_odds_difference"]),
            ("Demographic Parity Ratio", fm["demographic_parity_ratio"], 0.80, "> 0.80", targets["demographic_parity_ratio"]),
            ("Disparate Impact", fm["disparate_impact"], 0.80, "> 0.80", targets["disparate_impact"])]
    t = THEMES[st.session_state.theme]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h", name="Observed",
        marker=dict(color=["#10B981" if r[4] else "#F43F5E" for r in rows]),
        customdata=[[r[3], "PASS" if r[4] else "FAIL"] for r in rows],
        hovertemplate="<b>%{y}</b><br>Value: %{x:.3f}<br>Target: %{customdata[0]}<br>Status: %{customdata[1]}<extra></extra>"))
    fig.add_trace(go.Scatter(
        y=[r[0] for r in rows], x=[r[2] for r in rows], mode="markers", name="Target threshold",
        marker=dict(symbol="line-ns", size=30, line=dict(color="#F59E0B", width=4)),
        hovertemplate="Threshold: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.1], title="Metric value")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(legend=dict(orientation="h", y=1.18))
    return fig


def fig_gap_hist(gaps, user_gap, mean_gap, source):
    a = acc()["hex"]
    t = THEMES[st.session_state.theme]
    fig = go.Figure(go.Histogram(x=gaps, nbinsx=20, marker=dict(color=a, line=dict(color=t["chart_text"], width=0.5)),
                                 opacity=0.85, hovertemplate="Gap %{x:.2f}%<br>Count: %{y}<extra></extra>"))
    fig.add_vline(x=user_gap, line=dict(color="#F59E0B", width=3, dash="dash"),
                  annotation_text=f"Your gap {user_gap:.2f}%", annotation_font_color=t["chart_text"])
    if mean_gap is not None:
        fig.add_vline(x=mean_gap, line=dict(color=t["chart_text"], width=2, dash="dot"),
                      annotation_text=f"Model-wide mean {mean_gap:.2f}%", annotation_position="bottom right",
                      annotation_font_color=t["chart_text"])
    fig.update_xaxes(title=f"Matched-pair gap, % lower for female ({source})")
    fig.update_yaxes(title="Count")
    return fig


def fig_gap_explorer(df, dim):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if dim in NUMERIC_DIMS:
        fig.add_trace(go.Scatter(x=df["x"], y=df["male"], name="Male", mode="lines", line=dict(color="#38BDF8", width=3),
                                 hovertemplate=f"{dim}: %{{x}}<br>Male: NGN %{{y:,.0f}}<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df["x"], y=df["female"], name="Female", mode="lines", line=dict(color="#F472B6", width=3),
                                 hovertemplate=f"{dim}: %{{x}}<br>Female: NGN %{{y:,.0f}}<extra></extra>"), secondary_y=False)
    else:
        xs = df["x"].astype(str)
        fig.add_trace(go.Bar(x=xs, y=df["male"], name="Male", marker=dict(color="#38BDF8"),
                             hovertemplate=f"{dim}: %{{x}}<br>Male: NGN %{{y:,.0f}}<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Bar(x=xs, y=df["female"], name="Female", marker=dict(color="#F472B6"),
                             hovertemplate=f"{dim}: %{{x}}<br>Female: NGN %{{y:,.0f}}<extra></extra>"), secondary_y=False)
        fig.update_xaxes(tickangle=-20)
    fig.add_trace(go.Scatter(x=df["x"] if dim in NUMERIC_DIMS else df["x"].astype(str), y=df["gap_pct"], name="Gap %",
                             mode="lines+markers", line=dict(color="#F59E0B", width=2, dash="dot"),
                             hovertemplate=f"{dim}: %{{x}}<br>Gap: %{{y:.2f}}%<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="Predicted salary (NGN)", secondary_y=False)
    fig.update_yaxes(title_text="Gap (%)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title=dim)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15))
    return fig


def fig_prob_curve(fair, ask, scenario_asks):
    a = acc()["hex"]
    asks = np.linspace(fair * 0.9, fair * 1.5, 70)
    pct = (asks / fair - 1) * 100
    band = np.vstack([prob_array(pct, k, m) for k in (0.12448 * 0.9, 0.12448 * 1.1) for m in (16.588, 18.588)])
    lo, hi = band.min(axis=0), band.max(axis=0)
    mid = prob_array(pct)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([asks, asks[::-1]]), y=np.concatenate([hi, lo[::-1]]), fill="toself",
                             fillcolor=rgba(a, 0.18), line=dict(width=0), hoverinfo="skip",
                             name="Sensitivity band"))
    fig.add_trace(go.Scattergl(x=asks, y=mid, mode="lines", name="Success probability", line=dict(color=a, width=3),
                               customdata=np.column_stack([pct, lo, hi]),
                               hovertemplate="Ask: NGN %{x:,.0f} (%{customdata[0]:+.1f}% above market)<br>"
                                             "<b>%{y:.1f}% success chance</b> "
                                             "(band %{customdata[1]:.0f}-%{customdata[2]:.0f}%)<extra></extra>"))
    sx = list(scenario_asks.values())
    fig.add_trace(go.Scattergl(x=sx, y=[calculate_negotiation_probability(fair, v)[0] for v in sx], mode="markers+text",
                               name="Quick scenarios", text=list(scenario_asks.keys()), textposition="top right",
                               marker=dict(size=10, color="#38BDF8", line=dict(color="#FFFFFF", width=1)),
                               textfont=dict(color=THEMES[st.session_state.theme]["chart_text"]),
                               hovertemplate="%{text}: NGN %{x:,.0f} -> %{y:.1f}%<extra></extra>"))
    p, pc = calculate_negotiation_probability(fair, ask)
    fig.add_trace(go.Scattergl(x=[ask], y=[p], mode="markers", name="Your ask",
                               marker=dict(size=17, color="#F59E0B", line=dict(color="#FFFFFF", width=2)),
                               hovertemplate=f"<b>Your ask</b><br>NGN %{{x:,.0f}} ({pc:+.1f}% above market)<br>{p}% success chance<extra></extra>"))
    fig.update_xaxes(title="Asking amount (NGN per year)")
    fig.update_yaxes(title="Success probability (%)", range=[0, 100])
    fig.update_layout(legend=dict(orientation="h", y=1.15))
    return fig


def fig_treemap(att, ask):
    margin = max(0.0, ask - att["fair"])
    parts = [("Base (entry-level equivalent)", att["base"], "#334155"), ("Experience premium", att["experience"], "#7E22CE"),
             ("Skills & certifications premium", att["skills"], "#0369A1"), ("Location premium", att["location"], "#0F766E"),
             ("Negotiation margin", margin, "#BE185D")]
    total = sum(v for _, v, _ in parts)
    fig = go.Figure(go.Treemap(
        labels=["Your ask"] + [p[0] for p in parts], parents=[""] + ["Your ask"] * len(parts),
        values=[total] + [p[1] for p in parts], branchvalues="total",
        marker=dict(colors=["rgba(0,0,0,0)"] + [p[2] for p in parts], line=dict(color="rgba(255,255,255,0.25)", width=1)),
        textfont=dict(color="#FFFFFF", size=14),
        texttemplate="<b>%{label}</b><br>NGN %{value:,.0f}<br>%{percentRoot:.1%}",
        hovertemplate="<b>%{label}</b><br>NGN %{value:,.0f}<br>%{percentRoot:.1%} of your ask<extra></extra>",
        tiling=dict(pad=3)))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def fig_scenarios(fair, current_ask):
    labels, asks, probs = [], [], []
    for pct in (5, 10, 15):
        v = round(fair * (1 + pct / 100), -3)
        labels.append(f"{pct}% above"); asks.append(v); probs.append(calculate_negotiation_probability(fair, v)[0])
    labels.append("Your ask"); asks.append(current_ask); probs.append(calculate_negotiation_probability(fair, current_ask)[0])
    a = acc()["hex"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=asks, name="Ask (NGN)", offsetgroup="ask", marker=dict(color=a),
                         customdata=probs, hovertemplate="<b>%{x}</b><br>Ask: NGN %{y:,.0f}<br>"
                                                         "Success chance: %{customdata:.1f}%<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Bar(x=labels, y=probs, name="Success probability (%)", offsetgroup="prob", marker=dict(color="#38BDF8"),
                         customdata=asks, hovertemplate="<b>%{x}</b><br>Success chance: %{y:.1f}%<br>"
                                                        "Ask: NGN %{customdata:,.0f}<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="Ask (NGN)", secondary_y=False)
    fig.update_yaxes(title_text="Success probability (%)", range=[0, 100], secondary_y=True, showgrid=False)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15))
    return fig


def aggressiveness(pct_above):
    return float(np.clip(pct_above / 30 * 100, 0, 100))


def fig_gauge(score):
    t = THEMES[st.session_state.theme]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix=" / 100", font=dict(color=t["chart_text"])),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=t["chart_text"], tickfont=dict(color=t["chart_text"])),
                   bar=dict(color=acc()["hex"], thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, 35], color="rgba(16,185,129,0.55)"), dict(range=[35, 65], color="rgba(245,158,11,0.55)"),
                          dict(range=[65, 100], color="rgba(239,68,68,0.60)")],
                   threshold=dict(line=dict(color=t["chart_text"], width=4), thickness=0.85, value=score))))
    return fig


def fig_graph(graph, role, location, ask_pct):
    t = THEMES[st.session_state.theme]
    nbrs = list(graph.successors(role))
    extra = []
    if ask_pct > 12:
        extra = [n for n, d in graph.nodes(data=True) if d.get("type") == "tactic" and "leverage" in n.lower() and n not in nbrs]
    palette = {"role": acc()["hex"], "market_rate": "#38BDF8", "location": "#FBBF24", "tactic": "#34D399"}
    pos = {role: (0.0, 0.0)}
    for i, n in enumerate(nbrs):
        ang = 2 * np.pi * i / max(len(nbrs), 1)
        pos[n] = (np.cos(ang), np.sin(ang))
    for j, n in enumerate(extra):
        pos[n] = (1.7 * np.cos(np.pi / 4 + j), 1.7 * np.sin(np.pi / 4 + j))
    fig = go.Figure()
    for n in nbrs + extra:
        fig.add_trace(go.Scatter(x=[0, pos[n][0]], y=[0, pos[n][1]], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=t["chart_grid"].replace("0.08", "0.45").replace("0.10", "0.45"),
                                           width=1.5, dash="dash" if n in extra else "solid")))
    names = [role] + nbrs + extra
    ntype = lambda n: graph.nodes[n].get("type", "role")
    used = [True] + [not (graph.nodes[n].get("type") == "location" and n != location) for n in nbrs] + [True] * len(extra)
    wrap = lambda s: "<br>".join(textwrap.wrap(s, 50))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in names], y=[pos[n][1] for n in names], mode="markers+text", showlegend=False,
        text=[(n if len(n) <= 26 else n[:24] + "..") for n in names], textposition="bottom center",
        textfont=dict(color=t["chart_text"], size=11),
        marker=dict(size=[34 if n == role else 24 for n in names], color=[palette.get(ntype(n), "#94A3B8") if u else "rgba(148,163,184,0.35)" for n, u in zip(names, used)],
                    line=dict(color="#FFFFFF", width=1.5)),
        customdata=[[ntype(n), wrap(graph.nodes[n].get("fact", n)), "retrieved" if u else "not retrieved (other location)"] for n, u in zip(names, used)],
        hovertemplate="<b>%{text}</b> (%{customdata[0]})<br>%{customdata[1]}<br><i>%{customdata[2]}</i><extra></extra>"))
    fig.update_xaxes(visible=False, range=[-2.2, 2.2])
    fig.update_yaxes(visible=False, range=[-1.7, 2.2])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig



# SIDEBAR  (appearance FIRST, so CSS reflects the choice in the SAME run)

with st.sidebar:
    md('<div class="hero-title" style="font-size:1.9rem;text-align:left;margin:0;color:var(--text) !important;'
       'text-shadow:none;">Pay Her Right</div>')
    st.caption("AI Negotiation Assistant for Gender Pay Equity in Nigerian Tech")
    st.markdown("---")
    st.radio("Theme", ["Dark", "Light"], key="theme", horizontal=True)
    st.selectbox("Accent colour", list(ACCENTS), key="accent")

inject_css()   # runs after the two widgets above, so changes apply instantly with no refresh

with st.sidebar:
    st.markdown("---")
    ctitle("Input Profile")
    job_category = st.selectbox("Job Category", list(JOB_CATEGORIES), key="job_category_sel")  # outside the form -> title list updates live
    with st.form("profile_form"):
        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        gender = st.selectbox("Gender", ["Female", "Male"])
        job_title = st.selectbox("Job Title", JOB_CATEGORIES[job_category])
        years_experience = st.slider("Years of Experience", 0, 25, 3)
        years_current_company = st.number_input("Years at Current Company", min_value=0, max_value=25, value=1)
        education = st.selectbox("Education Level", EDUCATION_OPTIONS, index=2)
        location = st.selectbox("Location", LOCATION_OPTIONS)
        company_size = st.selectbox("Company Size", COMPANY_SIZE_OPTIONS, index=1)
        remote_ratio = st.select_slider("Remote Work Ratio (%)", options=[0, 50, 100], value=50)
        skills = st.multiselect("Technical Skills", SKILL_POOL[job_category], default=SKILL_POOL[job_category][:3])
        certifications = st.number_input("Number of Certifications", min_value=0, max_value=10, value=1)
        age = st.number_input("Age", min_value=18, max_value=65, value=28)
        current_salary = st.number_input("Current Salary (NGN, optional)", min_value=0, value=0, step=50000)
        c1, c2 = st.columns(2)
        submitted = c1.form_submit_button("Submit Profile", type="primary", **SUBMIT_W,
                                          **_icon(st.form_submit_button, ":material/login:"))
        cleared = c2.form_submit_button("Clear All", type="secondary", **SUBMIT_W,
                                        **_icon(st.form_submit_button, ":material/restart_alt:"))

    if not os.environ.get("OPENAI_API_KEY"):
        with st.expander("\U0001F511 OpenAI API Key (optional)"):
            st.caption("Only needed for AI-generated advice. Without it, template advice is used. Not saved to disk.")
            _k = st.text_input("API key", type="password", key="openai_key_input")
            if _k:
                st.session_state["openai_key_runtime"] = _k

    if model is None:
        st.error(f"Model file not found: {MODEL_PATH.name}")
    if negotiation_graph is None:
        st.warning(f"Negotiation graph not found: {GRAPH_PATH.name}")
    if bias_artifacts is None:
        st.warning(f"Bias artefacts not found: {BIAS_PATH.name}")
    if background is None:
        st.warning("Background image not found, using a gradient instead. Put the PNG next to app.py "
                   "or fix BG_CANDIDATES at the top of the file.")
    with st.expander("\u2699\uFE0F Diagnostics"):
        st.caption(f"Artefacts + background ready in {_LOAD_MS:.0f} ms (cached after first load; the console logs a "
                   f"'LOAD ... (cache miss)' line only once per artefact).")
        if background:
            st.caption(f"Background: {Path(background['path']).name} | {background['mime']} | "
                       f"{background['orig_kb']} KB" + (f" -> {background['kb']} KB (optimised)" if background["recompressed"] else ""))
        if not CONTAINER_HAS_KEY:
            st.caption("Tip: run `pip install -U streamlit` for full glass-card styling.")

if cleared:
    st.session_state.update(profile_submitted=False, prediction=None, profile_key=None, negotiation_result=None)

if submitted and model is not None and bias_artifacts is not None:
    ycc = min(int(years_current_company), int(years_experience))
    if ycc != years_current_company:
        st.sidebar.warning("Years at current company can't exceed total experience; adjusted.")
    _df = pd.DataFrame([{"gender": gender, "age": age, "years_experience": years_experience,
                         "years_current_company": ycc, "education": education, "job_category": job_category,
                         "job_title": job_title, "location": location, "company_size": company_size,
                         "remote_ratio": remote_ratio, "num_skills": len(skills), "certifications": certifications}])
    st.session_state.profile_key = profile_to_key(_df)
    st.session_state.full_name = full_name.strip() or "Candidate"
    st.session_state.current_salary = int(current_salary)
    st.session_state.prediction = predict_salary(st.session_state.profile_key)
    st.session_state.profile_submitted = True
    st.session_state.negotiation_result = None


# LANDING PAGE (before a profile is submitted)

hero("AI Negotiation Assistant", "Pay Her Right",
     "Data-backed salary insight, bias detection and negotiation support for women in Nigerian tech.")

if not st.session_state.profile_submitted:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        md('<div class="welcome"><div class="welcome-title">Welcome</div>'
           '<div class="welcome-sub">Three steps to a confident salary conversation</div>'
           '<div class="step"><b>1.</b> Complete your profile in the sidebar</div>'
           '<div class="step"><b>2.</b> Press <b>Submit Profile</b> to see your fair market salary</div>'
           '<div class="step"><b>3.</b> Check for gender bias, then rehearse your ask</div>'
           '<div class="card-sub" style="margin-top:1rem;">Authorised use only. Estimates are decision aids, not guarantees.</div></div>')
    md('<div class="tile-grid">'
       '<div class="tile"><div class="tile-icon">\U0001F4B0</div><div class="tile-title">Predict</div><div class="tile-body">A Random Forest + XGBoost ensemble estimates your fair annual salary with an honest error range.</div></div>'
       '<div class="tile"><div class="tile-icon">\u2696\uFE0F</div><div class="tile-title">Detect</div><div class="tile-body">SHAP, a matched-pair test and three fairness metrics reveal whether gender moves the number.</div></div>'
       '<div class="tile"><div class="tile-icon">\U0001F91D</div><div class="tile-title">Negotiate</div><div class="tile-body">Simulate asks, see success odds, and get Graph RAG advice grounded in market facts.</div></div></div>')
    if bias_artifacts:
        kpi_row([kpi("Model R\u00b2", f"{bias_artifacts['test_r2']:.3f}", "on held-out test data"),
                 kpi("Typical error (MAE)", ngn(bias_artifacts["test_mae"]), "per year"),
                 kpi("Matched-pair gap", f"{bias_artifacts['matched_pair_mean_gap_pct']:.2f}%", "model-wide, same profile"),
                 kpi("Model", "Ensemble", bias_artifacts["model_name"].replace("Combined Ensemble ", "").strip("()"))])
    st.stop()

if model is None or bias_artifacts is None:
    st.error("Required model artefacts are missing. Check the file names at the top of app.py.")
    st.stop()


# RESULTS

pkey = st.session_state.profile_key
prow = dict(pkey)
prediction = st.session_state.prediction
core = cached_core(pkey)
fair = float(prediction["predicted_salary"])
mae = bias_artifacts["test_mae"]

kpi_row([kpi("Fair annual salary", ngn(fair), f"{st.session_state.full_name} \u00b7 {prow['job_title']}"),
         kpi("Monthly equivalent", ngn(fair / 12), f"Range {ngn(prediction['range_low'] / 12)} to {ngn(prediction['range_high'] / 12)}"),
         kpi("Gender gap (this profile)", f"{core['gap_pct']:.2f}%", "female vs identical male profile",
             "down" if core["gap_pct"] > 0.5 else "up"),
         kpi("Model confidence", f"{prediction['confidence_pct']}%", f"R\u00b2 {bias_artifacts['test_r2']:.3f}")])

tab_pred, tab_bias, tab_neg = st.tabs(["\U0001F4B0 Salary Prediction", "\u2696\uFE0F Bias Report", "\U0001F91D Negotiation Simulator"])


@fragment
def importance_section(records):
    with card("importance"):
        ctitle("Feature Importance", "What drives predictions across the whole trained model (not just your profile).")
        fi = pd.DataFrame(records)
        fi["label"] = (fi["feature"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
                       .str.replace("_", " ", regex=False))
        n = st.slider("Features to show", 5, min(15, len(fi)), min(10, len(fi)), key="fi_n")
        show(fig_importance(fi, n), "chart_fi", 380)


@fragment
def whatif_section(key, mode):
    base = dict(key)
    with card(f"whatif_{mode}"):
        if mode == "salary":
            ctitle("What-If Explorer", "Change one factor at a time and see how the model's fair salary responds.")
        else:
            ctitle("Gap Explorer", "Male vs female predictions for otherwise identical profiles, across one factor.")
        dim = st.selectbox("Factor to explore", list(DIMENSIONS), key=f"dim_{mode}")
        df = cached_sweep(key, dim)
        if mode == "salary":
            show(fig_whatif(df, dim, base["gender"]), f"chart_whatif_{mode}", 360)
        else:
            show(fig_gap_explorer(df, dim), f"chart_gapexp_{mode}", 380)



# TAB 1: SALARY PREDICTION

with tab_pred:
    hero("Step 1", "Your Fair Market Salary", "What the model estimates you should earn, and how confident it is.", big=False)
    col_main, col_side = st.columns([2, 1])
    breakdown = pay_breakdown(fair)

    with col_main:
        with card("salary"):
            md(f'<div class="card-sub">ESTIMATED FAIR SALARY</div><div class="salary-big">{ngn(fair)} <small>per year</small></div>'
               f'<div class="chips"><span class="chip">Range {ngn(prediction["range_low"])} to {ngn(prediction["range_high"])}</span>'
               f'<span class="chip">Confidence {prediction["confidence_pct"]}%</span></div>')
        with card("breakdown"):
            ctitle("Pay Breakdown", "Annual, monthly, weekly, daily and hourly, with the estimated range as whiskers.")
            md('<div class="period-grid">' + "".join(
                f'<div class="period"><div class="p-label">{p.upper()}</div><div class="p-val">{ngn(v)}</div></div>'
                for p, v in breakdown.items() if p != "Annual") + "</div>")
            show(fig_pay_breakdown(prediction), "chart_pay", 300)
            st.caption(PAY_PERIOD_NOTE)
            with st.expander("Estimated range for every pay period"):
                html_table(pd.DataFrame({
                    "Pay period": list(PAY_PERIODS),
                    "Low": [ngn(prediction["range_low"] / d) for d in PAY_PERIODS.values()],
                    "Predicted": [ngn(v) for v in breakdown.values()],
                    "High": [ngn(prediction["range_high"] / d) for d in PAY_PERIODS.values()]}))
        cur = st.session_state.current_salary
        if cur > 0:
            diff = fair - cur
            with card("compare"):
                ctitle("Comparison to Your Current Salary", "Hover the markers for the exact gap in NGN and %.")
                kpi_row([kpi("Current salary", ngn(cur)), kpi("Predicted fair salary", ngn(fair)),
                         kpi("Gap", ngn(diff), f"{diff / cur * 100:+.1f}%", "up" if diff >= 0 else "down")])
                show(fig_salary_compare(cur, prediction), "chart_compare", 230)
        importance_section(bias_artifacts["feature_importance"])
        whatif_section(pkey, "salary")

    with col_side:
        with card("summary"):
            ctitle("Profile Summary")
            chips = [("Name", st.session_state.full_name), ("Role", prow["job_title"]), ("Category", prow["job_category"]),
                     ("Experience", f"{prow['years_experience']} years"), ("Education", prow["education"]),
                     ("Location", prow["location"]), ("Company size", prow["company_size"]),
                     ("Remote ratio", f"{prow['remote_ratio']}%"), ("Skills", prow["num_skills"]),
                     ("Certifications", prow["certifications"])]
            md("".join(f'<div class="metric-row"><span style="color:var(--soft)">{k}</span>'
                       f'<span><b>{html.escape(str(v))}</b></span></div>' for k, v in chips))
        with card("radar"):
            ctitle("Profile Strength", "Your profile on each scale. The dotted ring is the scale midpoint, not a market average.")
            show(fig_radar(prow), "chart_radar", 340)
        with card("export"):
            ctitle("Export")
            report = (f"PAY HER RIGHT -- SALARY PREDICTION REPORT\n{'=' * 45}\nName: {st.session_state.full_name}\n"
                      f"Role: {prow['job_title']} ({prow['job_category']})\nExperience: {prow['years_experience']} years\n"
                      f"Education: {prow['education']}\nLocation: {prow['location']}\n\n"
                      f"Predicted Fair Salary (per year): {ngn(fair)}\n"
                      f"Range (per year): {ngn(prediction['range_low'])} - {ngn(prediction['range_high'])}\n"
                      f"Confidence: {prediction['confidence_pct']}%\n\nPay breakdown (predicted):\n"
                      + "".join(f"  {p}: {ngn(v)}\n" for p, v in breakdown.items()) + f"\n{PAY_PERIOD_NOTE}\n")
            st.download_button("Export Results (.txt)", data=report, file_name="pay_her_right_report.txt", **DL_W,
                               **_icon(st.download_button, ":material/download:"))


# TAB 2: BIAS REPORT

with tab_bias:
    hero("Step 2", "Is Gender Moving Your Number?", "Matched-pair testing and model-wide fairness metrics.", big=False)
    detected = bias_artifacts["bias_detected"]
    pill_cls = "pill-fail" if detected else "pill-pass"
    pill_txt = "\u26A0\uFE0F Gender Bias Detected" if detected else "\u2705 No Significant Bias Detected"
    md(f'<div style="text-align:center"><span class="pill {pill_cls}">{pill_txt}</span></div>')
    colA, colB = st.columns(2)
    with colA:
        with card("pair"):
            ctitle("Matched-Pair Check For This Profile", "Same profile, only the gender field changed. Whiskers show the model's typical error (MAE).")
            kpi_row([kpi("If listed as Male", ngn(core["male"])), kpi("If listed as Female", ngn(core["female"])),
                     kpi("Gap", f"{core['gap_pct']:.2f}%", "lower for female" if core["gap_pct"] > 0 else "no penalty",
                         "down" if core["gap_pct"] > 0.5 else "up")])
            show(fig_matched_pair(core, mae), "chart_pair", 340)
    with colB:
        with card("fairness"):
            ctitle("Fairness Metrics (Model-Wide)", "Hover each bar for its target and pass/fail status.")
            fm, tg = bias_artifacts["fairness_metrics"], bias_artifacts["targets_passed"]
            rows = [("Equalized Odds Difference", fm["equalized_odds_difference"], "< 0.05", tg["equalized_odds_difference"]),
                    ("Demographic Parity Ratio", fm["demographic_parity_ratio"], "> 0.80", tg["demographic_parity_ratio"]),
                    ("Disparate Impact", fm["disparate_impact"], "> 0.80", tg["disparate_impact"])]
            md("".join(f'<div class="metric-row"><span>{l}</span><span><b>{v:.3f}</b> &nbsp;target {t_} &nbsp;'
                       f'<span class="{"ok" if p else "bad"}">{"PASS" if p else "FAIL"}</span></span></div>' for l, v, t_, p in rows))
            show(fig_fairness(fm, tg), "chart_fair", 250)
    with card("gaphist"):
        if "matched_pair_gaps" in bias_artifacts:
            gaps, source = bias_artifacts["matched_pair_gaps"], "test set"
        else:
            gaps, source = cached_gap_distribution(pkey), "variations of your profile"
        ctitle("Gap Distribution", f"How the male-female gap is spread across {source}. Dashed line = your gap.")
        show(fig_gap_hist(gaps, core["gap_pct"], bias_artifacts.get("matched_pair_mean_gap_pct"), source), "chart_hist", 330)
        if source != "test set":
            st.caption("The notebook saves only the mean gap, so this histogram is computed live from the model across "
                       "variations of your profile. To plot the full test-set distribution, add "
                       "`matched_pair_gaps` (a list of % gaps) to bias_artifacts.json.")
    whatif_section(pkey, "gap")
    with card("reco"):
        ctitle("Recommendation")
        if detected:
            st.write(f"A residual gender gap of about {bias_artifacts['matched_pair_mean_gap_pct']:.1f}% was found across matched "
                     f"profiles in the model's test set. This appears influenced by factors beyond qualifications alone. "
                     f"Consider reviewing compensation bands for this role and use the Negotiation Simulator to prepare a data-backed ask.")
        else:
            st.write("No significant gender bias was detected in current salary predictions.")



# TAB 3: NEGOTIATION SIMULATOR

def _set_ask(key, value):
    st.session_state[key] = value


@fragment
def negotiation_section(key, fair_salary):
    prow_ = dict(key)
    role, loc = prow_["job_title"], prow_["location"]
    skey = f"ask_{int(fair_salary)}"
    lo_v, hi_v = int(round(fair_salary * 0.9, -4)), int(round(fair_salary * 1.5, -4))
    if skey not in st.session_state:
        st.session_state[skey] = int(min(max(round(fair_salary * 1.10, -4), lo_v), hi_v))

    with card("ask"):
        ctitle("Set Your Asking Amount", f"{role} | {prow_['years_experience']} years experience | {loc}")
        ask = st.slider("Asking amount (NGN per year)", min_value=lo_v, max_value=hi_v, step=10000, key=skey)
        prob, pct_above = calculate_negotiation_probability(fair_salary, ask)
        score = aggressiveness(pct_above)
        label = "Conservative" if score < 35 else "Assertive" if score < 65 else "Aggressive"
        kpi_row([kpi("Your ask", ngn(ask), f"{ngn(ask / 12)} per month"),
                 kpi("Versus market", f"{pct_above:+.1f}%", f"fair salary {ngn(fair_salary)}", "up" if pct_above >= 0 else "down"),
                 kpi("Success probability", f"{prob}%", "calibrated estimate, not a guarantee")])
        st.progress(prob / 100)
    g1, g2 = st.columns([2, 1])
    scenarios = {f"{p}% above": round(fair_salary * (1 + p / 100), -3) for p in (5, 10, 15)}
    with g1:
        with card("curve"):
            ctitle("Success Probability Curve", "Shaded band = sensitivity to the curve's steepness and midpoint (the curve is calibrated, not learned).")
            show(fig_prob_curve(fair_salary, ask, scenarios), "chart_curve", 380)
    with g2:
        with card("gauge"):
            ctitle("Aggressiveness", f"{label}: 0 = at market, 100 = 30%+ above market")
            show(fig_gauge(score), "chart_gauge", 280)
    with card("scenarios"):
        ctitle("Scenario Comparison", "Click a scenario to move the slider to that ask.")
        show(fig_scenarios(fair_salary, ask), "chart_scen", 340)
        cols = st.columns(3)
        for col, (lbl, val) in zip(cols, scenarios.items()):
            p_, _ = calculate_negotiation_probability(fair_salary, val)
            v_clip = int(min(max(round(val, -4), lo_v), hi_v))
            col.button(f"{lbl}: {ngn(val)} ({p_}%)", key=f"scn_{lbl}", on_click=_set_ask, args=(skey, v_clip), **BTN_W)
    with card("treemap"):
        ctitle("Anatomy of Your Ask", "Sequential counterfactual build-up from the model: baseline -> experience -> skills & certifications -> location -> your negotiation margin.")
        att = cached_attribution(key)
        show(fig_treemap(att, ask), "chart_tree", 360)
        st.caption(f"Baseline = 0 years of experience, no listed skills or certifications, in the lowest-paying location "
                   f"for your profile ({att['cheapest']}). Order of steps affects the split.")
    with card("advice"):
        h1, h2 = st.columns([3, 1])
        with h1:
            ctitle("Negotiation Advice", "Graph RAG: facts retrieved from the knowledge graph, then written up for you.")
        get_advice = h2.button("Get Advice", key="get_advice", **BTN_W, **_icon(st.button, ":material/login:"))
        facts = query_negotiation_graph(negotiation_graph, role, loc, pct_above)
        if get_advice:
            with st.spinner("Consulting the negotiation knowledge graph..."):
                text, used_ai = generate_negotiation_advice(role, loc, fair_salary, ask, prob, facts)
            st.session_state.negotiation_result = {"advice": text, "used_ai": used_ai, "facts": facts,
                                                   "ask": ask, "prob": prob}
        res = st.session_state.negotiation_result
        if res:
            if not res["used_ai"]:
                st.caption("Using template-based advice (no OpenAI API key configured).")
            md(f'<div class="advice">{html.escape(res["advice"])}</div>')
            with st.expander("Retrieved facts from the negotiation graph"):
                md("".join(f'<div class="fact">{html.escape(f)}</div>' for f in res["facts"]))
            neg = (f"PAY HER RIGHT -- NEGOTIATION ADVICE\n{'=' * 40}\nRole: {role}\nLocation: {loc}\n"
                   f"Fair Market Salary: {ngn(fair_salary)}\nAsking Amount (per year): {ngn(res['ask'])}\n"
                   f"Asking Amount (per month): {ngn(res['ask'] / 12)}\nSuccess Probability: {res['prob']}%\n\nAdvice:\n{res['advice']}\n")
            st.download_button("Export Advice (.txt)", data=neg, file_name="negotiation_advice.txt", **DL_W,
                               **_icon(st.download_button, ":material/download:"))
    if negotiation_graph is not None and role in negotiation_graph:
        with card("graph"):
            ctitle("Knowledge Graph Explorer", "The facts behind the advice. Hover a node to read it; greyed nodes were not retrieved for your location.")
            show(fig_graph(negotiation_graph, role, loc, pct_above), "chart_graph", 420)


with tab_neg:
    hero("Step 3", "Rehearse Your Ask", "Explore asking amounts, odds of success and tailored negotiation advice.", big=False)
    negotiation_section(pkey, fair)
