"""
Streamlit UI for the Europe Job Search Agent.

    streamlit run app.py
"""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

import job_agent as agent

st.set_page_config(page_title="Europe Job Search Agent", page_icon="🌍", layout="wide")

ALL_COUNTRIES = [c[0] for c in agent.REST_OF_EUROPE]
SAMPLE_DIR = Path(__file__).parent / "demo" / "output"

# Hosted demo (Streamlit Community Cloud runs apps from /mount/src, Hugging Face sets SPACE_ID): every visitor
# gets a private temporary folder and uses their own OpenAI key. Searches run in a separate process, so
# visitors never share a profile, key or results.
HOSTED = bool(os.getenv("JOB_AGENT_HOSTED") or os.getenv("SPACE_ID")
              or str(Path(__file__).resolve()).startswith("/mount/src/"))
if HOSTED:
    if "home" not in st.session_state:
        st.session_state.home = Path(tempfile.mkdtemp(prefix="job-agent-"))
    HOME = st.session_state.home
else:
    HOME = agent.HOME_DIR
OUTPUT_DIR, PROFILE_PATH = HOME / "output", HOME / "profile.json"
COMPANY_TYPES = ["scaleup", "startup", "big_tech_international", "corporate", "consultancy",
                 "recruitment_agency", "unknown"]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

class LiveLog:
    """File-like object that mirrors print() output into a Streamlit code block.
    Worker threads only append; the UI is refreshed from the script thread."""

    def __init__(self, placeholder):
        self.placeholder, self.lines, self.owner = placeholder, [], threading.current_thread()

    def write(self, text):
        self.lines.extend(line for line in text.splitlines() if line.strip())
        if threading.current_thread() is self.owner:
            self.placeholder.code("\n".join(self.lines[-25:]), language=None)

    def flush(self):
        pass


def list_runs():
    """Run folders, newest first. The hosted demo adds the bundled sample run after the visitor's own runs."""
    def runs(folder):
        return sorted((str(p.parent) for p in folder.glob("*/results.json")), reverse=True)
    return runs(OUTPUT_DIR) + (runs(SAMPLE_DIR) if HOSTED else [])


def run_label(run_dir: str) -> str:
    """'.../2026-10-13_0802' -> 'Run of 2026-10-13 08:02 · 57 jobs' (older folders are date-only)."""
    date, _, hhmm = Path(run_dir).name.partition("_")
    try:
        count = len(json.loads((Path(run_dir) / "results.json").read_text())["jobs"])
    except Exception:
        count = "?"
    sample = "Sample run (a Senior AI Engineer's CV) · " if Path(run_dir).parent == SAMPLE_DIR else "Run of "
    return f"{sample}{date}{' ' + hhmm[:2] + ':' + hhmm[2:] if hhmm else ''} · {count} jobs"


def read_profile():
    return json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else None


def run_agent(args, log, api_key):
    """Hosted mode: runs job_agent.py in the visitor's folder with the visitor's key, streaming its output."""
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env.update(OPENAI_API_KEY=api_key, JOB_AGENT_HOME=str(HOME), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-u", agent.__file__, "--no-open", *map(str, args)], env=env,
                            cwd=HOME, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log.write(line)
    if proc.wait():
        raise RuntimeError(log.lines[-1] if log.lines else "the search process failed")


@st.cache_data(show_spinner=False)
def load_run(run_dir: str, _mtime: float):
    data = json.loads((Path(run_dir) / "results.json").read_text())
    # re-score and merge here too, so runs saved before the latest ranking and de-duplication look the same
    rows = [agent.rescore_saved(r) for r in data["jobs"] if not agent._TOO_JUNIOR.search(r.get("title") or "")]
    df = pd.DataFrame(agent.merge_duplicates(rows, score=lambda r: r["rank_score"]))
    if not df.empty:
        df["country_label"] = df.apply(
            lambda r: r["country"] + (f"  (+ {', '.join(r['also_in'])})" if r.get("also_in") else ""), axis=1)
        df["posted_label"] = df["posted"].map(lambda p: agent._posted_label(p)[0])
        df["why"] = df["signals"].map(lambda s: " · ".join(s))
        df["new"] = df.get("is_new", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        df["title_label"] = df.apply(lambda r: ("🆕 " if r["new"] else "") + r["title"], axis=1)
    return data["meta"], df


def results_table(df: pd.DataFrame, key: str):
    if df.empty:
        st.info("No jobs match these filters.")
        return
    view = df[["title_label", "company", "country_label", "hiring_chance", "cv_fit", "rank_score", "posted_label",
               "company_type", "visa_sponsorship", "local_language", "source", "url", "reason", "why"]]
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=min(38 * len(view) + 40, 720),
        column_config={
            "title_label": st.column_config.TextColumn("Role", width="large", help="🆕 = not in any earlier run"),
            "company": st.column_config.TextColumn("Company"),
            "country_label": st.column_config.TextColumn("Country", help="(+ …) = same job also listed there"),
            "hiring_chance": st.column_config.ProgressColumn(
                "Hiring chance", min_value=0, max_value=100, format="%d",
                help="How likely this employer hires a non-German-speaking candidate who needs a visa"),
            "cv_fit": st.column_config.ProgressColumn("CV fit", min_value=0, max_value=100, format="%d"),
            "rank_score": st.column_config.NumberColumn("Rank", format="%.1f",
                                                        help="55% hiring chance + 45% CV fit"),
            "posted_label": st.column_config.TextColumn("Posted"),
            "company_type": st.column_config.TextColumn("Company type"),
            "visa_sponsorship": st.column_config.TextColumn("Visa"),
            "local_language": st.column_config.TextColumn("Local language"),
            "source": st.column_config.TextColumn("Source"),
            "url": st.column_config.LinkColumn("Apply", display_text="Open ↗"),
            "reason": st.column_config.TextColumn("Summary", width="large"),
            "why": st.column_config.TextColumn("Score breakdown", width="large"),
        },
    )
    st.download_button("Download as CSV", view.to_csv(index=False).encode(), f"{key}.csv",
                       "text/csv", key=f"dl-{key}")


# --------------------------------------------------------------------------------------
# Sidebar: profile + search settings
# --------------------------------------------------------------------------------------

profile = read_profile()
api_key = ""

with st.sidebar:
    if HOSTED:
        st.header("0 · Your OpenAI key")
        api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...",
                                help="Needed to read your CV and score jobs. A search costs roughly "
                                     "$0.05–0.30 on gpt-4.1-mini. The key stays in this browser session "
                                     "and is never saved to disk.").strip()
    st.header("1 · Your profile")
    cv_file = st.file_uploader("CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"],
                               help="Read once. Upload again only when your CV changes.")
    if cv_file and st.button("Analyse CV", type="primary", width="stretch", disabled=HOSTED and not api_key):
        (HOME / "data").mkdir(parents=True, exist_ok=True)
        cv_path = HOME / "data" / f"uploaded_cv{Path(cv_file.name).suffix.lower()}"
        cv_path.write_bytes(cv_file.getvalue())
        with st.spinner("Reading your CV ..."):
            try:
                if HOSTED:
                    run_agent(["--cv", cv_path, "--profile-only"], LiveLog(st.empty()), api_key)
                    profile = read_profile()
                else:
                    profile = agent.load_profile(cv_path)
                st.success("Profile saved")
            except (SystemExit, RuntimeError) as e:  # unreadable / scanned PDF, bad key
                st.error(str(e))
    if HOSTED and cv_file and not api_key:
        st.caption("Add your OpenAI key above to analyse the CV.")

    if profile:
        st.markdown(f"**{profile.get('name', '')}** · {profile.get('years_experience', '?')} yrs · "
                    f"{profile.get('seniority', '')}")
        st.caption(profile.get("summary", ""))
        terms = st.text_area("Job titles to search (one per line)", "\n".join(profile["search_terms"]),
                             height=140, help="Write them the way European employers title their ads. "
                                              "3–5 titles is the sweet spot - each one adds searches.")
        new_terms = [t.strip() for t in terms.splitlines() if t.strip()]
        if new_terms != profile["search_terms"] and st.button("Save job titles", width="stretch"):
            profile["search_terms"] = new_terms
            PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
            st.success("Saved - title variants will be refreshed on the next search")
        with st.expander(f"Titles that count as a match ({len(agent.match_phrases(profile))})"):
            st.caption("Your titles + AI-generated variants. Used to filter the smaller boards and career "
                       "pages; LinkedIn/Indeed/Glassdoor search your titles directly.")
            st.write(", ".join(agent.match_phrases(profile)))
    else:
        st.info("Upload your CV to get started." + (" Meanwhile, browse the sample run →" if HOSTED else ""))

    st.header("2 · Search settings")
    saved = agent.load_settings() if not HOSTED else {
        **agent.DEFAULT_SETTINGS, "countries": ["United Kingdom", "Netherlands"], "results_per_query": 15}
    countries = st.multiselect("Countries besides Germany", ALL_COUNTRIES,
                               default=[c for c in saved["countries"] if c in ALL_COUNTRIES],
                               help="Germany is always searched. More countries = longer runs.")
    sources = st.multiselect("Job sources", agent.SOURCES, default=saved["sources"],
                             format_func=agent.SOURCE_LABELS.get)
    hours = st.select_slider("Posted within", [24, 48, 72, 168], value=saved["hours"],
                             format_func=lambda h: f"{h // 24} day{'s' if h > 24 else ''}",
                             help="48h overlaps daily runs, so a missed or failed day loses nothing. "
                                  "Jobs you already saw in earlier runs are not marked 🆕.")
    per_query = st.slider("LinkedIn/Indeed/Glassdoor results per search", 10, 100, saved["results_per_query"],
                          step=5, help="Per job title × country. Higher finds more but runs longer.")
    searches = len(profile["search_terms"]) * (len(countries) + 1) if profile else 0
    if "linkedin-indeed" in sources and searches:
        st.caption(f"≈ {searches} LinkedIn/Indeed searches · roughly {max(1, searches // 4)}–"
                   f"{max(2, searches // 2)} min")
    if "careers" in sources and agent.COMPANIES_PATH.exists():
        n = len(json.loads(agent.COMPANIES_PATH.read_text())["companies"])
        st.caption(f"Career pages: {n} companies (edit companies.json to add your targets)")
    current = dict(countries=countries, sources=sources, hours=hours, results_per_query=per_query)
    if not HOSTED and current != {k: saved[k] for k in current} and st.button("Save as default settings",
                                                                               width="stretch"):
        agent.save_settings(**current)
        st.success("Saved - the daily run uses these")
    run_clicked = st.button(f"🔎 Find jobs from the last {hours // 24} day{'s' if hours > 24 else ''}",
                            type="primary", width="stretch",
                            disabled=not (profile and sources and (api_key or not HOSTED)))

if not HOSTED and sys.platform == "darwin":  # the daily run uses macOS launchd
    with st.sidebar:
        st.header("3 · Daily automatic run")
        scheduled = agent.schedule_status()
        if scheduled:
            st.success(f"Runs every day at {scheduled} with the saved settings")
            if agent.SCHEDULE_LOG.exists():
                with st.expander("Last run log"):
                    st.code("\n".join(agent.SCHEDULE_LOG.read_text().splitlines()[-30:]), language=None)
        when = st.time_input("Run every day at", value=pd.Timestamp(f"2000-01-01 {scheduled or '08:00'}").time(),
                             step=900)
        b1, b2 = st.columns(2)
        if b1.button("Schedule" if not scheduled else "Update", width="stretch", disabled=not profile):
            try:
                st.toast(agent.install_schedule(when.hour, when.minute))
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if scheduled and b2.button("Remove", width="stretch"):
            st.toast(agent.remove_schedule())
            st.rerun()
        st.caption("If the Mac is asleep at that time, the run starts when it wakes up.")


# --------------------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------------------

st.title("🌍 Europe Job Search Agent")
st.caption("Fresh jobs across Germany, the UK and Europe - ranked by how likely "
           "the employer is to hire an international candidate who needs a visa and doesn't speak German.")

if HOSTED:
    st.info("**Live demo.** The sample run below was made for a Senior AI Engineer's CV, so you can try the "
            "filters straight away. To search for yourself, add your OpenAI key and CV in the sidebar. Your CV, "
            "profile and results go to a temporary folder only your session can see, and are wiped when the Space "
            "restarts. For daily automatic runs, [run it on your own machine]"
            "(https://github.com/JYOTSNACHOUDHARY/ai-job-search-agent).", icon="👋")

if run_clicked:
    with st.status("Searching ... (usually 3–10 minutes)", expanded=True) as status:
        log = LiveLog(st.empty())
        try:
            if HOSTED:
                run_agent(["--sources", *sources, "--countries", *countries, "--results-per-query", per_query,
                           "--hours", hours], log, api_key)
            else:
                with contextlib.redirect_stdout(log):
                    agent.run_search(profile, sources, countries, per_query, hours)
            status.update(label="Search finished", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"Search failed: {e}", state="error")
    load_run.clear()

runs = list_runs()
if not runs:
    st.info("No results yet. Upload your CV in the sidebar, then click **Find jobs**.")
    st.stop()

run_dir = st.selectbox("Results from", runs, index=0, label_visibility="collapsed", format_func=run_label)
run_date = Path(run_dir).name
meta, jobs = load_run(run_dir, (Path(run_dir) / "results.json").stat().st_mtime)
if jobs.empty:
    st.warning("That run found no jobs.")
    st.stop()

with st.expander("Filters", expanded=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1.4])
    min_fit = c1.slider("Min CV fit", 0, 100, agent.MIN_FIT, step=5)
    min_chance = c2.slider("Min hiring chance", 0, 100, 0, step=5)
    types = c3.multiselect("Company type", COMPANY_TYPES, default=[],
                           placeholder="All types", format_func=lambda t: t.replace("_", " "))
    query = c4.text_input("Search title / company", placeholder="e.g. LLM, Zalando")
    c5, c6, c7, c8, c9 = st.columns(5)
    only_new = c5.toggle("🆕 Only new jobs", value=True,
                         help="Runs cover 48 hours, so about half of each run repeats yesterday's jobs. "
                              "Turn off to see everything in this run.")
    hide_no_visa = c6.toggle("Hide 'no visa sponsorship'", value=True)
    hide_lang = c7.toggle("Hide 'local language required'", value=False)
    hide_agencies = c8.toggle("Hide recruitment agencies", value=False)
    hide_corporates = c9.toggle("Hide traditional corporates", value=False,
                                help="Large non-tech companies (insurers, banks, retailers ...), which more often need German")

f = jobs[(jobs.cv_fit >= min_fit) & (jobs.hiring_chance >= min_chance)]
if types:
    f = f[f.company_type.fillna("unknown").isin(types)]
if query:
    q = query.lower()
    f = f[f.title.str.lower().str.contains(q, regex=False) | f.company.str.lower().str.contains(q, regex=False)]
if only_new:
    f = f[f.new]
if hide_no_visa:
    f = f[f.visa_sponsorship != "explicitly_not_offered"]
if hide_lang:
    f = f[f.local_language != "required"]
if hide_agencies:
    f = f[f.company_type != "recruitment_agency"]
if hide_corporates:
    f = f[f.company_type != "corporate"]
f = f.sort_values("rank_score", ascending=False)

germany, rest = f[f.region == "Germany"], f[f.region != "Germany"]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Matching jobs", len(f), help=f"{len(jobs)} evaluated in this run")
m2.metric("🆕 New", int(f.new.sum()), help="Not in any earlier run")
m3.metric("Germany", len(germany))
m4.metric("Rest of Europe & UK", len(rest))
m5.metric("High hiring chance (≥ 70)", int((f.hiring_chance >= 70).sum()))

tab_de, tab_eu = st.tabs([f"🇩🇪 Germany ({len(germany)})", f"🇪🇺 Rest of Europe & UK ({len(rest)})"])
with tab_de:
    results_table(germany, f"germany_{run_date}")
with tab_eu:
    results_table(rest, f"rest_of_europe_uk_{run_date}")

st.caption(f"Searched: {', '.join(meta['search_terms'])} · posted within {meta.get('hours', 24)}h · "
           f"evaluated with {meta['model']} · run at {meta['created'].replace('T', ' ')}")

with st.expander("How the ranking works"):
    st.markdown("""
**Rank = 55% hiring chance + 45% CV fit.** CV fit comes from an LLM comparing the ad with your CV.
Hiring chance starts at 50 and moves with what the ad and official registers say:

| Signal | Points |
|---|---|
| Visa sponsorship offered | +25 |
| On UK / Netherlands official visa-sponsor register | +20 |
| Startup / scale-up / big international tech | +18 / +18 / +5 |
| Traditional corporate / consultancy | −10 / −5 |
| Ad in English · English working language · no local language needed | +10 · +10 · +12 |
| Relocation support | +12 |
| Ad in German | −25 |
| Local language required | −35 |
| UK / NL employer *not* on sponsor register (and no visa mentioned) | −25 |
| Remote only for residents | −20 |
| "No visa sponsorship" | −50 |

In Germany any employer can hire you on an EU Blue Card (2026 minimum for IT roles: €45,934),
so language is the main barrier there. In the UK and Netherlands the employer must be a licensed sponsor.
""")
