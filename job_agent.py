"""
Europe Job Search Agent
=======================

One-time input: your CV.  Every run: all jobs posted in the last 48 hours (configurable) that match
your profile, split into (1) Germany and (2) Rest of Europe + UK, ranked so that roles
most likely to hire a non-EU (Indian) candidate who doesn't speak German come first.

Usage  (or use the Streamlit UI:  streamlit run app.py)
-----
    # first run - parses your CV once and saves profile.json
    python job_agent.py --cv ~/Documents/my_cv.pdf

    # every later run (e.g. daily from cron) - reuses the saved profile
    python job_agent.py

Pipeline
--------
1. Profile   : LLM reads the CV -> target job titles, skills, seniority (cached)
2. Collect   : LinkedIn + Indeed + Glassdoor (python-jobspy), company career pages
               (Greenhouse / Lever / Ashby / SmartRecruiters, companies.json), Arbeitnow API,
               Berlin Startup Jobs, Make it in Germany / Bundesagentur für Arbeit, EURES,
               XING, NextJobFinder, Adzuna API (optional, if ADZUNA_APP_ID / ADZUNA_APP_KEY are set)
3. Enrich    : official visa-sponsor registers - UK Home Office register of licensed
               sponsors, Netherlands IND register of recognised sponsors
4. Evaluate  : LLM reads each posting -> CV fit, German requirement, visa / relocation
               statements, startup vs corporate vs agency
5. Rank      : transparent rule-based "international friendliness" score + CV fit
6. Report    : results.json, HTML report + two CSVs in ./output/<date>_<time>/; jobs not seen
               in earlier runs are flagged NEW
7. Schedule  : --schedule 08:00 runs it daily (macOS launchd)
"""

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import functools
import html
import io
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

# JOB_AGENT_HOME moves all personal files (profile, settings, results, caches) to another folder.
# The hosted demo gives every visitor their own folder and their own API key, which a .env must not override.
HOME_DIR = Path(os.environ["JOB_AGENT_HOME"]) if os.getenv("JOB_AGENT_HOME") else None
load_dotenv(override=HOME_DIR is None)

HERE = Path(__file__).parent
HOME_DIR = HOME_DIR or HERE
DATA_DIR = HOME_DIR / "data"
OUTPUT_DIR = HOME_DIR / "output"
PROFILE_PATH = HOME_DIR / "profile.json"
EVAL_CACHE_PATH = DATA_DIR / "evaluations_cache.json"
SEEN_PATH = DATA_DIR / "seen_jobs.json"
COMPANIES_PATH = HERE / "companies.json"
SETTINGS_PATH = HOME_DIR / "settings.json"

MODEL = os.getenv("JOB_AGENT_MODEL", "gpt-4.1-mini")
MIN_FIT = int(os.getenv("JOB_AGENT_MIN_FIT", "45"))  # hide jobs that don't match the CV
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130 Safari/537.36")
HTTP_HEADERS = {"User-Agent": BROWSER_UA}

# Countries searched on LinkedIn / Indeed. Indeed needs its own country name.
# (display name, LinkedIn location, Indeed country)
GERMANY = ("Germany", "Germany", "germany")
REST_OF_EUROPE = [
    ("United Kingdom", "United Kingdom", "uk"),
    ("Netherlands", "Netherlands", "netherlands"),
    ("Ireland", "Ireland", "ireland"),
    ("Switzerland", "Switzerland", "switzerland"),
    ("Austria", "Austria", "austria"),
    ("France", "France", "france"),
    ("Spain", "Spain", "spain"),
    ("Portugal", "Portugal", "portugal"),
    ("Poland", "Poland", "poland"),
    ("Sweden", "Sweden", "sweden"),
    ("Denmark", "Denmark", "denmark"),
    ("Finland", "Finland", "finland"),
    ("Norway", "Norway", "norway"),
    ("Belgium", "Belgium", "belgium"),
    ("Czech Republic", "Czechia", "czech republic"),
    ("Luxembourg", "Luxembourg", "luxembourg"),
    ("Italy", "Italy", "italy"),
]
SOURCES = ["linkedin-indeed", "glassdoor", "careers", "arbeitnow", "berlinstartupjobs", "bundesagentur",
           "eures", "xing", "nextjobfinder", "adzuna"]
SOURCE_LABELS = {"linkedin-indeed": "LinkedIn + Indeed", "glassdoor": "Glassdoor",
                 "careers": "Company career pages (companies.json)", "arbeitnow": "Arbeitnow",
                 "berlinstartupjobs": "Berlin Startup Jobs", "bundesagentur": "Make it in Germany (Arbeitsagentur)",
                 "eures": "EURES (EU portal)", "xing": "XING", "nextjobfinder": "NextJobFinder",
                 "adzuna": "Adzuna (needs API key)"}

ADZUNA_COUNTRIES = {"de": "Germany", "gb": "United Kingdom", "nl": "Netherlands", "fr": "France",
                    "at": "Austria", "be": "Belgium", "ch": "Switzerland", "es": "Spain",
                    "it": "Italy", "pl": "Poland"}

# EURES uses ISO country codes
EURES_COUNTRIES = {"DE": "Germany", "NL": "Netherlands", "IE": "Ireland", "CH": "Switzerland",
                   "AT": "Austria", "FR": "France", "ES": "Spain", "PT": "Portugal", "PL": "Poland",
                   "SE": "Sweden", "DK": "Denmark", "FI": "Finland", "NO": "Norway", "BE": "Belgium",
                   "CZ": "Czech Republic", "LU": "Luxembourg", "IT": "Italy"}

GERMAN_PLACES = ["germany", "deutschland", "berlin", "munich", "münchen", "hamburg", "frankfurt",
                 "cologne", "köln", "stuttgart", "düsseldorf", "dusseldorf", "leipzig", "dresden",
                 "hannover", "nuremberg", "nürnberg", "karlsruhe", "bonn", "essen", "dortmund",
                 "bremen", "mannheim", "heidelberg", "potsdam", "aachen", "freiburg", "münster"]

# Map free-text locations (Arbeitnow) to countries, so sponsor registers can be applied
COUNTRY_PLACES = {
    "United Kingdom": ["united kingdom", "uk", "england", "scotland", "london", "manchester", "edinburgh",
                       "cambridge", "oxford", "bristol", "leeds", "birmingham", "glasgow"],
    "Netherlands": ["netherlands", "amsterdam", "rotterdam", "utrecht", "eindhoven", "the hague", "den haag"],
    "Ireland": ["ireland", "dublin", "cork"],
    "France": ["france", "paris", "lyon", "toulouse"],
    "Switzerland": ["switzerland", "zurich", "zürich", "geneva", "basel", "lausanne"],
    "Austria": ["austria", "vienna", "wien"],
    "Spain": ["spain", "madrid", "barcelona"],
    "Portugal": ["portugal", "lisbon", "porto"],
    "Poland": ["poland", "warsaw", "krakow", "kraków", "wroclaw"],
    "Sweden": ["sweden", "stockholm", "gothenburg", "malmö", "malmo"],
    "Denmark": ["denmark", "copenhagen", "aarhus"],
    "Finland": ["finland", "helsinki", "espoo"],
    "Norway": ["norway", "oslo"],
    "Belgium": ["belgium", "brussels", "antwerp", "ghent"],
    "Czech Republic": ["czech", "czechia", "prague", "brno"],
    "Luxembourg": ["luxembourg"],
    "Italy": ["italy", "milan", "milano", "rome", "turin", "torino"],
}

# Germany's EU Blue Card reduced threshold for IT specialists / shortage occupations, 2026
# (standard threshold is EUR 50,700). Source: BMI / § 18g AufenthG.
BLUE_CARD_MIN_EUR = 45_934


# --------------------------------------------------------------------------------------
# 1. Profile from CV
# --------------------------------------------------------------------------------------

def read_cv(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")


PROFILE_PROMPT = """You are an expert tech recruiter for the European market.
Read the candidate's CV and return JSON with exactly these keys:
- "name": string
- "summary": 2-sentence professional summary
- "years_experience": number
- "seniority": one of "mid", "senior", "lead", "principal", "manager"
- "search_terms": 3 to 5 job titles exactly as European employers would write them in job ads,
   most relevant first (e.g. "Senior Data Engineer", "Senior Backend Engineer")
- "core_skills": up to 15 most important hard skills
- "languages": spoken languages with level, e.g. ["English (fluent)", "Hindi (native)"]
- "domain": main industry/domain experience
"""


def _file_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_profile(cv_path: Path, client: OpenAI) -> dict:
    cv_text = read_cv(cv_path)
    if len(cv_text.strip()) < 200:
        raise SystemExit(f"Could not read enough text from {cv_path} - is it a scanned image PDF?")
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROFILE_PROMPT},
                  {"role": "user", "content": cv_text[:20000]}],
    )
    profile = json.loads(resp.choices[0].message.content)
    profile["cv_text"] = cv_text[:12000]
    profile["cv_sha256"] = _file_hash(cv_path)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    EVAL_CACHE_PATH.unlink(missing_ok=True)  # cached fit scores belong to the old CV
    return profile


# --------------------------------------------------------------------------------------
# 2. Collect jobs posted in the last 24 hours
# --------------------------------------------------------------------------------------

def _row(source, region, country, title, company, location, url, description, posted,
         company_size="", salary_eur=None):
    return {"source": source, "region": region, "country": country, "title": title or "",
            "company": company or "", "location": location or "", "url": url or "",
            "description": description or "", "posted": posted or "",
            "company_size": company_size or "", "salary_eur": salary_eur}


def collect_jobspy(search_terms, countries, region, results_per_query, hours=24):
    """LinkedIn + Indeed via python-jobspy (no login, public job pages).
    LinkedIn throttles hard (HTTP 429). Fetching every job page during the search used ~28 requests
    per search and got most searches cut off, so LinkedIn is searched without descriptions (about 1
    request per 10 jobs, so twice as many results fit) and fill_linkedin_descriptions() then fetches
    only the jobs that haven't been evaluated before, slowly."""
    from jobspy import scrape_jobs

    rows = []
    for name, linkedin_loc, indeed_country in countries:
        for term in search_terms:
            frames = []
            for site, wanted, describe in (("indeed", results_per_query, True),
                                           ("linkedin", results_per_query * 2, False)):
                try:
                    frames.append(scrape_jobs(
                        site_name=[site],
                        search_term=term,
                        location=linkedin_loc,
                        country_indeed=indeed_country,
                        hours_old=hours,
                        results_wanted=wanted,
                        fetch_description=describe,
                        description_format="markdown",
                        verbose=0,
                    ))
                except Exception as e:  # one failing query must not kill the run
                    print(f"   ! {name} / {term} / {site}: {e}")
            frames = [f for f in frames if len(f)]
            if not frames:
                print(f"   {name:<15} {term:<35}   0 jobs")
                continue
            df = pd.concat(frames, ignore_index=True)
            for _, j in df.iterrows():
                salary = None
                if str(j.get("currency", "")).upper() == "EUR" and pd.notna(j.get("min_amount")):
                    amount = float(j["min_amount"])
                    salary = amount * 12 if j.get("interval") == "monthly" else amount
                rows.append(_row(
                    source=str(j.get("site", "")).title(), region=region, country=name,
                    title=j.get("title"), company=j.get("company"),
                    location=j.get("location"), url=j.get("job_url_direct") or j.get("job_url"),
                    description=j.get("description") if pd.notna(j.get("description")) else "",
                    posted=str(j.get("date_posted", "")),
                    company_size=j.get("company_num_employees") if pd.notna(j.get("company_num_employees", None)) else "",
                    salary_eur=salary,
                ))
            counts = df["site"].value_counts() if len(df) else {}
            print(f"   {name:<15} {term:<35} {counts.get('linkedin', 0):>3} LinkedIn {counts.get('indeed', 0):>3} Indeed")
            time.sleep(random.uniform(4, 8))  # be polite - LinkedIn rate-limits aggressive scraping
    return rows


def fill_linkedin_descriptions(jobs):
    """Fetches the description of LinkedIn jobs not evaluated before, one page every few seconds.
    On a 429 it waits and retries once, then stops: those jobs are evaluated from the title and
    re-tried on the next run (evaluations without a description aren't cached)."""
    from bs4 import BeautifulSoup

    cache = _load_eval_cache()
    todo = [j for j in jobs if j["source"] == "Linkedin" and not j["description"] and j["url"] not in cache]
    if not todo:
        return jobs
    print(f"   Fetching {len(todo)} new LinkedIn job descriptions (slowly, to avoid LinkedIn's rate limit) ...")
    session, fetched = requests.Session(), 0
    session.headers.update({"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"})
    for job in todo:
        job_id = re.search(r"/jobs/view/(?:[^/?]*-)?(\d+)", job["url"])
        if not job_id:
            continue
        for attempt in range(2):
            try:
                r = session.get(f"https://www.linkedin.com/jobs/view/{job_id.group(1)}", timeout=30)
            except Exception:
                break
            if r.status_code != 429:
                break
            time.sleep(90)
        else:
            print(f"   ! LinkedIn is throttling - {len(todo) - fetched} descriptions left for the next run")
            return jobs
        div = BeautifulSoup(r.text, "html.parser").find("div", class_="show-more-less-html__markup")
        if div:
            job["description"] = div.get_text("\n", strip=True)
            fetched += 1
        time.sleep(random.uniform(2, 4))
    print(f"   Got {fetched} of {len(todo)} LinkedIn descriptions")
    return jobs


def collect_arbeitnow(profile, hours=24):
    """Arbeitnow: free API, Germany-focused, mostly English-speaking / visa-friendly companies."""
    cutoff = time.time() - hours * 3600
    rows, url = [], "https://www.arbeitnow.com/api/job-board-api"
    for _ in range(15):  # newest first; stop once we're past the window
        data = requests.get(url, headers=HTTP_HEADERS, timeout=30).json()
        page_jobs = data.get("data", [])
        for j in page_jobs:
            if j["created_at"] < cutoff:
                continue
            if not is_relevant(j["title"], profile):
                continue
            loc = j.get("location", "")
            country = _guess_country(loc)
            rows.append(_row(
                source="Arbeitnow", region="Germany" if country == "Germany" else "Rest of Europe & UK",
                country=country, title=j["title"], company=j["company_name"],
                location=loc + (" (remote)" if j.get("remote") else ""), url=j["url"],
                description=re.sub(r"<[^>]+>", " ", j.get("description", "")),
                posted=dt.datetime.fromtimestamp(j["created_at"], dt.timezone.utc).isoformat(timespec="minutes"),
            ))
        url = (data.get("links") or {}).get("next")
        if not url or (page_jobs and min(j["created_at"] for j in page_jobs) < cutoff):
            break
    print(f"   Arbeitnow: {len(rows)} relevant jobs")
    return rows


def collect_adzuna(search_terms, hours=24):
    """Adzuna: free API key from https://developer.adzuna.com (optional)."""
    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        print("   Adzuna: skipped (set ADZUNA_APP_ID and ADZUNA_APP_KEY in .env to enable)")
        return []
    rows = []
    for code, name in ADZUNA_COUNTRIES.items():
        for term in search_terms:
            try:
                r = requests.get(
                    f"https://api.adzuna.com/v1/api/jobs/{code}/search/1",
                    params={"app_id": app_id, "app_key": app_key, "what": term,
                            "max_days_old": _days(hours), "results_per_page": 50,
                            "content-type": "application/json"},
                    timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"   ! Adzuna {name} / {term}: {e}")
                continue
            for j in r.json().get("results", []):
                rows.append(_row(
                    source="Adzuna", region="Germany" if code == "de" else "Rest of Europe & UK",
                    country=name, title=j.get("title"), company=(j.get("company") or {}).get("display_name"),
                    location=(j.get("location") or {}).get("display_name"), url=j.get("redirect_url"),
                    description=j.get("description"), posted=j.get("created"),
                    salary_eur=j.get("salary_min") if code not in ("gb", "ch", "pl") else None,
                ))
            time.sleep(1)
    print(f"   Adzuna: {len(rows)} jobs")
    return rows


def collect_berlin_startup_jobs(profile, hours=24):
    """berlinstartupjobs.com RSS feed - Berlin startups, almost all English-speaking."""
    import feedparser

    cutoff = time.time() - hours * 3600
    rows = []
    for page in range(1, 4):
        feed = feedparser.parse(f"https://berlinstartupjobs.com/feed/?paged={page}", agent=BROWSER_UA)
        for e in feed.entries:
            published = time.mktime(e.published_parsed) if e.get("published_parsed") else 0
            if published < cutoff:
                continue
            title, _, company = html.unescape(e.title).rpartition(" // ")
            if not is_relevant(title, profile):
                continue
            body = e.content[0].value if e.get("content") else e.get("summary", "")
            rows.append(_row(source="Berlin Startup Jobs", region="Germany", country="Germany",
                             title=title, company=company, location="Berlin", url=e.link,
                             description=_strip_html(body), posted=e.published))
        if not feed.entries or all(time.mktime(e.published_parsed) < cutoff
                                   for e in feed.entries if e.get("published_parsed")):
            break
    print(f"   Berlin Startup Jobs: {len(rows)} relevant jobs")
    return rows


def collect_bundesagentur(profile, hours=24):
    """Federal Employment Agency job board (Bundesagentur für Arbeit) - the source behind the
    job listings on make-it-in-germany.com. Biggest German board; many ads are in German,
    which the ranking will push down."""
    api = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v6/jobs"
    # The API ignores `veroeffentlichtseit` (2 days and 30 days return the same jobs, median age ~4 weeks),
    # so every page is read and the window is applied to the first-publication date here.
    since = (dt.datetime.now() - dt.timedelta(hours=hours)).date().isoformat()
    found = {}
    for phrase in search_phrases(profile):
        for page in range(1, 11):
            try:
                r = requests.get(api, headers={**HTTP_HEADERS, "X-API-Key": "jobboerse-jobsuche"},
                                 params={"was": phrase, "veroeffentlichtseit": _days(hours), "angebotsart": 1,
                                         "size": 100, "page": page}, timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"   ! Bundesagentur / {phrase}: {e}")
                break
            results = r.json().get("ergebnisliste") or []
            for j in results:
                if ((j.get("datumErsteVeroeffentlichung") or "") >= since
                        and is_relevant(j.get("stellenangebotsTitel", ""), profile)):
                    found[j.get("referenznummer") or j.get("refnr")] = j
            if len(results) < 100:
                break

    def details(refnr):
        """The public job page embeds the full ad as JSON (Angular ng-state)."""
        try:
            page = requests.get(f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}",
                                headers=HTTP_HEADERS, timeout=30).text
            state = re.search(r'<script[^>]*id="ng-state"[^>]*>(.*?)</script>', page, re.S)
            return json.loads(state.group(1)).get("jobdetail", {}).get("stellenangebotsBeschreibung", "")
        except Exception:
            return ""

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        descriptions = dict(zip(found, pool.map(details, found)))

    rows = []
    for refnr, j in found.items():
        places = [loc.get("adresse", {}) for loc in j.get("stellenlokationen") or []]
        if places and places[0].get("land") not in (None, "DEUTSCHLAND"):
            continue
        rows.append(_row(source="Make it in Germany (BA)", region="Germany", country="Germany",
                         title=j.get("stellenangebotsTitel"), company=j.get("firma"),
                         location=", ".join(sorted({p.get("ort", "") for p in places}))[:80],
                         url=f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}",
                         description=descriptions.get(refnr, ""),
                         posted=j.get("datumErsteVeroeffentlichung") or j.get("aenderungsdatum")))
    print(f"   Make it in Germany / Bundesagentur: {len(rows)} relevant jobs")
    return rows


def collect_eures(profile, countries, hours=24):
    """EURES - the EU's official job mobility portal (all EU/EEA countries + CH, not UK)."""
    api = "https://europa.eu/eures/api/jv-searchengine/public/jv-search/search?lang=en"
    wanted = {"Germany"} | {c[0] for c in countries}
    period = "LAST_DAY" if hours <= 24 else "LAST_THREE_DAYS" if hours <= 72 else "LAST_WEEK"
    cutoff = time.time() - hours * 3600
    rows = []
    for phrase in search_phrases(profile):
        for page in range(1, 5):
            body = {"resultsPerPage": 50, "page": page, "sortSearch": "MOST_RECENT",
                    "keywords": [{"keyword": phrase, "specificSearchCode": "TITLE"}],
                    "publicationPeriod": period, "occupationUris": [], "skillUris": [],
                    "requiredExperienceCodes": [], "positionScheduleCodes": [], "sectorCodes": [],
                    "educationAndQualificationLevelCodes": [], "positionOfferingCodes": [],
                    "locationCodes": [], "euresFlagCodes": [], "otherBenefitsCodes": [],
                    "requiredLanguages": [], "minNumberPost": None}
            try:
                r = requests.post(api, json=body, headers=HTTP_HEADERS, timeout=60)
                r.raise_for_status()
            except Exception as e:
                print(f"   ! EURES / {phrase}: {e}")
                break
            jvs = r.json().get("jvs") or []
            for j in jvs:
                country = EURES_COUNTRIES.get(next(iter(j.get("locationMap") or {}), ""), "")
                if country not in wanted:
                    continue
                english = (j.get("translations") or {}).get("en") or {}
                title = english.get("title") or j.get("title", "")
                created = int(j.get("creationDate") or 0) / 1000
                if not is_relevant(title, profile) or (created and created < cutoff):
                    continue
                rows.append(_row(
                    source="EURES", region="Germany" if country == "Germany" else "Rest of Europe & UK",
                    country=country, title=title, company=(j.get("employer") or {}).get("name"),
                    location=country, url=f"https://europa.eu/eures/portal/jv-se/jv-details/{j['id']}?lang=en",
                    # original-language text, so the evaluator can see which language the ad is in
                    description=_strip_html(j.get("description", "")),
                    posted=dt.datetime.fromtimestamp(created, dt.timezone.utc).isoformat(timespec="minutes") if created else ""))
            if len(jvs) < 50:
                break
    print(f"   EURES: {len(rows)} relevant jobs")
    return rows


def collect_xing(profile, hours=24):
    """XING - the German-speaking LinkedIn (Germany, Austria, Switzerland)."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    found = {}
    for phrase in search_phrases(profile):
        try:
            page = requests.get("https://www.xing.com/jobs/search", headers=HTTP_HEADERS, timeout=30,
                                params={"keywords": phrase, "sort": "date"}).content.decode("utf-8", "ignore")
        except Exception as e:
            print(f"   ! XING / {phrase}: {e}")
            continue
        # Search results are server-rendered into an Apollo cache: "VisibleJob:<id>": {...}
        decoder = json.JSONDecoder()
        companies = {}
        for m in re.finditer(r'"(VisibleJob|Company):[^"]+":\{', page):
            try:
                obj, _ = decoder.raw_decode(page, m.end() - 1)
            except ValueError:
                continue
            if m.group(1) == "Company":
                companies[obj.get("id")] = obj.get("companyName", "")
                continue
            refreshed = obj.get("refreshedAt") or ""
            if not refreshed or dt.datetime.fromisoformat(refreshed.replace("Z", "+00:00")) < cutoff:
                continue
            if is_relevant(obj.get("title", ""), profile):
                found[obj["id"]] = obj
        time.sleep(2)

    def details(url):
        try:
            page = requests.get(url, headers=HTTP_HEADERS, timeout=30).content.decode("utf-8", "ignore")
            return _strip_html(page)[:8000]
        except Exception:
            return ""

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        descriptions = dict(zip(found, pool.map(details, [j["url"] for j in found.values()])))

    rows = []
    for jid, j in found.items():
        info = j.get("companyInfo") or {}
        company_ref = (info.get("company") or {}).get("__ref", "").split(":", 1)[-1]
        city = (j.get("location") or {}).get("city") or ""
        country = _guess_country(city)
        salary = j.get("salary") or {}
        rows.append(_row(
            source="XING", region="Germany" if country == "Germany" else "Rest of Europe & UK",
            country=country, title=j.get("title"),
            company=info.get("companyNameOverride") or companies.get(company_ref, ""),
            location=city, url=j.get("url"), description=descriptions.get(jid, ""),
            posted=j.get("refreshedAt"),
            # XING shows its own salary *estimate* when the ad has none - only trust EUR minimums
            salary_eur=salary.get("minimum") if salary.get("currency") == "EUR" else None))
    print(f"   XING: {len(rows)} relevant jobs")
    return rows


def collect_nextjobfinder(profile, countries, hours=24):
    """nextjobfinder.com - job aggregator; results show a publish date, not sorted by it."""
    since = dt.date.today() - dt.timedelta(days=_days(hours))
    rows = []
    for name, *_ in [GERMANY] + countries:
        for phrase in search_phrases(profile):
            for page in (1, 2):
                try:
                    r = requests.get("https://nextjobfinder.com/search-result.php", headers=HTTP_HEADERS,
                                     params={"kw": phrase, "lc": name, "ct": "", "cp": "", "p": page},
                                     timeout=30)
                    r.raise_for_status()
                except Exception as e:
                    print(f"   ! NextJobFinder {name} / {phrase}: {e}")
                    break
                items = r.text.split('<div class="item">')[1:]
                for item in items:
                    title = re.search(r"<h2>(.*?)</h2>", item, re.S)
                    link = re.search(r'<div class="apply">\s*<a href="([^"]+)"', item)
                    date = re.search(r"Published Date:</span>\s*([^<]+)", item)
                    if not (title and link and date):
                        continue
                    try:
                        posted = dt.datetime.strptime(date.group(1).strip(), "%d %b %Y").date()
                    except ValueError:
                        continue
                    title = _strip_html(title.group(1))
                    if posted < since or not is_relevant(title, profile):
                        continue
                    company = re.search(r"Company:</span>\s*([^<]+)", item)
                    location = re.search(r"Location:</span>\s*([^<]+)", item)
                    snippet = re.search(r"</h2>\s*<p>(.*?)</p>", item, re.S)
                    rows.append(_row(
                        source="NextJobFinder", region="Germany" if name == "Germany" else "Rest of Europe & UK",
                        country=name, title=title, company=company.group(1).strip() if company else "",
                        location=location.group(1).strip() if location else name, url=link.group(1),
                        description=_strip_html(snippet.group(1)) if snippet else "",
                        posted=posted.isoformat()))
                if len(items) < 10:
                    break
                time.sleep(1)
    print(f"   NextJobFinder: {len(rows)} relevant jobs")
    return rows


def collect_glassdoor(profile, countries, region, results_per_query, hours=24):
    """Glassdoor via python-jobspy. Called separately from LinkedIn/Indeed because Glassdoor
    doesn't support every country and one unsupported country must not break the others."""
    from jobspy import scrape_jobs

    rows, skipped = [], 0
    for name, linkedin_loc, indeed_country in countries:
        for term in profile["search_terms"]:
            try:
                df = scrape_jobs(site_name=["glassdoor"], search_term=term, location=linkedin_loc,
                                 country_indeed=indeed_country, hours_old=hours,
                                 results_wanted=results_per_query, description_format="markdown", verbose=0)
            except Exception as e:
                print(f"   ! Glassdoor {name}: not available ({str(e)[:80]})")
                break  # usually "country not supported" - skip the remaining terms too
            for _, j in df.iterrows():
                if not is_relevant(j.get("title"), profile):  # Glassdoor's search is very loose
                    skipped += 1
                    continue
                rows.append(_row(source="Glassdoor", region=region, country=name, title=j.get("title"),
                                 company=j.get("company"), location=j.get("location"),
                                 url=j.get("job_url_direct") or j.get("job_url"),
                                 description=j.get("description") if pd.notna(j.get("description")) else "",
                                 posted=str(j.get("date_posted", ""))))
            time.sleep(2)
    print(f"   Glassdoor: {len(rows)} relevant jobs ({skipped} off-target titles skipped)")
    return rows


def _ats_country(location: str):
    """Country for a careers-page location string, 'Remote (Europe)', or None if outside Europe."""
    loc = (location or "").lower()
    words = set(re.split(r"[^a-zà-ÿ]+", loc))
    if any(p in loc for p in GERMAN_PLACES):
        return "Germany"
    for country, places in COUNTRY_PLACES.items():
        if any((p in words) if len(p) <= 3 else (p in loc) for p in places):
            return country
    if re.search(r"\b(europe|emea|eu)\b", loc):
        return "Remote (Europe)"
    return None


def _fetch_career_page(company, cutoff):
    """Recent jobs from one company's applicant-tracking system. Returns
    [(title, location, url, posted_iso, description_or_fetcher)]."""
    ats, slug = company["ats"], company["slug"]
    out = []
    if ats == "greenhouse":
        jobs = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
                            headers=HTTP_HEADERS, timeout=30).json().get("jobs", [])
        for j in jobs:
            posted = j.get("first_published") or j.get("updated_at") or ""
            url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs/{j['id']}"
            out.append((j["title"], (j.get("location") or {}).get("name", ""), j.get("absolute_url"), posted,
                        lambda u=url: html.unescape(requests.get(u, headers=HTTP_HEADERS, timeout=30).json().get("content", ""))))
    elif ats == "lever":
        for j in requests.get(f"https://api.lever.co/v0/postings/{slug}?mode=json",
                              headers=HTTP_HEADERS, timeout=30).json():
            posted = dt.datetime.fromtimestamp(j["createdAt"] / 1000, dt.timezone.utc).isoformat() if j.get("createdAt") else ""
            text = j.get("descriptionPlain", "") + "\n" + "\n".join(
                f"{l.get('text', '')}: {_strip_html(l.get('content', ''))}" for l in j.get("lists") or [])
            out.append((j["text"], (j.get("categories") or {}).get("location", ""), j.get("hostedUrl"), posted, text))
    elif ats == "ashby":
        for j in requests.get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}",
                              headers=HTTP_HEADERS, timeout=30).json().get("jobs", []):
            locations = [j.get("location") or ""] + [l.get("location", "") for l in j.get("secondaryLocations") or []]
            out.append((j["title"], " / ".join(filter(None, locations)), j.get("jobUrl"), j.get("publishedAt") or "",
                        j.get("descriptionPlain", "")))
    elif ats == "smartrecruiters":
        api = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
        for offset in range(0, 1000, 100):
            page = requests.get(api, params={"limit": 100, "offset": offset}, headers=HTTP_HEADERS, timeout=30).json()
            for j in page.get("content", []):
                loc = j.get("location") or {}
                url = f"{api}/{j['id']}"
                out.append((j["name"], ", ".join(filter(None, [loc.get("city"), loc.get("country")])),
                            f"https://jobs.smartrecruiters.com/{slug}/{j['id']}", j.get("releasedDate") or "",
                            lambda u=url: " ".join(_strip_html(s.get("text", "")) for s in
                                                   ((requests.get(u, headers=HTTP_HEADERS, timeout=30).json()
                                                     .get("jobAd") or {}).get("sections") or {}).values())))
            if len(page.get("content", [])) < 100:
                break
    recent = []
    for title, location, url, posted, desc in out:
        ts = pd.to_datetime(posted, utc=True, errors="coerce")
        if pd.notna(ts) and ts >= cutoff:
            recent.append((title, location, url, ts.isoformat(timespec="minutes"), desc))
    return recent


def collect_career_pages(profile, countries, hours=24):
    """Career pages of the companies in companies.json (Greenhouse, Lever, Ashby, SmartRecruiters).
    Many startups and scale-ups post here first - or only here."""
    companies = json.loads(COMPANIES_PATH.read_text())["companies"] if COMPANIES_PATH.exists() else []
    wanted = {"Germany", "Remote (Europe)"} | {c[0] for c in countries}
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)

    def fetch(company):
        try:
            return company, _fetch_career_page(company, cutoff), None
        except Exception as e:
            return company, [], e

    candidates, failed = [], []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for company, jobs, error in pool.map(fetch, companies):
            if error:
                failed.append(company["name"])
            for title, location, url, posted, desc in jobs:
                country = _ats_country(location)
                if country in wanted and is_relevant(title, profile):
                    candidates.append((company["name"], title, location, url, posted, desc, country))

    def describe(desc):
        try:
            return _strip_html(desc() if callable(desc) else desc)
        except Exception:
            return ""

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        descriptions = list(pool.map(describe, [c[5] for c in candidates]))
    rows = [_row(source="Career page", region="Germany" if country == "Germany" else "Rest of Europe & UK",
                 country=country, title=title, company=name, location=location, url=url,
                 description=desc, posted=posted)
            for (name, title, location, url, posted, _, country), desc in zip(candidates, descriptions)]
    note = f" ({len(failed)} unreachable: {', '.join(failed[:5])})" if failed else ""
    print(f"   Career pages: {len(rows)} relevant jobs from {len(companies)} companies{note}")
    return rows


def _strip_html(text):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text or "", flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def _guess_country(location):
    words = set(re.split(r"[^a-zà-ÿ]+", location.lower()))
    loc = location.lower()
    if not loc or any(p in loc for p in GERMAN_PLACES):
        return "Germany"  # Arbeitnow is a German board; unlabelled/remote jobs are mostly German
    for country, places in COUNTRY_PLACES.items():
        if any((p in words) if len(p) <= 3 else (p in loc) for p in places):
            return country
    return location


_SENIORITY = re.compile(r"\b(senior|sr\.?|lead|principal|staff|head of|junior|jr\.?|mid|level)\b")


def _core_phrase(title: str) -> str:
    """'Senior Data Engineer (Python)' -> 'data engineer'."""
    t = _SENIORITY.sub(" ", re.sub(r"\(.*?\)", "", title.lower()))
    return re.sub(r"\s+", " ", t).strip(" -/,")


def search_phrases(profile):
    """What the smaller boards are queried with: your titles without seniority words."""
    phrases = dict.fromkeys(_core_phrase(t) for t in profile["search_terms"])
    return [p for p in phrases if " " in p or len(p) > 6]  # skip over-broad single words like "data"


def match_phrases(profile):
    """What a job title must contain to be kept: search phrases + LLM-generated title variants
    ('llm engineer', 'applied scientist', 'ki engineer', ...)."""
    phrases = (set(search_phrases(profile))
               | {_core_phrase(t) for t in profile.get("previous_search_terms") or []}  # titles merged away
               | {v.lower().strip() for v in profile.get("title_variants") or []})
    return sorted(p for p in phrases if " " in p or len(p) > 6)


@functools.lru_cache(maxsize=8)
def _phrase_regex(phrases: tuple):
    alternatives = "|".join(re.escape(p).replace(r"\ ", r"[\s\-/]+") for p in phrases)
    return re.compile(r"(?<![a-z0-9])(?:" + alternatives + r")(?![a-z0-9])")


_TOO_JUNIOR = re.compile(r"\b(junior|jr\.?|intern|internship|werkstudent\w*|working student|praktik\w*|"
                         r"trainee|graduate|thesis|abschlussarbeit|masterarbeit|bachelorarbeit|ausbildung)\b", re.I)


def is_relevant(title: str, profile) -> bool:
    title = (title or "").lower()
    return not _TOO_JUNIOR.search(title) and bool(_phrase_regex(tuple(match_phrases(profile))).search(title))


VARIANTS_PROMPT = """You help a job seeker search European job boards.
Given their target job titles and profile, list 20 to 30 alternative job-title phrases that European
employers use in job ads for the SAME kind of role this candidate is qualified for.
Rules:
- lowercase, 2-4 words, WITHOUT seniority words (senior, lead, staff, principal, junior)
- include common English variants and abbreviations (e.g. "ml engineer", "llm engineer", "applied scientist")
- include titles for specialisations visible in the profile (e.g. "nlp engineer", "ai research engineer",
  "computer vision engineer") and hybrid titles (e.g. "software engineer ai", "ai platform engineer")
- include a few common German variants (e.g. "ki engineer", "ki-entwickler")
- do NOT include roles the candidate is not qualified for, and nothing generic like "engineer" or "developer"
Return JSON: {"variants": [...]}"""


def ensure_title_variants(profile, client=None):
    """Generate title variants once per set of search terms (cached in profile.json)."""
    if profile.get("title_variants") and profile.get("title_variants_for") == profile["search_terms"]:
        return profile
    print("   Generating job-title variants for broader matching ...")
    about = {k: profile.get(k) for k in ("summary", "seniority", "core_skills", "domain")}
    resp = (client or OpenAI()).chat.completions.create(
        model=MODEL, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": VARIANTS_PROMPT},
                  {"role": "user", "content": json.dumps({"target_titles": profile["search_terms"], **about})}])
    variants = json.loads(resp.choices[0].message.content).get("variants", [])
    profile["title_variants"] = sorted({_core_phrase(v) for v in variants if isinstance(v, str)} - {""})
    profile["title_variants_for"] = profile["search_terms"]
    save_profile(profile)
    return profile


def _days(hours: int) -> int:
    return max(1, -(-hours // 24))  # ceil


def _ascii(text: str) -> str:
    text = (text or "").lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _norm_company(name: str) -> str:
    """'Siemens Energy Global GmbH & Co. KG' and 'Siemens Energy' -> 'siemens energy'."""
    name = _ascii(name)
    name = re.sub(r"\bt/a\b.*|\bstandort\b.*|\.com\b", "", name)
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\b(ltd|limited|plc|llp|llc|inc|gmbh|mbh|ag|kg|kgaa|vvag|se|bv|b v|nv|n v|uk|group|holdings?|"
                  r"europe|international|global|germany|deutschland|dienstleistungen|investors|"
                  r"co|company|the)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


_GENDER_TAGS = re.compile(r"\((?:[mwfdxhi]\s*[/|,]\s*)+[mwfdxhi]\)|\(gn\)|\b(?:all genders?|alle geschlechter|"
                          r"m/w/d|f/m/d|m/f/d|w/m/d|f/m/x|m/f/x|w/m/x|d/f/m)\b", re.I)
_PLACES = sorted({p for places in COUNTRY_PLACES.values() for p in places if len(p) > 3}
                 | set(GERMAN_PLACES) | {"remote", "hybrid", "europe", "emea", "eu"}, key=len, reverse=True)
_PLACES_RE = re.compile(r"\b(?:in\s+|-\s*)?(?:" + "|".join(map(re.escape, _PLACES)) + r")\b", re.I)


def _norm_title(title: str) -> str:
    """'Data Engineer (m/w/d) in Hamburg' and 'Data Engineer (all genders) Berlin' -> 'data engineer'.
    German ads put extras after the gender tag ('(m/w/d), Würzburg', '(m/w/d) (Data Scientist)'),
    so everything after it is dropped, as are reference codes like 'E362/B - '."""
    t = (title or "").lower()
    tag = _GENDER_TAGS.search(t)
    if tag and tag.start() > 0:
        t = t[:tag.start()]
    t = re.sub(r"^\s*[a-z]{0,3}\d+[a-z0-9/._-]*\s*[-–:|]\s*", "", t)
    t = _PLACES_RE.sub(" ", _GENDER_TAGS.sub(" ", t))
    t = re.sub(r"[^a-z0-9äöüß+# ]", " ", t)
    t = re.sub(r"\b(senior|sr)\b", " ", t)  # 'Data Scientist' and 'Senior Data Scientist' at one company: apply once
    return re.sub(r"\s+", " ", t).strip()


def merge_duplicates(rows, score=None):
    """One row per (company, normalised title). The same ad is often listed in several countries
    or cities, or on several boards. Keeps the best row (highest `score`, else the Germany one,
    else the employer's own career page, else the most detailed) and records the other countries in `also_in`."""
    groups = {}
    for r in rows:
        if not r.get("title"):
            continue
        groups.setdefault((_norm_company(r["company"]), _norm_title(r["title"])), []).append(r)
    out = []
    for group in groups.values():
        best = max(group, key=score or (lambda r: (r["country"] == "Germany", r["source"] == "Career page",
                                                   len(r.get("description") or ""))))
        others = {r["country"] for r in group if r["country"]} | {c for r in group for c in r.get("also_in") or []}
        others = sorted(others - {best["country"]})
        best = {**best, "also_in": others}
        out.append(best)
    return out


def deduplicate(rows):
    return merge_duplicates(rows)


# --------------------------------------------------------------------------------------
# 3. Official visa-sponsor registers
# --------------------------------------------------------------------------------------

def _seen_key(job) -> str:
    return f'{_norm_company(job["company"])}|{_norm_title(job["title"])}'


def mark_new(jobs):
    """Flags jobs never shown in any earlier run. data/seen_jobs.json maps job key -> first seen.
    Earlier runs' results are always included, so old jobs aren't flagged as new."""
    seen = json.loads(SEEN_PATH.read_text()) if SEEN_PATH.exists() else {}
    # Also re-key earlier runs' jobs every time, so improving _seen_key never makes old jobs look new
    for path in sorted(OUTPUT_DIR.glob("*/results.json")):
        data = json.loads(path.read_text())
        for j in data["jobs"]:
            seen.setdefault(_seen_key(j), data["meta"].get("created", ""))
    now = dt.datetime.now().isoformat(timespec="minutes")
    for job in jobs:
        key = _seen_key(job)
        job["is_new"] = key not in seen
        job["first_seen"] = seen.setdefault(key, now)
    SEEN_PATH.write_text(json.dumps(seen, ensure_ascii=False))
    print(f"   {sum(j['is_new'] for j in jobs)} jobs are new since earlier runs")
    return jobs


def _cached_download(name, fetch, max_age_days=7):
    path = DATA_DIR / name
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return set(json.loads(path.read_text()))
    try:
        names = fetch()
        path.write_text(json.dumps(sorted(names)))
        return names
    except Exception as e:
        print(f"   ! could not download {name}: {e}")
        return set(json.loads(path.read_text())) if path.exists() else set()


def _fetch_uk_sponsors():
    page = requests.get("https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers",
                        headers=HTTP_HEADERS, timeout=30).text
    csv_url = re.search(r"https://assets\.publishing\.service\.gov\.uk/[^\"]+\.csv", page).group(0)
    text = requests.get(csv_url, headers=HTTP_HEADERS, timeout=120).content.decode("utf-8", "ignore")
    return {_norm_company(r["Organisation Name"]) for r in csv.DictReader(io.StringIO(text))
            if "Skilled Worker" in r.get("Route", "") or "Global Business Mobility" in r.get("Route", "")}


def _fetch_nl_sponsors():
    page = requests.get("https://ind.nl/en/public-register-recognised-sponsors/"
                        "public-register-regular-labour-and-highly-skilled-migrants",
                        headers=HTTP_HEADERS, timeout=60).text
    cells = re.findall(r"<tr[^>]*>\s*<t[dh][^>]*>(.*?)</t[dh]>", page, re.S)
    return {_norm_company(html.unescape(re.sub(r"<[^>]+>", "", c)).strip('" ')) for c in cells[1:]}


def load_sponsor_registers():
    DATA_DIR.mkdir(exist_ok=True)
    uk = _cached_download("uk_sponsors.json", _fetch_uk_sponsors)
    nl = _cached_download("nl_sponsors.json", _fetch_nl_sponsors)
    print(f"   UK licensed sponsors: {len(uk):,}   NL IND recognised sponsors: {len(nl):,}")
    return {"United Kingdom": uk, "Netherlands": nl}


# --------------------------------------------------------------------------------------
# 4. LLM evaluation of each posting
# --------------------------------------------------------------------------------------

EVAL_PROMPT = """You evaluate European job postings for a specific candidate from India who needs a
work visa / EU Blue Card and does NOT speak German (or the local language).

CANDIDATE PROFILE:
{profile}

Read the job posting and return JSON with exactly these keys:
- "fit_score": 0-100, how well the candidate's CV matches this role (skills, seniority, domain).
   Be strict: wrong function or wrong tech stack <= 30; strong match >= 75.
- "posting_language": "english" | "german" | "other"
- "local_language_required": "required" | "nice_to_have" | "not_mentioned" | "explicitly_not_required"
   (German, Dutch, French, etc. - any language other than English)
- "english_working_language": true if the ad says the team/company works in English or is international
- "visa_sponsorship": "offered" | "explicitly_not_offered" | "not_mentioned"
   ("must already have the right to work in X" / "no sponsorship" = explicitly_not_offered)
- "relocation_support": true/false
- "remote_restricted_to_residents": true if remote but only for people already living in the country/EU
- "company_type": "startup" | "scaleup" | "big_tech_international" | "corporate" | "consultancy" | "recruitment_agency" | "unknown"
   (startup = early-stage/seed/Series A; scaleup = Series B+ / fast-growing tech, ~100-2000 people)
- "salary_eur_per_year": number or null (only if stated, converted to EUR gross per year)
- "reason": one short sentence explaining the fit and the main hiring-chance factor
"""


def _load_eval_cache():
    return json.loads(EVAL_CACHE_PATH.read_text()) if EVAL_CACHE_PATH.exists() else {}


def evaluate_jobs(jobs, profile, client):
    cache = _load_eval_cache()
    profile_text = json.dumps({k: v for k, v in profile.items() if k != "cv_text"}, ensure_ascii=False)
    profile_text += "\n\nCV EXCERPT:\n" + profile.get("cv_text", "")[:4000]
    system = EVAL_PROMPT.format(profile=profile_text)

    def evaluate(job):
        key = job["url"] or f'{job["company"]}|{job["title"]}'
        if key in cache:
            return key, cache[key]
        posting = (f"TITLE: {job['title']}\nCOMPANY: {job['company']}\nLOCATION: {job['location']}\n"
                   f"COUNTRY: {job['country']}\nCOMPANY SIZE: {job['company_size']}\n\n"
                   f"{job['description'][:6000]}")
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": posting}])
                return key, json.loads(resp.choices[0].message.content)
            except Exception as e:
                if attempt == 2:
                    print(f"   ! evaluation failed for {job['title']} @ {job['company']}: {e}")
                    return key, None
                time.sleep(3 * (attempt + 1))

    results = {}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i, (job, (key, result)) in enumerate(zip(jobs, pool.map(evaluate, jobs)), 1):
            results[key] = result
            if result and job["description"]:  # title-only evaluations are retried next run
                cache[key] = result
            if i % 25 == 0:
                print(f"   evaluated {i}/{len(jobs)}")

    EVAL_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False))
    for job in jobs:
        job["eval"] = results.get(job["url"] or f'{job["company"]}|{job["title"]}')
    return [j for j in jobs if j["eval"]]


# --------------------------------------------------------------------------------------
# 5. Ranking - "how likely is this employer to hire a non-German-speaking Indian candidate"
# --------------------------------------------------------------------------------------

# Startups and scale-ups get the biggest boost. In this agent's own results, traditional corporates
# required the local language about twice as often (44% vs 20%) and almost never mentioned visa
# support, and Bitkom finds most German tech startups work in English. Big tech does sponsor, but every
# applicant in the world applies there, so it only gets a small boost. In Germany a Blue Card needs no
# sponsor licence, so a small startup can hire from India as easily as a corporate; in the UK/NL the
# sponsor-register check below catches startups that can't sponsor.
COMPANY_POINTS = {"startup": 18, "scaleup": 18, "big_tech_international": 5,
                  "corporate": -10, "consultancy": -5, "recruitment_agency": -8}
_COMPANY_LABELS = {"startup", "scaleup", "big tech international", "corporate", "consultancy", "recruitment agency"}


def rescore_saved(row):
    """Re-applies the current COMPANY_POINTS to a job saved by an earlier run, using its signals list."""
    signals = [s for s in row.get("signals") or [] if s.split(" ", 1)[-1] not in _COMPANY_LABELS]
    ctype = row.get("company_type")
    if COMPANY_POINTS.get(ctype):
        points = COMPANY_POINTS[ctype]
        signals.append(f"{'+' if points > 0 else ''}{points} {ctype.replace('_', ' ')}")
    chance = max(0, min(100, 50 + sum(int(s.split(" ", 1)[0]) for s in signals)))
    return {**row, "signals": signals, "hiring_chance": chance,
            "rank_score": round(0.55 * chance + 0.45 * (row.get("cv_fit") or 0), 1)}


def international_score(job, sponsors):
    """Rule-based, explainable score 0-100. Returns (score, list of signals)."""
    e, score, signals = job["eval"], 50, []

    def add(points, label):
        nonlocal score
        score += points
        signals.append(f"{'+' if points > 0 else ''}{points} {label}")

    # Language: the biggest barrier in Germany / continental Europe
    if e.get("posting_language") == "english":
        add(10, "ad in English")
    elif e.get("posting_language") == "german":
        add(-25, "ad in German")
    lang = e.get("local_language_required")
    if lang == "required":
        add(-35, "local language required")
    elif lang == "nice_to_have":
        add(-5, "local language a plus")
    elif lang == "explicitly_not_required":
        add(12, "no local language needed")
    if e.get("english_working_language"):
        add(10, "English working language")

    # Visa / relocation statements in the ad
    visa = e.get("visa_sponsorship")
    if visa == "offered":
        add(25, "visa sponsorship offered")
    elif visa == "explicitly_not_offered":
        add(-50, "NO visa sponsorship")
    if e.get("relocation_support"):
        add(12, "relocation support")
    if e.get("remote_restricted_to_residents"):
        add(-20, "remote for residents only")

    # Company type - see COMPANY_POINTS
    ctype = e.get("company_type")
    if COMPANY_POINTS.get(ctype):
        add(COMPANY_POINTS[ctype], ctype.replace("_", " "))

    # Country-specific legal reality
    country = job["country"]
    register = sponsors.get(country)
    if register is not None and ctype != "recruitment_agency":
        if _norm_company(job["company"]) in register:
            add(20, f"on official {country} sponsor register")
        elif visa != "offered":
            # UK Skilled Worker and NL highly-skilled-migrant visas legally need a registered sponsor
            add(-25, f"NOT on {country} sponsor register")
    if job["region"] == "Germany":
        salary = e.get("salary_eur_per_year") or job.get("salary_eur")
        if salary and salary >= BLUE_CARD_MIN_EUR:
            add(5, "salary meets Blue Card")
        elif salary and salary < BLUE_CARD_MIN_EUR:
            add(-10, "salary below Blue Card")

    return max(0, min(100, score)), signals


def rank(jobs, sponsors):
    for job in jobs:
        job["fit"] = int(job["eval"].get("fit_score") or 0)
        job["intl"], job["signals"] = international_score(job, sponsors)
        # Hiring chance first, CV fit second - both matter
        job["rank_score"] = round(0.55 * job["intl"] + 0.45 * job["fit"], 1)
    return sorted(jobs, key=lambda j: j["rank_score"], reverse=True)


# --------------------------------------------------------------------------------------
# 6. Report
# --------------------------------------------------------------------------------------

def _posted_label(raw):
    """Sources report posting time in different formats, some only as a date.
    Returns a short human label ('5h ago', 'Yesterday', '8 Oct') and a sortable ISO value."""
    raw = str(raw or "").strip()
    if not raw or raw.lower() in ("nan", "none", "nat"):
        return "–", ""
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return html.escape(raw[:16]), raw
    date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw))
    now = pd.Timestamp.now(tz="UTC")
    days = (now.normalize() - ts.normalize()).days
    if not date_only and now - ts < pd.Timedelta(hours=24):
        hours = max(int((now - ts).total_seconds() // 3600), 0)
        label = "< 1h ago" if hours == 0 else f"{hours}h ago"
    elif days <= 0:
        label = "Today"
    elif days == 1:
        label = "Yesterday"
    else:
        label = f"{ts.day} {ts.strftime('%b')} ({days} days)"
    return label, ts.date().isoformat() if date_only else ts.isoformat(timespec="minutes")


def _table(jobs):
    if not jobs:
        return "<p class='empty'>No matching jobs in this window.</p>"
    rows = []
    for i, j in enumerate(jobs, 1):
        e = j["eval"]
        tier = "high" if j["intl"] >= 70 else "mid" if j["intl"] >= 45 else "low"
        signals = "".join(f"<span class='sig {'pos' if s.startswith('+') else 'neg'}'>{html.escape(s)}</span>"
                          for s in j["signals"])
        rows.append(f"""<tr>
<td class='num'>{i}</td>
<td>{"<span class='new'>NEW</span> " if j.get('is_new') else ""}<a href='{html.escape(j['url'])}' target='_blank' rel='noopener'>{html.escape(j['title'])}</a>
    <div class='muted'>{html.escape(j['company'])} · {html.escape(j['location'])} · {j['source']}{
        (' · also in ' + html.escape(', '.join(j['also_in']))) if j.get('also_in') else ''}</div>
    <div class='reason'>{html.escape(e.get('reason', ''))}</div></td>
<td class='num'><span class='pill {tier}'>{j['intl']}</span></td>
<td class='num'>{j['fit']}</td>
<td class='num'><b>{j['rank_score']}</b></td>
<td class='posted' title='{html.escape(str(j['posted']))}'>{_posted_label(j['posted'])[0]}</td>
<td class='signals'>{signals}</td></tr>""")
    return ("<table><thead><tr><th>#</th><th>Role</th><th>Hiring chance</th><th>CV fit</th>"
            "<th>Rank</th><th>Posted</th><th>Why</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def write_report(germany, rest, profile, out_dir, hours=48):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, jobs in (("germany", germany), ("rest_of_europe_uk", rest)):
        pd.DataFrame([{
            "rank_score": j["rank_score"], "hiring_chance": j["intl"], "cv_fit": j["fit"],
            "title": j["title"], "company": j["company"], "country": j["country"],
            "location": j["location"], "source": j["source"], "posted": _posted_label(j["posted"])[1],
            "new": "NEW" if j.get("is_new") else "",
            "company_type": j["eval"].get("company_type"),
            "visa_sponsorship": j["eval"].get("visa_sponsorship"),
            "local_language": j["eval"].get("local_language_required"),
            "signals": "; ".join(j["signals"]), "reason": j["eval"].get("reason"), "url": j["url"],
        } for j in jobs]).to_csv(out_dir / f"{name}.csv", index=False)

    today = dt.date.today().isoformat()
    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Europe Jobs {today}</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--card:#fff;--pos:#15803d;--neg:#b91c1c;--acc:#1d4ed8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--line:#44403c;--card:#292524;--pos:#4ade80;--neg:#f87171;--acc:#93c5fd}}}}
body{{font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:24px 16px;max-width:1200px;margin:auto}}
h1{{margin:0 0 4px}} h2{{margin-top:36px}} a{{color:var(--acc);font-weight:600;text-decoration:none}}
.muted{{color:var(--muted);font-size:12px}} .reason{{font-size:12px;margin-top:4px}}
.tabs button{{font:inherit;padding:8px 14px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;cursor:pointer;margin-right:6px}}
.tabs button.on{{border-color:var(--acc);color:var(--acc);font-weight:600}}
table{{width:100%;border-collapse:collapse;background:var(--card);margin-top:12px}}
th,td{{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-size:12px;color:var(--muted);font-weight:600}} .num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.pill{{padding:2px 8px;border-radius:99px;font-weight:700}} .pill.high{{color:var(--pos)}} .pill.mid{{color:#b45309}} .pill.low{{color:var(--neg)}}
.sig{{display:inline-block;font-size:11px;margin:0 4px 4px 0;padding:1px 6px;border-radius:4px;border:1px solid var(--line)}}
.sig.pos{{color:var(--pos)}} .sig.neg{{color:var(--neg)}} .signals{{max-width:320px}}
.posted{{white-space:nowrap;font-size:12px;color:var(--muted)}}
.new{{font-size:10px;font-weight:700;color:#fff;background:var(--acc);border-radius:4px;padding:1px 5px;vertical-align:2px}}
@media (max-width:760px){{.signals,th:nth-child(7){{display:none}}}}
</style></head><body>
<h1>Europe jobs posted in the last {hours}h</h1>
<div class='muted'>{today} · {html.escape(profile.get('name', ''))} · searched: {html.escape(', '.join(profile['search_terms']))}
 · showing CV fit ≥ {MIN_FIT}, ranked by 55% hiring chance + 45% CV fit</div>
<p class='tabs'><button class='on' onclick="show('de',this)">Germany ({len(germany)})</button><button onclick="show('eu',this)">Rest of Europe &amp; UK ({len(rest)})</button></p>
<section id='de'>{_table(germany)}</section>
<section id='eu' hidden>{_table(rest)}</section>
<script>function show(id,b){{document.querySelectorAll('section').forEach(s=>s.hidden=s.id!==id);
document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b))}}</script>
</body></html>"""
    path = out_dir / "jobs.html"
    path.write_text(page, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------

def load_profile(cv_path: Path | None = None, client: OpenAI | None = None) -> dict | None:
    """Saved profile, or a new one built from cv_path. Passing the same CV again keeps the
    saved profile (and any edits you made to its search_terms)."""
    saved = json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else None
    if cv_path is None:
        return saved
    cv_path = Path(cv_path).expanduser()
    if saved and saved.get("cv_sha256") == _file_hash(cv_path):
        print("   Same CV as before - keeping saved profile.json and your edits to it")
        return saved
    print(f"   Reading CV {cv_path.name} ...")
    return build_profile(cv_path, client or OpenAI())


def save_profile(profile: dict):
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False))


DEFAULT_SETTINGS = {"countries": ["United Kingdom", "Netherlands", "Ireland", "Switzerland", "Austria"],
                    "sources": SOURCES, "results_per_query": 25, "hours": 48}


def load_settings() -> dict:
    """Search settings shared by the UI and scheduled runs (settings.json)."""
    saved = json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
    settings = {**DEFAULT_SETTINGS, **saved}
    settings["sources"] = [s for s in settings["sources"] if s in SOURCES]
    return settings


def save_settings(**settings):
    SETTINGS_PATH.write_text(json.dumps({**load_settings(), **settings}, indent=2))


def run_search(profile, sources=SOURCES, countries=None, results_per_query=25, hours=48, client=None):
    """Full pipeline: collect -> sponsor registers -> LLM evaluation -> rank -> save.
    `hours` is the posting window; 48h overlaps consecutive daily runs so nothing slips through,
    and jobs already seen in earlier runs are not flagged NEW. Returns (all ranked jobs, output folder)."""
    client = client or OpenAI()
    DATA_DIR.mkdir(exist_ok=True)
    rest = REST_OF_EUROPE
    if countries:
        wanted = {c.lower() for c in countries}
        rest = [c for c in REST_OF_EUROPE if c[0].lower() in wanted]
    print(f"Target roles: {', '.join(profile['search_terms'])}")
    profile = ensure_title_variants(profile, client)
    print(f"   Matching titles containing any of {len(match_phrases(profile))} phrases")

    print(f"1. Collecting jobs posted in the last {hours} hours ...")
    terms = profile["search_terms"]
    collectors = {
        "linkedin-indeed": lambda: collect_jobspy(terms, [GERMANY], "Germany", results_per_query, hours)
                                 + collect_jobspy(terms, rest, "Rest of Europe & UK", results_per_query, hours),
        "glassdoor": lambda: collect_glassdoor(profile, [GERMANY], "Germany", results_per_query, hours)
                           + collect_glassdoor(profile, rest, "Rest of Europe & UK", results_per_query, hours),
        "careers": lambda: collect_career_pages(profile, rest, hours),
        "arbeitnow": lambda: collect_arbeitnow(profile, hours),
        "berlinstartupjobs": lambda: collect_berlin_startup_jobs(profile, hours),
        "bundesagentur": lambda: collect_bundesagentur(profile, hours),
        "eures": lambda: collect_eures(profile, rest, hours),
        "xing": lambda: collect_xing(profile, hours),
        "nextjobfinder": lambda: collect_nextjobfinder(profile, rest, hours),
        "adzuna": lambda: collect_adzuna(terms, hours),
    }
    jobs = []
    for name in sources:
        try:
            jobs += collectors[name]()
        except Exception as e:  # a site changing its layout must not kill the whole run
            print(f"   ! {name} failed: {e}")
    jobs = fill_linkedin_descriptions(deduplicate(jobs))
    print(f"   {len(jobs)} unique jobs")

    print("2. Loading official visa-sponsor registers ...")
    sponsors = load_sponsor_registers()

    print(f"3. Evaluating postings with {MODEL} ...")
    jobs = evaluate_jobs(jobs, profile, client)

    print("4. Ranking ...")
    ranked = mark_new(rank(jobs, sponsors))

    out_dir = OUTPUT_DIR / dt.datetime.now().strftime("%Y-%m-%d_%H%M")  # one folder per run, never overwritten
    save_results(ranked, profile, out_dir, hours)
    shown = [j for j in ranked if j["fit"] >= MIN_FIT]
    germany = [j for j in shown if j["region"] == "Germany"]
    rest_jobs = [j for j in shown if j["region"] != "Germany"]
    write_report(germany, rest_jobs, profile, out_dir, hours)
    print(f"5. Done: {len(germany)} Germany + {len(rest_jobs)} Rest of Europe & UK jobs "
          f"(CV fit >= {MIN_FIT}, {sum(j['is_new'] for j in shown)} new) -> {out_dir}")
    return ranked, out_dir


def save_results(ranked, profile, out_dir, hours=48):
    """All evaluated jobs (any CV fit) as JSON - the Streamlit app filters these itself."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{
        "region": j["region"], "country": j["country"], "title": j["title"], "company": j["company"],
        "location": j["location"], "also_in": j.get("also_in", []), "source": j["source"], "url": j["url"],
        "posted": _posted_label(j["posted"])[1], "is_new": j.get("is_new", False),
        "first_seen": j.get("first_seen", ""), "rank_score": j["rank_score"],
        "hiring_chance": j["intl"], "cv_fit": j["fit"], "signals": j["signals"],
        "company_type": j["eval"].get("company_type"),
        "visa_sponsorship": j["eval"].get("visa_sponsorship"),
        "local_language": j["eval"].get("local_language_required"),
        "relocation_support": j["eval"].get("relocation_support"),
        "reason": j["eval"].get("reason"),
    } for j in ranked]
    meta = {"created": dt.datetime.now().isoformat(timespec="minutes"), "hours": hours,
            "search_terms": profile["search_terms"], "model": MODEL}
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "jobs": rows}, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------------------
# 7. Daily schedule (macOS launchd; prints a cron line elsewhere)
# --------------------------------------------------------------------------------------

SCHEDULE_LABEL = "com.europe-job-search-agent.daily"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{SCHEDULE_LABEL}.plist"
SCHEDULE_LOG = DATA_DIR / "daily_run.log"


def schedule_status():
    """'HH:MM' if a daily run is installed, else None."""
    if not PLIST_PATH.exists():
        return None
    import plistlib
    when = plistlib.loads(PLIST_PATH.read_bytes()).get("StartCalendarInterval", {})
    return f"{when.get('Hour', 0):02d}:{when.get('Minute', 0):02d}"


def install_schedule(hour: int, minute: int = 0) -> str:
    """Run the agent every day at hour:minute with the saved settings. On macOS this uses launchd,
    which also runs a missed job as soon as the Mac wakes up."""
    import platform
    import plistlib
    import subprocess
    import sys

    command = [sys.executable, str(HERE / "job_agent.py"), "--no-open"]
    if platform.system() != "Darwin":
        line = f"{minute} {hour} * * * cd {HERE} && {' '.join(command)} >> {SCHEDULE_LOG} 2>&1"
        return f"Not macOS - add this line with `crontab -e`:\n{line}"
    DATA_DIR.mkdir(exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_bytes(plistlib.dumps({
        "Label": SCHEDULE_LABEL, "ProgramArguments": command, "WorkingDirectory": str(HERE),
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "StandardOutPath": str(SCHEDULE_LOG), "StandardErrorPath": str(SCHEDULE_LOG),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
    }))
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(PLIST_PATH)], capture_output=True)
    result = subprocess.run(["launchctl", "bootstrap", domain, str(PLIST_PATH)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"launchctl failed: {result.stderr.strip()}")
    return f"Daily run scheduled at {hour:02d}:{minute:02d}. Log: {SCHEDULE_LOG}"


def remove_schedule() -> str:
    import subprocess
    if not PLIST_PATH.exists():
        return "No daily run was scheduled."
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(PLIST_PATH)], capture_output=True)
    PLIST_PATH.unlink()
    return "Daily run removed."


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv", type=Path, help="CV file (.pdf/.docx/.txt). Needed only on first run or for a new CV.")
    parser.add_argument("--results-per-query", type=int, default=settings["results_per_query"],
                        help="max LinkedIn/Indeed/Glassdoor results per (job title, country) search")
    parser.add_argument("--countries", nargs="*", default=settings["countries"],
                        help="rest-of-Europe countries, e.g. --countries 'United Kingdom' Netherlands "
                             "(default: saved settings)")
    parser.add_argument("--sources", nargs="*", choices=SOURCES, default=settings["sources"],
                        help="which job sources to search (default: saved settings)")
    parser.add_argument("--hours", type=int, default=settings["hours"],
                        help="posting window in hours (default 48: overlaps daily runs; new jobs are flagged)")
    parser.add_argument("--schedule", metavar="HH:MM", help="run automatically every day at this time")
    parser.add_argument("--unschedule", action="store_true", help="remove the daily run")
    parser.add_argument("--no-open", action="store_true", help="don't open the report in the browser")
    parser.add_argument("--profile-only", action="store_true", help="only read the CV (--cv) and save profile.json")
    args = parser.parse_args()

    if args.unschedule:
        return print(remove_schedule())
    if args.schedule:
        hour, minute = map(int, args.schedule.split(":"))
        return print(install_schedule(hour, minute))

    profile = load_profile(args.cv)
    if profile is None:
        raise SystemExit("First run: pass your CV, e.g.  --cv ~/Documents/cv.pdf")
    if args.profile_only:
        return print("   Profile saved")
    print(f"=== {dt.datetime.now():%Y-%m-%d %H:%M} ===")
    _, out_dir = run_search(profile, args.sources, args.countries, args.results_per_query, args.hours)
    if not args.no_open:
        import webbrowser
        webbrowser.open((out_dir / "jobs.html").resolve().as_uri())


if __name__ == "__main__":
    main()
