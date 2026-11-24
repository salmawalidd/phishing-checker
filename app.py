# Phishing Link Checker – web app (bachelor thesis "Cybersecurity in Digital Marketing", GIU 2026)
# Built from the dashboard notebook (cells E1–E4) + the platform additions (live check, feedback, About tab).
# Needs in the same folder: engine_v4.joblib, dataset_v4.csv, common.py, features_v4.py, extract_features.py, live_check.py, feedback.py
# Optional settings (environment variables): GSB_API_KEY (Google Safe Browsing), HF_TOKEN + FEEDBACK_REPO (to keep the feedback)

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # small servers: one thread is faster than many
# Step E1 – Load the frozen v4.2 engine and define how ONE link is scored
# link -> clean -> 47 features + character patterns -> LightGBM + character model -> stacked probability
#      -> risk level + top 3 SHAP reasons + one reason sentence + recommended action
import re, html, io, joblib
import numpy as np, pandas as pd
from urllib.parse import urlsplit
import gradio as gr, shap
import common as C, features_v4 as F

def url_text(u):                      # needed to load the character model (same function as in the notebook)
    return u.lower().split("://", 1)[-1]
import __main__; __main__.url_text = url_text

ENG = joblib.load("engine_v4.joblib")
TAB, NGR, (PLATT, ISO, STACK) = ENG["tabular"], ENG["ngram"], ENG["combiners"]
FEATS = ENG["features"]
EXPLAINER = shap.TreeExplainer(TAB)
df = pd.read_csv("dataset_v4.csv", keep_default_na=False, na_values=[""])
KNOWN = dict(zip(df["url"], df["label"]))          # links in the dataset
# the time-based test set of the ML notebook (same rule, same seed): these links were NOT used to build the engine
SEED = 42
_ls = df.loc[df.label == 0, "site"].drop_duplicates().sample(frac=1, random_state=SEED)
HELD_OUT_SITES = set(_ls.head(int(len(_ls) * 0.15))) | set(df.loc[(df.label == 1) & (df.period == "recent"), "site"])
TEST_URLS = set(df.loc[df.site.isin(HELD_OUT_SITES), "url"])
print("engine:", ENG["final"], "|", len(FEATS), "features | cost threshold", round(ENG["threshold_cost"], 2))
if ENG.get("version") != "4.2":
    print("NOTE: this engine is older than v4.2 – re-run ml_v4_2.ipynb to get the newest engine_v4.joblib")

# ---- risk levels (fixed cut-offs) and what the team should do ----
LEVELS = ["Critical", "High", "Medium", "Low"]
def level_of(p):
    return "Critical" if p >= .75 else "High" if p >= .50 else "Medium" if p >= .25 else "Low"
ACTION = {"Critical": "Escalate immediately for security investigation. Block or report it after verification.",
          "High":     "Don't publish or click it – check it by hand first.",
          "Medium":   "Check it by hand if it leads to a login, payment or data form.",
          "Low":      "Lower priority – no immediate escalation needed, but verify the link before entering passwords or payment details."}
# share of links per level that were phishing: development (out-of-fold) / time-based test.
# Read from the engine file, so the dashboard always shows the numbers of the engine it uses.
SHARE = {"Critical": ("97%", "97%"), "High": ("65%", "66%"), "Medium": ("36%", "40%"), "Low": ("5%", "6%")}   # engines before v4.2
if "risk_share" in ENG:
    SHARE = {lvl: (f"{d * 100:.0f}%", f"{t * 100:.0f}%") for lvl, (d, t) in ENG["risk_share"].items()}

# ---- plain-language names and reasons for the 47 features ----
NICE = {"url_length": "link length", "num_digits": "digits", "num_special_chars": "special characters",
        "num_slashes": "slashes", "num_params": "query parameters", "num_subdomains": "subdomains",
        "has_https": "HTTPS", "has_suspicious_words": "suspicious words", "suspicious_word_count": "suspicious words (count)",
        "repeated_chars": "repeated characters", "has_hex": "hex / encoded strings", "tld_length": "length of the ending",
        "is_common_tld": "common ending", "num_directories": "folders in the path", "path_length": "path length",
        "entropy": "randomness of the domain name", "letter_digit_ratio": "letters per digit",
        "host_length": "host-name length", "domain_label_length": "domain-name length", "host_hyphens": "hyphens in the host",
        "host_digits": "digits in the host", "host_dots": "dots in the host", "domain_label_entropy": "randomness of the domain label",
        "is_hosting_platform": "free hosting platform", "punycode": "international look-alike characters",
        "tld_type": "type of ending", "tld_in_subdomain_or_path": "'.com' in the wrong place", "brand_mismatch": "brand name in the host",
        "brand_in_path": "brand name in the path", "brand_lookalike": "look-alike brand name", "is_homepage": "bare homepage",
        "cms_path": "website-system folder (/wp-includes/ …)", "php_page": ".php page", "html_page": ".html page",
        "double_slash_in_path": "'//' in the path", "at_symbol": "'@' in the link", "tilde": "'~' in the link", "has_port": "port number",
        "redirect_param": "redirect parameter", "url_in_query": "another link inside the link", "email_in_url": "e-mail address in the link",
        "query_length": "query length", "digit_ratio": "share of digits", "uppercase_ratio": "share of capital letters",
        "longest_token": "longest word", "path_tokens": "words in the path", "path_entropy": "randomness of the path"}
YESNO = {"has_https", "has_suspicious_words", "repeated_chars", "has_hex", "is_common_tld", "is_hosting_platform", "punycode",
         "tld_in_subdomain_or_path", "brand_mismatch", "brand_in_path", "brand_lookalike", "is_homepage", "cms_path", "php_page",
         "html_page", "double_slash_in_path", "at_symbol", "tilde", "has_port", "redirect_param", "url_in_query", "email_in_url"}
TLD_TYPE = {0: "classic (.com, .org …)", 1: "country (.eg, .sa …)", 2: "new (.top, .xyz …)"}

def fmt(f, v):
    if f == "tld_type": return TLD_TYPE.get(int(v), str(v))
    if f in YESNO: return "yes" if v else "no"
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) and v != int(v) else str(int(v)) if float(v) == int(v) else str(v)

def reason_up(f, v):      # a feature that pushes towards phishing, in plain words
    special = {"has_https": "no HTTPS", "is_hosting_platform": "hosted on a free website builder",
               "brand_mismatch": "a brand name in the host that is not the brand's own domain",
               "brand_lookalike": "a look-alike of a brand name", "punycode": "international look-alike characters",
               "cms_path": "a hidden website-system folder", "redirect_param": "a redirect parameter",
               "url_in_query": "another link hidden inside it", "email_in_url": "an e-mail address inside it",
               "tld_in_subdomain_or_path": "'.com' in the wrong place", "has_suspicious_words": "suspicious words (login, verify, wallet …)",
               "brand_in_path": "a brand name in the path", "php_page": "a .php page", "double_slash_in_path": "'//' inside the path",
               "has_hex": "random-looking code", "at_symbol": "an '@' in the link", "is_homepage": "a bare homepage",
               "tld_type": f"a {TLD_TYPE.get(int(v), '').split(' (')[0]} domain ending"}
    if f in special and (f != "has_https" or not v): return special[f]
    return f"{NICE[f]}: {fmt(f, v)}"

def reason_down(f, v):    # a feature that pushes towards legitimate
    if f == "has_https" and v: return "uses HTTPS"
    if f in YESNO and not v: return f"no {NICE[f]}"
    return f"{NICE[f]}: {fmt(f, v)}"

TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|gbraid|wbraid|dclid|msclkid|igshid|igsh|ttclid|twclid|li_fat_id|mc_cid|mc_eid|_ga|_gl|si)$", re.I)
def strip_tracking(url):
    s = urlsplit(url)
    if not s.query: return url, []
    kept, removed = [], []
    for part in s.query.split("&"):
        (removed.append(part.split("=")[0]) if TRACKING.match(part.split("=")[0]) else kept.append(part))
    return C.normalize_url(s._replace(query="&".join(kept)).geturl().rstrip("?")), removed

def prepare(url):
    """Clean one link with the SAME rules as the dataset. Returns (clean_url, removed_tags, problem or None)."""
    url = C.normalize_url(str(url or ""))
    if not re.match(r"^https?://[^/\s]+\.[^/\s]+", url, flags=re.I):
        return url, [], "Not a full link (needs http:// or https://)"
    url, removed = strip_tracking(url)
    host = C.host_of(url)
    if C.reg_domain(url) in C.SHORTENERS: return url, removed, "Shortened link – expand it first"
    if C.is_ip(host) or host.endswith(C.IP_DNS): return url, removed, "IP address instead of a name – check by hand"
    if not C.eligible(url): return url, removed, "Not a web page the engine was built for (a file such as a PDF, or a link over 2,000 characters)"
    return url, removed, None

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def score(urls):
    """Score clean links. Returns features, part scores and, per link: probability, level, top 3 reasons, sentence, action."""
    urls = list(urls)
    X = pd.DataFrame([F.extract_all(u) for u in urls])[FEATS]
    p_tab = TAB.predict_proba(X)[:, 1]
    p_ngr = NGR.predict_proba(pd.Series(urls))[:, 1]
    p = STACK.predict_proba(np.column_stack([logit(p_tab), logit(p_ngr)]))[:, 1]
    sv = EXPLAINER(X).values
    if sv.ndim == 3: sv = sv[:, :, 1]
    rows = []
    for i in range(len(urls)):
        x = X.iloc[i]
        # a yes/no feature is only given as a reason when its value points the same way (e.g. "no HTTPS" up, "uses HTTPS" down)
        def shown(f, towards_phishing):
            if f not in YESNO: return True
            bad = (x[f] == 0) if f == "has_https" else (x[f] == 1)
            return bad if towards_phishing else not bad
        up = [FEATS[j] for j in np.argsort(-sv[i]) if sv[i, j] > 0 and shown(FEATS[j], True)][:3]
        down = [FEATS[j] for j in np.argsort(sv[i]) if sv[i, j] < 0 and shown(FEATS[j], False)][:1]
        r_up = [reason_up(f, X.iloc[i][f]) for f in up]
        lvl = level_of(p[i])
        if lvl in ("Critical", "High"):
            sentence = "Flagged mainly because of " + (", ".join(r_up[:-1]) + " and " + r_up[-1] if len(r_up) > 1 else (r_up[0] if r_up else "the overall pattern of the link")) + "."
        elif lvl == "Medium":
            sentence = ("Some warning signs (" + (", ".join(r_up) or "weak") + "), but " + (reason_down(down[0], X.iloc[i][down[0]]) if down else "also normal signs") + ".")
        else:
            sentence = "Looks like normal links; strongest normal sign: " + (reason_down(down[0], X.iloc[i][down[0]]) if down else "none") + "."
        rows.append({"probability": float(p[i]), "level": lvl, "reasons": r_up, "sentence": sentence, "action": ACTION[lvl],
                     "p_features": float(p_tab[i]), "p_characters": float(p_ngr[i])})
    return X, sv, pd.DataFrame(rows)

# Step E2 – One link in detail: risk level, the link taken apart, SHAP reasons, the two parts of the engine
TUNNELS = (".gradio.live", ".ngrok.io", ".ngrok-free.app", ".ngrok.app", ".trycloudflare.com", ".loca.lt",
           ".duckdns.org", ".no-ip.org", ".ddns.net", ".synology.me", ".serveo.net")

def cap(s):
    return s[:1].upper() + s[1:]          # capital first letter only (keeps 'HTTPS')

def url_parts_html(url, feats):
    # take the URL apart: scheme / subdomain / domain / TLD / path / query
    s = urlsplit(url)
    ext = C.EXT(s.hostname or "")
    parts = [("scheme", s.scheme + "://", "risk" if s.scheme == "http" else "")]
    if ext.subdomain: parts.append(("subdomain", ext.subdomain + ".", ""))
    parts.append(("domain", ext.domain, "key"))
    if ext.suffix: parts.append(("TLD", "." + ext.suffix, ""))
    if s.path: parts.append(("path", s.path, ""))
    if s.query: parts.append(("query", "?" + s.query, ""))
    segs = "".join(f'<span class="seg {c}"><span class="txt">{html.escape(t)}</span>'
                   f'<span class="cap">{n}</span></span>' for n, t, c in parts)
    return f'<div class="url-anatomy">{segs}</div>'

def analyze_url(url, info=None):
    if not (url or "").strip():
        raise gr.Error("Paste a link to check it.")
    url, removed, problem = prepare(url)
    if problem and problem.startswith("Not a full link"):
        raise gr.Error("Paste the full link, including http:// or https://")
    if problem:   # shortened link / IP address: the models never saw these, so don't guess
        if info is not None: info.update(link=url, level="Can't judge", probability=None, is_platform=False)
        return (f'<div class="result"><section class="verdict unsure"><div class="v-text">'
                f'<h2>Can’t judge this link</h2><p>{problem}. It hides the real website, and the engine was '
                f'never trained on such links, so any score would be a guess. Expand it first '
                f'(for example with a link-expander site) and check the full address.</p>'
                f'</div><div class="v-meter"></div></section><!--LIVE--></div>')

    X, sv, o = score([url])
    r = o.iloc[0]
    if info is not None:
        info.update(link=url, level=r.level, probability=round(float(r.probability), 4),
                    is_platform=bool(X.iloc[0]["is_hosting_platform"]))
    prob = r.probability * 100
    lvl = r.level
    kind, title = lvl.lower(), f"{lvl} risk"
    text = (f"{r.action} In our evaluation, {SHARE[lvl][0]} of links at this level were phishing "
            f"({SHARE[lvl][1]} on the time-based test of newer phishing and unseen websites). {r.sentence}")
    known = ""
    if url in KNOWN:
        lab = "phishing" if KNOWN[url] == 1 else "legitimate"
        if url in TEST_URLS:
            known = (f'<p class="known">This link is in the held-out test set (labelled {lab}): '
                     f'it was not used to build the engine, so this is a fair test.</p>')
        else:
            known = (f'<p class="known">This exact link was used to build the engine (labelled {lab}), '
                     f'so this is not a fair test. Try a link it has not seen.</p>')
    scope = C.out_of_scope(url)
    if scope in ("gambling", "adult"):
        known += (f'<p class="known">This looks like a {scope} website. Such sites were left out of the training data '
                  f'on both sides, so treat the result with extra care.</p>')

    if C.host_of(url).endswith(TUNNELS):
        known += ('<p class="known">This is a temporary tunnel or self-hosted address. Developers use these for demos, '
                  'but phishers use them too – check who sent the link before trusting it.</p>')

    if removed:
        known += (f'<p class="known">Ignored tracking tags added by ad platforms: '
                  f'{html.escape(", ".join(removed))}.</p>')

    verdict = f'''
    <section class="verdict {kind}">
      <div class="v-text">
        <h2>{title}</h2>
        <p>{text}</p>
        {known}
      </div>
      <div class="v-meter">
        <div class="score"><b>{prob:.0f}%</b><span>estimated phishing likelihood<br>(model score)</span></div>
        <div class="meter"><span class="q q1"></span><span class="q q2"></span><span class="q q3"></span><div class="fill" style="width:{prob:.1f}%"></div></div>
        <div class="scale"><span>Low</span><span>Medium · High</span><span>Critical</span></div>
      </div>
    </section>'''

    anatomy = f'''
    <section class="block">
      <h3>The link, taken apart</h3>
      <p class="sub">The models only read this text. The page itself is never opened.</p>
      {url_parts_html(url, X)}
    </section>'''

    # SHAP reasons for the LightGBM part (top 8 features by impact)
    contrib = sorted(zip(FEATS, sv[0], X.iloc[0]), key=lambda t: -abs(t[1]))[:8]
    scale = max(abs(c) for _, c, _ in contrib) or 1
    rows = ""
    for f, c, v in contrib:
        w = abs(c) / scale * 50
        side = "up" if c > 0 else "down"
        rows += (f'<div class="r"><span class="name">{cap(NICE[f])} <em>{fmt(f, v)}</em></span>'
                 f'<span class="bar"><span class="b {side}" style="width:{w:.1f}%"></span></span>'
                 f'<span class="val {side}">{c:+.2f}</span></div>')
    reasons = f'''
    <section class="block">
      <h3>What the feature-based model saw (its score: {r.p_features * 100:.0f}%, final engine: {prob:.0f}%)</h3>
      <p class="sub">SHAP values of the LightGBM part (in the model's internal units, not percentage points): how much each
        feature of this link pushed the score <span class="k up">towards phishing</span> or <span class="k down">towards legitimate</span>.
        The character model also affects the final score, but it is not part of these explanations.</p>
      <div class="reasons">{rows}</div>
    </section>'''

    parts = [("Feature model (LightGBM, 47 features)", r.p_features * 100), ("Character model (n-grams)", r.p_characters * 100),
             ("Final engine (stacked)", prob)]
    mrows = "".join(
        f'<div class="m"><span class="name">{n}</span>'
        f'<span class="track"><span class="mid"></span><span class="dot {"up" if p >= 50 else "down"}" style="left:{p:.1f}%"></span></span>'
        f'<span class="pct">{p:.0f}%</span></div>' for n, p in parts)
    models = f'''
    <section class="block">
      <h3>The two parts of the engine</h3>
      <p class="sub">The final engine combines both parts with learned weights into one score. Scores reflect patterns in our evaluation data;
        they are not guaranteed real-world probabilities.
        The line marks 50%.</p>
      <div class="models">{mrows}</div>
    </section>'''

    cells = "".join(f'<div class="f"><span>{cap(NICE[f])}</span><b>{fmt(f, v)}</b></div>' for f, v in X.iloc[0].items())
    features = f'''
    <details class="block feats"><summary>All 47 features measured from this link</summary>
      <div class="fgrid">{cells}</div></details>'''

    return f'<div class="result">{verdict}<!--LIVE-->{anatomy}<div class="two">{reasons}{models}</div>{features}</div>'

# Step E3 – The dashboard engine: check a whole list of links at once
# (upload CSV/Excel or paste links) -> risk level, action and reason per link,
# a summary for the team, a ranked alert list and a downloadable report
import os, tempfile, csv

ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Can't judge": 4}
MAX_LINKS = 2000

def read_table(file):
    """Read an uploaded CSV/Excel file. Handles ',' or ';' separators and files without a header row."""
    path = file if isinstance(file, str) else file.name
    excel = path.lower().endswith((".xlsx", ".xls"))
    sep = ","
    if not excel:                           # ',' or ';' (Excel in Arabic/European settings saves with ';')
        sample = open(path, encoding="utf-8-sig", errors="ignore").read(4096)
        try: sep = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error: sep = ","
    read = (lambda **k: pd.read_excel(path, dtype=str, **k)) if excel else \
           (lambda **k: pd.read_csv(path, dtype=str, sep=sep, encoding="utf-8-sig", **k))
    t = read()
    if any(re.match(r"^\s*https?://", str(c), flags=re.I) for c in t.columns):   # first row is a link, not a header
        t = read(header=None)
        t.columns = ["url"] + [f"column_{i}" for i in range(1, t.shape[1])]
    return t

def group_choices(file):
    """Columns the user can group the summary by (everything except the link column)."""
    if file is None:
        return gr.update(choices=["(none)"], value="(none)")
    t = read_table(file).head(5)
    cols = [c for c in t.columns if str(c).strip().lower() not in ("url", "link", "links", "urls")]
    pref = next((c for c in cols if any(k in str(c).lower() for k in ("channel", "platform", "source"))), cols[0] if cols else "(none)")
    return gr.update(choices=["(none)"] + cols, value=pref)

def read_links(file, pasted):
    """Return a table with a 'link' column (+ any extra columns from the upload, e.g. channel)."""
    if file is not None:
        t = read_table(file).fillna("")
        col = next((c for c in t.columns if str(c).strip().lower() in ("url", "link", "links", "urls")), t.columns[0])
        t = t.rename(columns={col: "link"})
    else:
        lines = [l.strip() for l in (pasted or "").splitlines() if l.strip()]
        t = pd.DataFrame({"link": lines})
    if t.empty:
        raise gr.Error("Upload a CSV/Excel file or paste at least one link.")
    if len(t) > MAX_LINKS:
        raise gr.Error(f"Please check at most {MAX_LINKS} links at a time.")
    return t

def check_list(file, pasted, group_by=None, live=False):
    t = read_links(file, pasted)
    prep = [prepare(u) for u in t["link"]]
    t["checked_link"] = [p[0] for p in prep]
    t["problem"] = [p[2] for p in prep]
    ok = t["problem"].isna()
    t["level"], t["risk_%"], t["reason"] = "Can't judge", np.nan, t["problem"]
    for k in (1, 2, 3):
        t[f"reason_{k}"] = ""
    if ok.any():
        _, _, o = score(list(t.loc[ok, "checked_link"]))
        t.loc[ok, "level"] = o["level"].values
        t.loc[ok, "risk_%"] = (o["probability"].values * 100).round(0)
        t.loc[ok, "reason"] = o["sentence"].values
        for k in (1, 2, 3):
            t.loc[ok, f"reason_{k}"] = [rs[k - 1] if len(rs) >= k else "" for rs in o["reasons"]]
    t["action"] = [ACTION.get(l, "Check by hand – the tool cannot score this link.") for l in t["level"]]
    add_live_columns(t, live)                     # Google Safe Browsing + domain age (outside the model)
    order = lambda s: s.map(ORDER) if s.name == "level" else -s.fillna(-1) if s.name == "risk_%" else s
    t = t.sort_values(["_prio", "level", "risk_%"], key=order).reset_index(drop=True)   # Google-listed links first

    base = ("link", "checked_link", "problem", "level", "risk_%", "reason", "reason_1", "reason_2", "reason_3", "action",
            "safe_browsing", "domain_registered", "domain_age_days", "live_warning", "_prio")
    extra = [c for c in t.columns if c not in base]
    live_cols = ["live_warning", "safe_browsing", "domain_registered", "domain_age_days"] if live else []
    report = t[["level", "risk_%", "link", "reason_1", "reason_2", "reason_3", "reason", "action"] + live_cols + extra]
    path = os.path.join(tempfile.gettempdir(), "link_risk_report.csv")
    report.to_csv(path, index=False, encoding="utf-8-sig")
    return summary_html(t, extra, group_by), alerts_html(t, extra, group_by), path

def summary_html(t, extra, group_by=None):
    n = len(t)
    counts = t["level"].value_counts()
    tiles = ""
    for lvl in LEVELS + ["Can't judge"]:
        c = int(counts.get(lvl, 0))
        if lvl == "Can't judge" and c == 0:
            continue
        cls = "cant" if lvl == "Can't judge" else lvl.lower()
        hint = ACTION[lvl].split(" – ")[0].split(". ")[0] if lvl in ACTION else "Shortened, broken or IP links"
        tiles += (f'<div class="tile {cls}"><b>{c}</b><span class="lvl">{lvl}</span>'
                  f'<span class="pct">{c / n:.0%} of the list</span><span class="hint">{html.escape(hint)}</span></div>')
    bar = "".join(f'<span class="seg-{ "cant" if l == "Can't judge" else l.lower()}" style="width:{counts.get(l, 0) / n * 100:.2f}%" '
                  f'title="{l}: {int(counts.get(l, 0))}"></span>' for l in LEVELS + ["Can't judge"])
    urgent = int(counts.get("Critical", 0) + counts.get("High", 0))
    medium = int(counts.get("Medium", 0))
    headline = (f"{urgent} of {n} links are high priority" if urgent else f"No Critical or High links among {n}")
    subline = (f"{urgent} Critical or High links need attention first. " if urgent else "") + \
              (f"Another {medium} Medium link{'s' if medium != 1 else ''} may need a manual check." if medium else "")
    # breakdown by the first extra column with few values (e.g. channel / campaign)
    by = ""
    # breakdown by the column the user chose (e.g. channel or campaign)
    group_col = group_by if group_by in extra and group_by != "(none)" else None
    if group_col:
        g = pd.crosstab(t[group_col], t["level"]).reindex(columns=LEVELS + ["Can't judge"], fill_value=0)
        g["total"] = g.sum(axis=1)
        g["share"] = (g["Critical"] + g["High"]) / g["total"]          # share of each channel's links that are high priority
        g = g.sort_values(["share", "total"], ascending=False)
        rows = "".join(f'<tr><td>{html.escape(str(i))}</td>' + "".join(
            f'<td class="{"hot" if l in ("Critical", "High") and r[l] else ""}">{int(r[l])}</td>' for l in LEVELS)
            + f'<td>{int(r["total"])}</td><td class="{"hot" if r["share"] >= .5 else ""}">{r["share"]:.0%}</td></tr>'
            for i, r in g.iterrows())
        by = (f'<section class="block"><h3>By {html.escape(str(group_col))}</h3>'
              f'<p class="sub">Where the risky links come from. The last column is the share of that group\'s links '
              f'that are Critical or High.</p><table class="by"><thead><tr><th></th>'
              + "".join(f"<th>{l}</th>" for l in LEVELS) + f"<th>Total</th><th>High priority</th></tr></thead><tbody>{rows}</tbody></table></section>")
    return (f'<div class="result"><section class="summary"><h2>{headline}</h2>'
            f'<p class="sub">{subline}</p>{live_summary(t)}'
            f'<p class="sub">Each link gets an estimated phishing likelihood (model score) from the stacked engine and a risk level. '
            f'Share of links per level that were phishing in our evaluation: '
            + ", ".join(f"{l} {SHARE[l][0]}" for l in LEVELS)
            + ' (time-based test: ' + ", ".join(SHARE[l][1] for l in LEVELS) + ').</p>'
            f'<div class="stack">{bar}</div><div class="tiles">{tiles}</div></section>{by}</div>')

def alerts_html(t, extra, group_by=None, limit=60):
    alert = (t["level"] != "Low") | (t["_prio"] == 0)          # Low links listed by Google are shown too
    show = t[alert].head(limit)
    if show.empty:
        return '<div class="result"><section class="block"><h3>Alert list</h3><p class="sub">Nothing to review – every link is Low risk.</p></section></div>'
    ctx = group_by if group_by in extra else (extra[0] if extra else None)   # shown next to each link
    rows = ""
    for _, r in show.iterrows():
        cls = "cant" if r["level"] == "Can't judge" else r["level"].lower()
        risk = "–" if pd.isna(r["risk_%"]) else f'{r["risk_%"]:.0f}%'
        sub = f'<span class="ctx">{html.escape(str(r[ctx]))}</span>' if ctx else ""
        rows += (f'<tr><td><span class="pill {cls}">{r["level"]}</span></td><td class="num">{risk}</td>'
                 f'<td><code title="{html.escape(r["link"])}">{html.escape(r["link"][:90] + ("…" if len(r["link"]) > 90 else ""))}</code>{sub}'
                 f'<div class="why">{html.escape(str(r["reason"]))}{live_badges(r)}</div></td>'
                 f'<td class="act">{html.escape(r["action"])}</td></tr>')
    more = int(alert.sum()) - len(show)
    note = f'<p class="sub">Showing the first {limit}. {more} more are in the downloadable report.</p>' if more > 0 else ""
    return (f'<div class="result"><section class="block"><h3>Alert list</h3>'
            f'<p class="sub">Most urgent first. Low-risk links are left out here but included in the report. '
            f'Links are shown as text only – don’t open them.</p>'
            f'<table class="alerts"><thead><tr><th>Level</th><th>Risk</th><th>Link</th><th>What to do</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>{note}{gsb_attr(show)}</section></div>')

# ---- a demo list for trying the dashboard: links from the TIME-BASED TEST SET of notebook v4.1
#      (phishing first reported from 1 Aug 2026 + legitimate websites held out), never used to build the engine,
#      plus tricky cases (ad tracking tags, a shortened link, a broken link). The channel column is SIMULATED.
pool = df[df.site.isin(HELD_OUT_SITES)]           # the time-based test set defined in E1
rng = np.random.default_rng(7)
demo = pd.concat([pool[pool.label == 1].sample(14, random_state=7), pool[pool.label == 0].sample(16, random_state=7)])
demo = pd.DataFrame({"url": demo["url"].values})
demo = pd.concat([demo, pd.DataFrame({"url": [
    demo["url"].iloc[-1] + "?utm_source=facebook&utm_medium=cpc&fbclid=IwAR3xYz",
    "https://bit.ly/3xYzAbC", "www.example-shop.com/offer"]})], ignore_index=True)
demo["channel (simulated)"] = rng.choice(["Facebook ad", "Instagram comments", "Email campaign", "Influencer post", "WhatsApp"], len(demo))
demo = demo.sample(frac=1, random_state=7).reset_index(drop=True)
demo.to_csv("demo_campaign_links.csv", index=False, encoding="utf-8-sig")


# Step P – The live check and the feedback button (platform additions, week 2)
# The research engine above reads ONLY the URL text. The live check adds two outside sources NEXT TO its score:
# Google Safe Browsing (is the link on Google's lists right now?) and RDAP (how old is the domain?).
# They are shown separately and never change the model score, so the thesis results stay URL-only.
from concurrent.futures import ThreadPoolExecutor
import live_check as L, feedback as FB

POOL = ThreadPoolExecutor(max_workers=6)
NEW_DOMAIN_DAYS, YOUNG_DOMAIN_DAYS, OLD_DOMAIN_DAYS = 30, 180, 730

def _gsb_row(g):
    if g is None:
        return '<span class="lmuted">Not checked.</span>'
    if g["status"] == "listed":
        parts = ""
        for t in g["threats"]:
            title, desc, link, link_text = L.GSB_TEXT.get(t, ("Possibly unsafe", "Google lists this page as possibly unsafe.", "", ""))
            more = f' <a href="{link}" target="_blank" rel="noopener">{link_text}</a>' if link else ""
            parts += f'<b class="bad">{title}</b><p>{desc}{more}</p>'
        return parts + (f'<p class="attr">Advisory provided by <a href="{L.ADVISORY}" target="_blank" rel="noopener">Google</a></p>')
    if g["status"] == "not_listed":
        return ('<b class="ok">Not on Google’s lists right now</b>'
                '<p>This does not prove the link is safe – new phishing pages are often online before they are listed.</p>')
    return f'<span class="lmuted">{html.escape(g["note"])}</span>'

def _age_row(a, platform):
    if a is None:
        return '<span class="lmuted">Not checked for shortened links or IP addresses.</span>'
    if a["status"] == "ok":
        d = a["age_days"]
        if platform:
            return (f'<b>Registered {L.age_words(d)}</b> <span class="lmuted">({a["registered"]})</span>'
                    f'<p>This is the age of <code>{html.escape(platform)}</code>, the hosting platform – not of whoever made this page, '
                    f'so it says little about this link.</p>')
        cls, word = ("bad", "very new") if d < NEW_DOMAIN_DAYS else ("warn", "fairly new") if d < YOUNG_DOMAIN_DAYS else ("ok", "established")
        return (f'<b class="{cls}">Registered {L.age_words(d)}</b> <span class="lmuted">({a["registered"]} – {word})</span>'
                + ('<p>Brand-new domains are a common sign of phishing.</p>' if cls == "bad" else ""))
    if a["status"] == "not_found":
        return f'<b class="warn">Domain not found</b><p>{html.escape(a["note"])}</p>'
    return f'<span class="lmuted">{html.escape(a["note"])}</span>'

def _advice(level, g, a, platform):
    """One combined sentence. Rules are simple and visible on purpose (no hidden second model)."""
    if g and g["status"] == "listed":
        return ("Google lists this link as possibly unsafe. Treat it as Critical whatever the model score: "
                "don't click or publish it, block it and report it.", "bad")
    if a and a["status"] == "ok" and not platform:
        d = a["age_days"]
        if d < NEW_DOMAIN_DAYS and level in ("Critical", "High"):
            return (f"Model and domain age agree: {level.lower()} risk and a domain only {L.age_words(d).replace(' ago', '')} old. Escalate.", "bad")
        if d < NEW_DOMAIN_DAYS:
            lv = "unknown" if level == "Can't judge" else level.lower()
            return (f"The model score is {lv}, but the domain was registered only "
                    f"{L.age_words(d)}. New domains are a common phishing sign – check it by hand before trusting it.", "warn")
        if d >= OLD_DOMAIN_DAYS and level in ("Critical", "High"):
            return ("The model flags this link, but the domain is years old. It may be a hacked page on an old website "
                    "or a false alarm – check it by hand.", "warn")
    if a and a["status"] == "not_found":
        return ("The registry has no record of this domain, so it may already have been taken down. Don’t trust it.", "warn")
    return ("", "")

def live_block_html(level, g, a, platform):
    adv, cls = _advice(level, g, a, platform)
    adv_html = f'<div class="advice {cls}"><b>Combined advice:</b> {adv}</div>' if adv else ""
    return (f'<section class="block live"><h3>Live check <span class="tag">outside the model</span></h3>'
            f'<p class="sub">Two outside sources, checked just now. They do not change the model score above.</p>'
            f'<div class="lrow"><span class="lname">Google Safe Browsing</span><div class="lval">{_gsb_row(g)}</div></div>'
            f'<div class="lrow"><span class="lname">Domain age</span><div class="lval">{_age_row(a, platform)}</div></div>'
            f'{adv_html}</section>')

def check_one(url, live=True):
    """'Check one link': model result + (optional) live check, prepared for the feedback button."""
    raw = (url or "").strip()
    fut_g = fut_a = None
    if live and raw:
        clean, _, problem = prepare(raw)
        if not (problem or "").startswith("Not a full link"):
            fut_g = POOL.submit(L.safe_browsing, [clean])          # runs while the model is scoring
            if problem is None:
                fut_a = POOL.submit(L.domain_age, C.reg_domain(clean))
    info = {}
    page = analyze_url(url, info)                                  # raises a friendly error for empty / partial links
    g = a = None
    if fut_g:
        try: g = fut_g.result(timeout=L.TIMEOUT + 2).get(info["link"])
        except Exception: g = {"status": "unavailable", "threats": [], "note": "Google Safe Browsing did not answer in time."}
    if fut_a:
        try: a = fut_a.result(timeout=L.TIMEOUT * 2 + 2)
        except Exception: a = {"status": "unavailable", "registered": "", "age_days": None, "note": "The registry did not answer in time."}
    platform = ""
    if info.get("is_platform"):
        platform = C.platform_of(info["link"]) or C.reg_domain(info["link"])
    live_html = live_block_html(info["level"], g, a, platform) if live else ""
    page = page.replace("<!--LIVE-->", live_html, 1)
    state = {"link": info["link"], "level": info["level"], "model_score": info.get("probability"),
             "safe_browsing": (g or {}).get("status", "not checked"), "safe_browsing_threats": (g or {}).get("threats", []),
             "domain_age_days": (a or {}).get("age_days"), "engine_version": ENG.get("version", "?")}
    return page, state, gr.update(visible=True), gr.update(value=None), gr.update(value=""), ""

FEEDBACK_CHOICES = ["Yes – the risk level fits", "No – it is actually phishing", "No – it is actually safe", "Not sure"]

def send_feedback(state, answer, comment):
    if not state:
        raise gr.Error("Check a link first.")
    if not answer:
        raise gr.Error("Choose an answer first.")
    if state.get("sent"):
        return state, '<p class="fbok">Already saved for this link – thank you.</p>'
    mode = FB.save({**{k: v for k, v in state.items() if k != "sent"}, "answer": answer, "comment": (comment or "").strip()[:500]})
    state = {**state, "sent": True}
    where = "" if mode == "hub" else " (stored on this server only)"
    return state, f'<p class="fbok">Thank you – your feedback was saved{where}.</p>'

# ---- live check for a whole list ----
MAX_AGE_LOOKUPS = 100      # domain-age look-ups per list (registries limit how fast we may ask)

def add_live_columns(t, live):
    """Adds safe_browsing / domain_registered / domain_age_days / live_warning and a sort helper _prio (0 = listed by Google)."""
    t["safe_browsing"], t["domain_registered"], t["domain_age_days"], t["live_warning"], t["_prio"] = "", "", np.nan, "", 1
    t.attrs["live"] = {"on": bool(live)}
    if not live:
        return t
    valid = t["checked_link"].str.match(r"^https?://", case=False) & ~t["problem"].fillna("").str.startswith("Not a full link")
    g = L.safe_browsing(t.loc[valid, "checked_link"]) if valid.any() else {}
    words = {"listed": None, "not_listed": "not listed", "off": "not set up", "unavailable": "unavailable"}
    for i in t.index[valid]:
        r = g.get(t.at[i, "checked_link"])
        if not r: continue
        if r["status"] == "listed":
            names = [L.GSB_TEXT.get(x, ("possibly unsafe",))[0] for x in r["threats"]]
            t.at[i, "safe_browsing"] = "listed: " + "; ".join(names)
            t.at[i, "live_warning"] = "Google Safe Browsing: " + names[0].lower()
            t.at[i, "_prio"] = 0
        else:
            t.at[i, "safe_browsing"] = words[r["status"]]
    ok = t["problem"].isna()
    want = ok & ((t["level"] != "Low") | (t["_prio"] == 0))
    for i in t.index[want]:                                 # hosting platforms: the domain age is the platform's, so skip
        u = t.at[i, "checked_link"]
        if C.platform_of(u) or F.extract_all(u).get("is_hosting_platform"):
            want.at[i] = False
    doms = list(dict.fromkeys(C.reg_domain(u) for u in t.loc[want, "checked_link"]))[:MAX_AGE_LOOKUPS]
    ages = L.domain_ages(doms)
    for i in t.index[want]:
        a = ages.get(C.reg_domain(t.at[i, "checked_link"]))
        if not a: continue
        if a["status"] == "ok":
            t.at[i, "domain_registered"], t.at[i, "domain_age_days"] = a["registered"], a["age_days"]
            if a["age_days"] < NEW_DOMAIN_DAYS and not t.at[i, "live_warning"]:
                t.at[i, "live_warning"] = f"domain registered {L.age_words(a['age_days'])}"
        elif a["status"] == "not_found":
            t.at[i, "domain_registered"] = "not found"
            if not t.at[i, "live_warning"]:
                t.at[i, "live_warning"] = "domain not found in the registry"
        else:
            t.at[i, "domain_registered"] = "unavailable"
    t.attrs["live"] = {"on": True, "gsb_on": bool(L.gsb_key()), "age_checked": len(doms),
                       "age_skipped": max(0, len(dict.fromkeys(C.reg_domain(u) for u in t.loc[want, "checked_link"])) - len(doms))}
    return t

def live_summary(t):
    info = t.attrs.get("live", {})
    if not info.get("on"):
        return ""
    listed = int((t["_prio"] == 0).sum())
    new = int((t["domain_age_days"] < NEW_DOMAIN_DAYS).sum())
    g = (f"<b>{listed}</b> link{'s' if listed != 1 else ''} on Google Safe Browsing lists" if info.get("gsb_on")
         else "Google Safe Browsing is not set up")
    a = (f"<b>{new}</b> with a domain registered in the last {NEW_DOMAIN_DAYS} days "
         f"(domain age checked for {info.get('age_checked', 0)} domains of the non-Low links"
         + (f", {info['age_skipped']} more not checked" if info.get("age_skipped") else "") + ")")
    return f'<p class="sub livesum">Live check (outside the model): {g}; {a}.</p>'

def live_badges(r):
    w = r.get("live_warning", "")
    if not w:
        return ""
    cls = "bad" if w.startswith("Google") or "registered" in w else "warn"
    return f' <span class="lb {cls}">{html.escape(w)}</span>'

def gsb_attr(show):
    if "_prio" in show and (show["_prio"] == 0).any():
        return (f'<p class="attr">Google Safe Browsing labels: advisory provided by '
                f'<a href="{L.ADVISORY}" target="_blank" rel="noopener">Google</a>. '
                f'Google cannot guarantee that its information is complete and error-free.</p>')
    return ""

# Step E4 – The interface: 'Check one link' + 'Check a list' (dashboard) + 'About', then launch

CSS = """
:root{--ink:#14233A;--muted:#5A6B80;--line:#DCE3EA;--paper:#FFFFFF;--bg:#EEF2F5;
      --safe:#1D7A58;--safe-bg:#E6F3EC;--danger:#B8322A;--danger-bg:#FBEAE8;--unsure:#A86A12;--unsure-bg:#FBF1DF;--brand:#1F4E8C;
      --crit:#A3201B;--crit-bg:#FBE6E4;--high:#C2560E;--high-bg:#FDEDE1;--med:#A1700B;--med-bg:#FBF3DC;--low:#1D7A58;--low-bg:#E6F3EC;--cant:#5A6B80;--cant-bg:#EDF0F4}
html,body,gradio-app,.gradio-container,.main,.wrap,.contain{background:var(--bg)!important}
.plain,.plain>div,.plain .prose{background:transparent!important;border:none!important;box-shadow:none!important;padding:0!important;margin:0!important}
#examples,#examples>div{background:transparent!important;border:none!important;box-shadow:none!important;padding:6px 0 0!important}
.gradio-container{max-width:980px!important;margin:0 auto!important;padding:0 20px!important}
footer{display:none!important}
.app-head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;padding:36px 0 22px;border-bottom:1px solid var(--line);margin-bottom:26px}
.app-head h1{font-size:30px;line-height:1.15;font-weight:700;color:var(--ink);margin:0;letter-spacing:-.01em}
.app-head .ar{font-size:19px;color:var(--brand);margin:6px 0 0 50px;font-weight:500;text-align:left}
.app-head .meta{font-size:13px;color:var(--muted);text-align:right;line-height:1.5;max-width:260px}
.mark{display:inline-flex;width:38px;height:38px;border-radius:10px;background:var(--brand);color:#fff;align-items:center;justify-content:center;margin-right:12px;vertical-align:-6px}
.lead{font-size:16px;color:var(--muted);margin:0 0 14px;max-width:62ch}
#checkbar{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:8px;gap:8px!important;box-shadow:0 1px 0 rgba(20,35,58,.04)}
#checkbar textarea, #checkbar input{font-family:'IBM Plex Mono',monospace!important;font-size:15px!important;border:none!important;box-shadow:none!important;background:transparent!important;color:var(--ink)!important}
#checkbar .block,#checkbar .form{border:none!important;box-shadow:none!important;background:transparent!important}
#checkbtn{background:var(--brand)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;font-size:16px!important;min-height:48px}
#checkbtn:hover{background:#173E72!important}
#examples{margin-top:4px}
#examples .label, #examples button{font-size:13px!important}
#examples button{background:var(--paper)!important;border:1px solid var(--line)!important;border-radius:999px!important;color:var(--ink)!important}
.empty{color:var(--muted);font-size:15px;padding:34px 0 10px;border-top:1px dashed var(--line);margin-top:18px}
.empty ol{margin:10px 0 0 18px;padding:0;line-height:1.9}
.result{margin-top:22px;color:var(--ink)}
.verdict{display:grid;grid-template-columns:1.25fr 1fr;gap:28px;padding:26px 28px;border-radius:16px;border-left:8px solid;background:var(--paper)}
.verdict.critical{border-color:var(--crit);background:var(--crit-bg)} .verdict.high{border-color:var(--high);background:var(--high-bg)}
.verdict.medium{border-color:var(--med);background:var(--med-bg)} .verdict.low{border-color:var(--low);background:var(--low-bg)} .verdict.unsure{border-color:var(--cant);background:var(--cant-bg)}
.verdict h2{font-size:34px;line-height:1.1;margin:0 0 10px;font-weight:700;letter-spacing:-.015em}
.verdict.critical h2{color:var(--crit)} .verdict.high h2{color:var(--high)} .verdict.medium h2{color:var(--med)} .verdict.low h2{color:var(--low)} .verdict.unsure h2{color:var(--cant)}
.verdict p{font-size:15.5px;line-height:1.55;margin:0;max-width:52ch}
.verdict .known{margin-top:12px;font-size:13.5px;color:var(--muted);border-top:1px solid rgba(20,35,58,.12);padding-top:10px}
.score{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.score b{font-size:44px;line-height:1;font-weight:700}
.score span{font-size:13px;color:var(--muted);line-height:1.35}
.meter{position:relative;height:12px;border-radius:6px;background:rgba(20,35,58,.1)}
.meter .fill{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:var(--ink);opacity:.85}
.verdict.critical .fill{background:var(--crit)} .verdict.high .fill{background:var(--high)} .verdict.medium .fill{background:var(--med)} .verdict.low .fill{background:var(--low)}
.meter .q{position:absolute;top:0;bottom:0;border-left:2px solid #fff;z-index:1} .q1{left:25%} .q2{left:50%} .q3{left:75%}
.meter .tick{position:absolute;top:-5px;width:2px;height:22px;background:var(--ink);opacity:.55;transform:translateX(-1px)}
.meter .mid,.track .mid{position:absolute;left:50%;top:-4px;bottom:-4px;border-left:1px dashed var(--muted)}
.scale{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:10px}
.scale span:nth-child(2){color:var(--ink);font-weight:600}
.block{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin-top:18px}
.block h3{font-size:18px;margin:0 0 4px;font-weight:600}
.block .sub{font-size:13.5px;color:var(--muted);margin:0 0 16px;line-height:1.5}
.url-anatomy{display:flex;flex-wrap:wrap;row-gap:14px;font-family:'IBM Plex Mono',monospace;font-size:16px}
.seg{display:inline-flex;flex-direction:column;border-bottom:3px solid var(--line);padding-bottom:4px;margin-right:2px;word-break:break-all}
.seg .txt{padding:0 1px}
.seg .cap{font-family:'IBM Plex Sans Arabic',sans-serif;font-size:11.5px;color:var(--muted);margin-top:4px}
.seg.key{border-color:var(--brand)} .seg.key .txt{font-weight:600;color:var(--brand)}
.seg.risk{border-color:var(--danger)} .seg.risk .txt{color:var(--danger)}
.two{display:grid;grid-template-columns:1.15fr 1fr;gap:18px}
.two .block{margin-top:18px}
.k{font-weight:600} .k.up{color:var(--danger)} .k.down{color:var(--safe)}
.reasons .r{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) 46px;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #EEF1F4;font-size:14px}
.reasons .r:last-child{border-bottom:none}
.reasons em{font-style:normal;color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:12.5px;margin-left:4px}
.reasons .bar{position:relative;height:10px;background:linear-gradient(var(--muted),var(--muted)) center/1px 100% no-repeat}
.reasons .b{position:absolute;top:0;height:10px;border-radius:2px}
.reasons .b.up{left:50%;background:var(--danger)} .reasons .b.down{right:50%;background:var(--safe)}
.reasons .val{font-family:'IBM Plex Mono',monospace;font-size:12.5px;text-align:right}
.val.up{color:var(--danger)} .val.down{color:var(--safe)}
.models .m{display:grid;grid-template-columns:130px 1fr 40px;align-items:center;gap:12px;padding:9px 0;font-size:14px}
.track{position:relative;height:4px;background:#E3E8EE;border-radius:2px}
.dot{position:absolute;top:50%;width:14px;height:14px;border-radius:50%;transform:translate(-50%,-50%);border:2px solid #fff;box-shadow:0 0 0 1px rgba(20,35,58,.25)}
.dot.up{background:var(--danger)} .dot.down{background:var(--safe)}
.pct{font-family:'IBM Plex Mono',monospace;font-size:13px;text-align:right}
.feats summary{cursor:pointer;font-weight:600;font-size:15px}
.feats summary:focus-visible{outline:2px solid var(--brand);outline-offset:3px}
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px 18px;margin-top:16px}
.fgrid .f{display:flex;justify-content:space-between;border-bottom:1px solid #EEF1F4;padding:6px 0;font-size:13.5px;color:var(--muted)}
.fgrid .f b{color:var(--ink);font-family:'IBM Plex Mono',monospace;font-weight:500}
.app-foot{font-size:12.5px;color:var(--muted);line-height:1.6;padding:28px 0 40px;border-top:1px solid var(--line);margin-top:30px}

/* tabs */
#tabs>.tab-wrapper, #tabs .tab-container{border-bottom:1px solid var(--line)!important}
#tabs button[role=tab]{font-size:15px!important;font-weight:600!important;color:var(--muted)!important;padding:10px 16px!important}
#tabs button[role=tab][aria-selected=true]{color:var(--brand)!important;border-bottom:3px solid var(--brand)!important}
#tabs .tabitem{background:transparent!important;border:none!important;padding:18px 0 0!important}
/* list input */
#listbox{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:14px;gap:14px!important}
#listbox .block{border:none!important;box-shadow:none!important}
#listbox textarea{font-family:'IBM Plex Mono',monospace!important;font-size:13.5px!important}
#listbtn{background:var(--brand)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;min-height:44px}
#demobtn{background:var(--paper)!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:10px!important;min-height:44px}
#report{background:transparent!important;border:none!important}
/* summary */
.summary{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:24px 26px}
.summary h2{font-size:28px;margin:0 0 6px;font-weight:700;letter-spacing:-.01em}
.stack{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:18px 0 18px;background:#E3E8EE}
.stack span{display:block;height:100%}
.seg-critical{background:var(--crit)} .seg-high{background:var(--high)} .seg-medium{background:var(--med)} .seg-low{background:var(--low)} .seg-cant{background:#9AA6B5}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{border-radius:12px;padding:14px 16px;border-top:5px solid;display:flex;flex-direction:column;gap:2px}
.tile b{font-size:34px;line-height:1.05;font-weight:700}
.tile .lvl{font-weight:600;font-size:15px} .tile .pct{font-size:12.5px;color:var(--muted)} .tile .hint{font-size:12.5px;margin-top:6px;line-height:1.4}
.tile.critical{background:var(--crit-bg);border-color:var(--crit)} .tile.critical b,.tile.critical .lvl{color:var(--crit)}
.tile.high{background:var(--high-bg);border-color:var(--high)} .tile.high b,.tile.high .lvl{color:var(--high)}
.tile.medium{background:var(--med-bg);border-color:var(--med)} .tile.medium b,.tile.medium .lvl{color:var(--med)}
.tile.low{background:var(--low-bg);border-color:var(--low)} .tile.low b,.tile.low .lvl{color:var(--low)}
.tile.cant{background:var(--cant-bg);border-color:var(--cant)} .tile.cant b,.tile.cant .lvl{color:var(--cant)}
/* tables */
table.by,table.alerts{width:100%;border-collapse:collapse;font-size:14px;border:none!important}
table.by th,table.by td,table.alerts th,table.alerts td{border-left:none!important;border-right:none!important;border-top:none!important;background:transparent!important}
table.by th,table.alerts th{text-align:left;font-size:12.5px;color:var(--muted);font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line)}
table.by td,table.alerts td{padding:10px;border-bottom:1px solid #EEF1F4;vertical-align:top}
table.by td:not(:first-child),table.by th:not(:first-child){text-align:right;font-family:'IBM Plex Mono',monospace}
table.by td.hot{color:var(--crit);font-weight:600}
table.alerts code{font-family:'IBM Plex Mono',monospace;font-size:12.5px;word-break:break-all;background:none;padding:0;color:var(--ink)}
table.alerts .ctx{display:inline-block;margin-left:8px;font-size:12px;color:var(--muted)}
table.alerts .why{font-size:12.5px;color:var(--muted);margin-top:4px}
table.alerts .num{font-family:'IBM Plex Mono',monospace;white-space:nowrap}
table.alerts .act{font-size:13px;max-width:260px}
.pill{display:inline-block;font-size:12px;font-weight:600;padding:3px 9px;border-radius:999px;white-space:nowrap}
.pill.critical{background:var(--crit-bg);color:var(--crit)} .pill.high{background:var(--high-bg);color:var(--high)}
.pill.medium{background:var(--med-bg);color:var(--med)} .pill.cant{background:var(--cant-bg);color:var(--cant)}
@media (max-width:760px){ table.alerts .act{display:none} table.alerts th:last-child{display:none} .summary h2{font-size:23px} }
@media (max-width:760px){
  .app-head{flex-direction:column;align-items:flex-start;padding-top:24px}.app-head .meta{text-align:left}.app-head h1{font-size:25px}
  #checkbar{flex-direction:column!important} #checkbar>*{width:100%!important;min-width:0!important}
  .verdict{grid-template-columns:1fr;padding:20px}.verdict h2{font-size:28px}
  .two{grid-template-columns:1fr} .models .m{grid-template-columns:110px 1fr 36px}
  .reasons .r{grid-template-columns:1fr 90px 42px}
}
"""

CSS += """
/* live check */
.live h3 .tag{font-size:11.5px;font-weight:600;color:var(--muted);background:var(--bg);border:1px solid var(--line);border-radius:999px;padding:2px 8px;margin-left:8px;vertical-align:3px}
.lrow{display:grid;grid-template-columns:170px 1fr;gap:16px;padding:12px 0;border-top:1px solid #EEF1F4;font-size:14.5px}
.lrow .lname{font-weight:600;color:var(--ink)}
.lval p{margin:4px 0 0;color:var(--muted);font-size:13.5px;line-height:1.5}
.lval b.bad{color:var(--crit)} .lval b.warn{color:var(--med)} .lval b.ok{color:var(--safe)}
.lval a,.attr a{color:var(--brand)}
.lmuted{color:var(--muted);font-size:13.5px}
.attr{font-size:12.5px!important;color:var(--muted)!important;margin-top:6px!important}
.advice{margin-top:12px;padding:12px 14px;border-radius:10px;font-size:14.5px;line-height:1.5;background:var(--bg)}
.advice.bad{background:var(--crit-bg);color:var(--crit)} .advice.warn{background:var(--med-bg);color:#6E4C06}
.lb{display:inline-block;font-size:11.5px;font-weight:600;padding:2px 8px;border-radius:999px;margin-left:4px}
.lb.bad{background:var(--crit-bg);color:var(--crit)} .lb.warn{background:var(--med-bg);color:var(--med)}
.livesum{margin-top:-4px!important}
.livecheck,.livecheck .wrap,.livecheck label,.livecheck .block{background:transparent!important;border:none!important;box-shadow:none!important}
.livecheck{padding:6px 2px 0!important}
.form:has(.livecheck),.block:has(> .livecheck){background:transparent!important;border:none!important;box-shadow:none!important}
.livecheck label span{font-size:13.5px!important;color:var(--muted)!important;line-height:1.45!important}
/* feedback */
.fbwrap{background:var(--paper)!important;border:1px solid var(--line)!important;border-radius:14px!important;padding:20px 24px!important;margin-top:18px!important;gap:12px!important}
.fbwrap .block,.fbwrap .form,.fbwrap fieldset,.fbwrap .wrap,.fbwrap .gr-group{background:transparent!important;border:none!important;box-shadow:none!important;padding:0!important}
.fbwrap h3{font-size:16px;margin:0 0 2px;font-weight:600;color:var(--ink)} .fbwrap .sub{font-size:13px;color:var(--muted);margin:0}
.fbwrap fieldset .wrap{gap:8px!important}
.fbwrap fieldset label{background:var(--paper)!important;border:1px solid var(--line)!important;border-radius:999px!important;padding:7px 14px!important;font-size:13.5px!important}
.fbwrap fieldset label.selected{border-color:var(--brand)!important;background:#EAF1FA!important}
.fbwrap textarea,.fbwrap input[type=text]{background:var(--paper)!important;border:1px solid var(--line)!important;border-radius:10px!important;font-size:14px!important}
#fbbtn{background:var(--brand)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;max-width:200px;min-height:42px}
.fbok{color:var(--safe);font-weight:600;font-size:14px;margin:4px 0 0}
/* about */
.about{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:24px 28px;color:var(--ink);font-size:15px;line-height:1.6}
.about h3{font-size:18px;margin:22px 0 6px} .about h3:first-child{margin-top:0}
.about ul{margin:6px 0 0 18px;padding:0} .about li{margin:3px 0} .about a{color:var(--brand)}
.about table{border-collapse:collapse;font-size:14px;margin-top:8px} .about td,.about th{padding:6px 14px 6px 0;border-bottom:1px solid #EEF1F4;text-align:left}
@media (max-width:760px){ .lrow{grid-template-columns:1fr;gap:4px} }
"""

LIVE_LABEL = ("Also run the live check – sends the cleaned link to Google Safe Browsing and its domain name to the "
              "domain registry. Not perfect: some risky sites are missed and some safe ones are flagged.")

def about_html():
    rows = "".join(f'<tr><td>{l}</td><td>{SHARE[l][0]}</td><td>{SHARE[l][1]}</td><td>{html.escape(ACTION[l])}</td></tr>' for l in LEVELS)
    return f"""<div class="about">
<h3>What this tool does</h3>
<p>It rates how likely a link is to be phishing, using only the text of the link (the page is never opened). It is built for
marketing teams that handle many links – in ads, comments, influencer posts and messages – with a focus on the Arab region.</p>
<h3>How to read the result</h3>
<p>The score is an estimated phishing likelihood from our evaluation data, not a guarantee. Share of links per level that were phishing:</p>
<table><tr><th>Level</th><th>Development</th><th>Time-based test</th><th>What to do</th></tr>{rows}</table>
<h3>Live check (outside the model)</h3>
<ul><li><b>Google Safe Browsing</b> – whether the link is on Google's lists of unsafe pages right now. A link that is not listed can
still be phishing: new pages are often online before they are listed.</li>
<li><b>Domain age</b> – when the domain was registered, read from the registry through RDAP (the modern WHOIS). Brand-new domains
are common in phishing. For links on hosting platforms, the age is the platform's, so it is not shown as a warning.</li>
<li>Both are shown next to the model score and never change it.</li></ul>
<p>Google works to provide accurate and up-to-date information about unsafe web pages, but Google cannot guarantee that its
information is comprehensive and error-free: some risky sites may not be identified, and some safe sites may be identified in error.
Warnings based on Google data say "Advisory provided by Google" and link to the
<a href="https://developers.google.com/safe-browsing/v4/advisory" target="_blank" rel="noopener">Safe Browsing Advisory</a>.
The Safe Browsing API is used here for non-commercial research only; a commercial version would use Google's Web Risk API.</p>
<h3>Privacy</h3>
<ul><li>Ad tracking tags (utm_…, fbclid, gclid …) are removed before anything else happens.</li>
<li>With the live check on, the cleaned link is sent to Google Safe Browsing and the domain name to its registry. Turn it off if a
link contains private information.</li>
<li>Feedback you send (the cleaned link, the result and your answer) is stored to improve the engine. Don't send links that contain personal data.</li></ul>
<h3>Limits</h3>
<ul><li>Shortened links (bit.ly …) and IP addresses are not scored – expand them first.</li>
<li>Gambling and adult sites were left out of the training data, so results for them are less reliable.</li>
<li>The engine was trained on {ENG.get("dev_links", 0):,} links; phishing changes fast, so it needs regular retraining.</li></ul>
<p class="lmuted">Bachelor thesis prototype, Business Informatics, German International University (GIU), 2026. Method extends Dirchaoui et al. (2026).</p>
</div>"""

# keep the page in light mode, so the colours always look as designed
HEAD = """<script>
if (!location.search.includes('__theme=light')) {
  const u = new URL(location.href); u.searchParams.set('__theme','light'); location.replace(u.href);
}
</script>"""

THEME = gr.themes.Base(
    primary_hue="blue", neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans Arabic"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
)

SHIELD = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
          'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6z"/>'
          '<path d="M9 12l2 2 4-4"/></svg>')

EMPTY = """<div class="empty">Paste a link above and press <b>Check link</b>. You'll get:
<ol><li>a risk level from the machine-learning engine (feature model + character model),</li>
<li>a live check: Google Safe Browsing and the age of the domain,</li>
<li>the link taken apart into its pieces and the features that pushed the score up or down.</li></ol></div>"""

EXAMPLES = [
    ["http://secure-update.example.tn/wp-includes/login/verify.php?id=7f3a9c21b7e4d605&session=2"],
    ["https://vodafone.secure-login.xyz/pay"],
    ["https://www.uaeu.ac.ae/en/admissions"],
    ["https://portal.giu-uni.de/GIUb/EXTStudent/Home.aspx"],
    ["https://emiratespost.pkgq.top/ae"],
]
LABELS = ["Fake login page", "Brand impersonation (made up)", "University admissions page",
          "Student portal", "Known phishing link (training data)"]

with gr.Blocks(title="Phishing Link Checker – Arab region") as demo:
    gr.HTML(elem_classes="plain", padding=False, value=f"""
    <header class="app-head">
      <div><h1><span class="mark">{SHIELD}</span>Phishing Link Checker</h1>
           <p class="ar" lang="ar" dir="rtl">كاشف روابط التصيّد الاحتيالي</p></div>
      <div class="meta">For marketing teams working with links connected to the Arab region.<br>Research prototype, bachelor thesis 2026.</div>
    </header>""")

    with gr.Tabs(elem_id="tabs"):
        with gr.Tab("Check one link"):
            gr.HTML(elem_classes="plain", padding=False, value="""<p class="lead">Paste a link from an ad, comment,
            message or email. A machine-learning engine rates its risk from the URL text alone and shows you why.</p>""")
            with gr.Row(elem_id="checkbar", equal_height=True):
                url_input = gr.Textbox(show_label=False, placeholder="https://…", lines=1, max_lines=1,
                                       container=False, scale=5, autofocus=True)
                check_button = gr.Button("Check link", elem_id="checkbtn", scale=1, min_width=140)
            gr.Examples(EXAMPLES, inputs=url_input, example_labels=LABELS, label="Try one", elem_id="examples")
            live_one = gr.Checkbox(value=True, label=LIVE_LABEL, elem_classes="livecheck")
            result = gr.HTML(EMPTY, elem_classes="plain", padding=False)
            last = gr.State(None)                    # the last result, for the feedback button
            with gr.Column(visible=False, elem_classes="fbwrap") as fb_box:
                gr.HTML('<h3>Was this result right?</h3><p class="sub">Your answer helps to improve the engine. '
                        'The cleaned link, the result and your answer are saved – don\'t send links with personal data.</p>',
                        elem_classes="plain", padding=False)
                fb_answer = gr.Radio(FEEDBACK_CHOICES, show_label=False)
                fb_comment = gr.Textbox(show_label=False, placeholder="Optional: what do you know about this link?", lines=1, max_lines=3)
                fb_send = gr.Button("Send feedback", elem_id="fbbtn")
                fb_status = gr.HTML(elem_classes="plain", padding=False)

        with gr.Tab("Check a list (dashboard)"):
            gr.HTML(elem_classes="plain", padding=False, value="""<p class="lead">Check all links of a campaign at once –
            for example the links in your ads, in comments under your posts, or in influencer posts. Upload a CSV or Excel
            file with a <b>url</b> column (extra columns such as <b>channel</b> are kept), or paste one link per line.</p>""")
            with gr.Row(elem_id="listbox"):
                with gr.Column(scale=1):
                    file_in = gr.File(label="Upload CSV or Excel", file_types=[".csv", ".xlsx", ".xls"])
                with gr.Column(scale=1):
                    paste_in = gr.Textbox(label="…or paste links (one per line)", lines=6, max_lines=12,
                                          placeholder="https://…\nhttps://…")
            with gr.Row():
                group_in = gr.Dropdown(["(none)"], value="(none)", label="Group the summary by (e.g. channel or campaign)", scale=2)
                list_button = gr.Button("Check list", elem_id="listbtn", scale=2)
                demo_button = gr.Button("Load a demo campaign list", elem_id="demobtn", scale=1)
            live_list = gr.Checkbox(value=True, label=LIVE_LABEL + " For lists, domain age is checked for up to "
                                    f"{MAX_AGE_LOOKUPS} domains of the non-Low links.", elem_id="livelist")
            summary_out = gr.HTML(elem_classes="plain", padding=False,
                                  value='<div class="empty">Upload or paste links, then press <b>Check list</b>. '
                                        'You\'ll get a summary, an alert list (most urgent first) and a report to download.</div>')
            alerts_out = gr.HTML(elem_classes="plain", padding=False)
            report_out = gr.File(label="Download the full report (CSV, opens in Excel)", elem_id="report")

        with gr.Tab("About & privacy"):
            gr.HTML(about_html(), elem_classes="plain", padding=False)

    gr.HTML(elem_classes="plain", padding=False, value=f"""<div class="app-foot">How it works: each link is cleaned with the
    same rules as the training data, described by 47 URL features and by its character patterns, and scored by a stacked engine
    (LightGBM on the features + a character n-gram model, combined by logistic regression), trained on {ENG.get("dev_links", 19358):,} links from
    two phishing feeds and Chrome's popular-site lists. Score = estimated phishing likelihood, based on our evaluation data, not a guaranteed real-world probability; levels: Low &lt;25%, Medium 25–50%,
    High 50–75%, Critical ≥75%. On a time-based test (phishing first reported from Aug 2026, unseen websites) F1 was {ENG.get("test_f1", 0.908):.2f} and {SHARE["Critical"][1]} of
    Critical links were phishing, {SHARE["Low"][1]} of Low links – so Critical means "investigate first" and Low is not a guarantee. Reasons come from
    SHAP explanations of the feature model. The live check (Google Safe Browsing + domain age) is shown next to the score and never changes it. Ad tracking tags are ignored; shortened links must be expanded first. Channels in the demo
    list are simulated. Method extends Dirchaoui et al. (2026).</div>""")

    one_out = [result, last, fb_box, fb_answer, fb_comment, fb_status]
    check_button.click(check_one, [url_input, live_one], one_out)
    url_input.submit(check_one, [url_input, live_one], one_out)
    fb_send.click(send_feedback, [last, fb_answer, fb_comment], [last, fb_status])
    file_in.change(group_choices, file_in, group_in)
    list_button.click(check_list, [file_in, paste_in, group_in, live_list], [summary_out, alerts_out, report_out])
    demo_button.click(lambda: "demo_campaign_links.csv", None, file_in)

if __name__ == "__main__":
    # Render gives the port in $PORT and needs 0.0.0.0; on your laptop it opens http://127.0.0.1:7860
    import os
    on_server = "PORT" in os.environ
    demo.queue(default_concurrency_limit=2).launch(theme=THEME, css=CSS, head=HEAD, ssr_mode=False,
                server_name="0.0.0.0" if on_server else "127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
