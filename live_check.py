# Live check – two outside sources shown NEXT TO the model score (they never change the model score):
#   1) Google Safe Browsing Lookup API v4: is the link on Google's lists of unsafe pages right now?
#   2) RDAP (the modern WHOIS): when was the domain registered? Very new domains are common in phishing.
# Both are optional: if a key is missing, a service is slow or a registry does not answer,
# the app keeps working and just says "not available".
import os, time, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests

TIMEOUT = 4                      # seconds per request – the page must stay fast
UA = {"User-Agent": "phishing-link-checker-thesis/1.0 (research prototype)"}
_lock = threading.Lock()

# --------------------------------------------------------------------------------------------
# 1) Google Safe Browsing
# --------------------------------------------------------------------------------------------
GSB_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
THREAT_TYPES = ["SOCIAL_ENGINEERING", "MALWARE", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"]
_gsb_cache = {}                  # url -> (expires_at, result)
NEG_CACHE = 300                  # a "not listed" answer is reused for at most 5 minutes

def gsb_key():
    return os.environ.get("GSB_API_KEY", "").strip()

def _parse_duration(s, default=300):
    try:
        return float(str(s).rstrip("s"))
    except ValueError:
        return default

def safe_browsing(urls):
    """Check up to any number of links (sent in groups of 500).
    Returns {url: {"status": "listed" | "not_listed" | "off" | "unavailable", "threats": [...], "note": str}}"""
    urls = list(dict.fromkeys(u for u in urls if u))
    key = gsb_key()
    if not key:
        return {u: {"status": "off", "threats": [], "note": "Google Safe Browsing is not set up (no API key)."} for u in urls}
    out, todo, now = {}, [], time.time()
    with _lock:
        for u in urls:
            hit = _gsb_cache.get(u)
            if hit and hit[0] > now:
                out[u] = hit[1]
            else:
                todo.append(u)
    for i in range(0, len(todo), 500):
        chunk = todo[i:i + 500]
        body = {"client": {"clientId": "giu-phishing-link-checker", "clientVersion": "1.0"},
                "threatInfo": {"threatTypes": THREAT_TYPES, "platformTypes": ["ANY_PLATFORM"],
                               "threatEntryTypes": ["URL"], "threatEntries": [{"url": u} for u in chunk]}}
        try:
            r = requests.post(GSB_URL, params={"key": key}, json=body, timeout=TIMEOUT, headers=UA)
        except requests.RequestException:
            for u in chunk:
                out[u] = {"status": "unavailable", "threats": [], "note": "Google Safe Browsing did not answer in time."}
            continue
        if r.status_code != 200:
            why = {400: "the request or API key was rejected", 403: "the API key is not allowed to use Safe Browsing",
                   429: "the daily quota is used up"}.get(r.status_code, f"error {r.status_code}")
            for u in chunk:
                out[u] = {"status": "unavailable", "threats": [], "note": f"Google Safe Browsing is unavailable ({why})."}
            continue
        try:
            matches = r.json().get("matches", [])          # an empty answer {} means: none of the links is listed
        except ValueError:
            matches = None
        if matches is None:
            for u in chunk:
                out[u] = {"status": "unavailable", "threats": [], "note": "Google Safe Browsing sent an unreadable answer."}
            continue
        found = {}
        for m in matches:
            u = (m.get("threat") or {}).get("url")
            if u in chunk:
                f = found.setdefault(u, {"threats": set(), "ttl": 300})
                f["threats"].add(m.get("threatType", "UNKNOWN"))
                f["ttl"] = min(f["ttl"], _parse_duration(m.get("cacheDuration", "300s")))
        now = time.time()
        with _lock:
            for u in chunk:
                if u in found:
                    order = lambda t: THREAT_TYPES.index(t) if t in THREAT_TYPES else 99
                    res = {"status": "listed", "threats": sorted(found[u]["threats"], key=order), "note": ""}
                    _gsb_cache[u] = (now + found[u]["ttl"], res)
                else:
                    res = {"status": "not_listed", "threats": [], "note": ""}
                    _gsb_cache[u] = (now + NEG_CACHE, res)
                out[u] = res
    return out

# Google's required wording: qualifying language + a "learn more" link per threat type (see Safe Browsing usage rules)
GSB_TEXT = {
    "SOCIAL_ENGINEERING": ("Suspected phishing (deceptive site)",
                           "Attackers on this page may try to trick people into revealing personal information or installing software.",
                           "https://www.antiphishing.org/", "Learn about phishing"),
    "MALWARE": ("Possible malware",
                "Visiting this page may harm your computer: it appears to contain malicious code.",
                "https://developers.google.com/search/docs/monitor-debug/security/malware", "Learn about malware"),
    "UNWANTED_SOFTWARE": ("May contain harmful programs",
                          "The page may try to install programs that change or harm the browsing experience.",
                          "https://www.google.com/about/unwanted-software-policy.html", "Unwanted Software Policy"),
    "POTENTIALLY_HARMFUL_APPLICATION": ("May contain harmful apps",
                                        "The page may offer apps that could steal or delete data on a phone.", "", ""),
}
ADVISORY = "https://developers.google.com/safe-browsing/v4/advisory"

# --------------------------------------------------------------------------------------------
# 2) Domain age through RDAP (IANA bootstrap list -> the registry's own RDAP server)
# --------------------------------------------------------------------------------------------
BOOTSTRAP_URL = "https://data.iana.org/rdap/dns.json"
_boot = {"servers": None, "loaded": 0.0}
_rdap_cache = {}                 # domain -> (expires_at, result)

def _bootstrap():
    """Map each top-level domain to its registry's RDAP server (refreshed once a day)."""
    with _lock:
        if _boot["servers"] is not None and time.time() - _boot["loaded"] < 86400:
            return _boot["servers"]
    try:
        r = requests.get(BOOTSTRAP_URL, timeout=TIMEOUT, headers=UA)
        r.raise_for_status()
        servers = {}
        for tlds, urls in r.json()["services"]:
            base = next((u for u in urls if u.startswith("https://")), urls[0])
            for t in tlds:
                servers[t.lower()] = base if base.endswith("/") else base + "/"
    except Exception:
        servers = None
    with _lock:
        if servers:
            _boot.update(servers=servers, loaded=time.time())
        return _boot["servers"]

def _registration_date(data):
    for ev in data.get("events", []) or []:
        if str(ev.get("eventAction", "")).lower() == "registration" and ev.get("eventDate"):
            s = str(ev["eventDate"]).strip().replace("Z", "+00:00")
            try:
                d = dt.datetime.fromisoformat(s)
            except ValueError:
                try:
                    d = dt.datetime.strptime(s[:10], "%Y-%m-%d")
                except ValueError:
                    return None
            return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    return None

def domain_age(domain):
    """Returns {"status": "ok" | "no_date" | "not_found" | "unsupported" | "unavailable",
                "registered": "YYYY-MM-DD" or "", "age_days": int or None, "note": str}"""
    domain = (domain or "").strip(".").lower()
    if not domain or "." not in domain:
        return {"status": "unsupported", "registered": "", "age_days": None, "note": "No domain name to look up."}
    try:
        domain = domain.encode("idna").decode("ascii")          # Arabic / international names -> xn--…
    except UnicodeError:
        pass
    now = time.time()
    with _lock:
        hit = _rdap_cache.get(domain)
        if hit and hit[0] > now:
            return hit[1]
    servers = _bootstrap()
    if servers is None:
        res = {"status": "unavailable", "registered": "", "age_days": None, "note": "The registry directory could not be reached."}
        return res                                                # not cached: try again next time
    tld = domain.rsplit(".", 1)[-1]
    base = servers.get(tld)
    if not base:
        res = {"status": "unsupported", "registered": "", "age_days": None,
               "note": f"The registry for .{tld} does not offer RDAP lookups."}
    else:
        try:
            r = requests.get(base + "domain/" + domain, timeout=TIMEOUT,
                             headers={**UA, "Accept": "application/rdap+json, application/json"})
            if r.status_code == 404:
                res = {"status": "not_found", "registered": "", "age_days": None,
                       "note": "The registry has no record of this domain (it may have been deleted or never existed)."}
            elif r.status_code != 200:
                res = {"status": "unavailable", "registered": "", "age_days": None,
                       "note": f"The registry did not answer (error {r.status_code})."}
            else:
                d = _registration_date(r.json())
                if d is None:
                    res = {"status": "no_date", "registered": "", "age_days": None,
                           "note": "The registry does not publish the registration date."}
                else:
                    age = max(0, (dt.datetime.now(dt.timezone.utc) - d).days)
                    res = {"status": "ok", "registered": d.date().isoformat(), "age_days": age, "note": ""}
        except (requests.RequestException, ValueError):
            res = {"status": "unavailable", "registered": "", "age_days": None, "note": "The registry did not answer in time."}
    if res["status"] != "unavailable":
        with _lock:
            _rdap_cache[domain] = (now + 6 * 3600, res)
    return res

def domain_ages(domains, max_workers=8, deadline=20):
    """Look up many domains in parallel; anything not finished within `deadline` seconds is 'unavailable'."""
    domains = list(dict.fromkeys(d for d in domains if d))
    out = {}
    if not domains:
        return out
    ex = ThreadPoolExecutor(max_workers=max_workers)
    futs = {d: ex.submit(domain_age, d) for d in domains}
    end = time.time() + deadline
    for d, f in futs.items():
        try:
            out[d] = f.result(timeout=max(0.0, end - time.time()))
        except Exception:
            out[d] = {"status": "unavailable", "registered": "", "age_days": None, "note": "Not checked in time."}
    ex.shutdown(wait=False, cancel_futures=True)
    return out

def age_words(days):
    if days is None: return ""
    if days < 1: return "today"
    if days < 60: return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 730: return f"about {round(days / 30.4)} months ago"
    return f"about {days // 365} years ago"
