"""
Feature extraction for the Arabic Social Media Phishing URL dataset,
following Dirchaoui et al. (2026), Procedia Computer Science 280, Section 2.2.

Usage:
    python extract_features.py raw_urls.csv features.csv

Input CSV needs at least two columns:
    url    - the full URL as it appeared in the ad / on the site
    label  - 1 = malicious, 0 = legitimate
Any other columns (source, date, notes...) are carried over unchanged.
Rows are never dropped: the output has exactly the input's rows, in the same
order (the script stops if there are blank or duplicate URLs).

Mapping to the paper (Section 2.2 names the 20 features but gives no formulas):
  EXACT (follows directly from the paper's wording)
    url_length, num_digits, num_slashes, num_fragments ('#' count),
    has_https, num_directories, path_length,
    entropy (Shannon entropy of the domain name: registered label, e.g. 'adnoc')
  ADAPTED (the paper leaves the detail open; choice documented here)
    num_special_chars   every non-alphanumeric character (incl. : / . - ? = &)
    num_params          number of non-empty key=value parts after '?'
    num_subdomains      labels before the registered domain; 'www' counts as one
    is_shortened_url    domain in our 19-item shortener list
    has_suspicious_words / suspicious_word_count
                        our 34-word list; a word counts when it is a whole token
                        of the URL (split on non-alphanumerics) or, for words of
                        5+ letters, the start or end of a token (loginverify)
    repeated_chars      same character 3+ times in a row, ignoring the scheme
                        and a leading 'www.' (otherwise 'www' alone triggers it)
    has_hex             %XX encoding, 0x followed by 2+ hex digits, or a run of
                        16+ hex characters containing both a digit and a-f
    tld_length          length of the TLD = last label (.com=3, .sa=2, .online=6)
    is_common_tld       last label in our list (generic TLDs + Arab ccTLDs + 4 others)
    has_ip              host is an IPv4 or IPv6 address (Python ipaddress)
    letter_digit_ratio  letters / max(digits, 1)
Revision 20 Oct 2026: tld_length (was the full public suffix, gov.sa=6),
repeated_chars, has_hex, suspicious-word matching and has_ip were changed as above.
"""

import ipaddress
import math
import re
import sys
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import tldextract

# Offline extractor: uses the public-suffix snapshot bundled with tldextract,
# so results are reproducible and need no internet.
_extract = tldextract.TLDExtract(suffix_list_urls=())

# --- Lists the paper describes but does not publish in full. -----------------
# Document whatever you finally use in your Methodology chapter.

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "s.id",
    "t.ly", "bl.ink", "short.io", "lnkd.in", "wa.me", "linktr.ee",
}

# Paper's examples: login, verify, banking, lottery. Extended with common
# ad-scam terms (English + transliterated Arabic). Keep lower-case.
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "account", "update",
    "secure", "banking", "bank", "lottery", "prize", "winner", "win",
    "gift", "free", "bonus", "reward", "claim", "offer", "promo",
    "job", "jobs", "hiring", "salary", "invest", "profit", "crypto",
    "wallet", "payment", "confirm", "password", "otp", "jaiza", "hadiya",
]

# "Common" TLDs used by legitimate sites. The paper names .com, .edu, .ma, .sa;
# the rest cover generic TLDs and Arab country codes. Compared on the last label.
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "int", "mil",
    # Arab-world country codes
    "ma", "sa", "eg", "ae", "qa", "kw", "bh", "om", "jo", "lb", "dz",
    "tn", "ly", "iq", "sy", "ps", "ye", "sd", "mr",
    # other widely used ccTLDs
    "uk", "fr", "de", "us",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
PCT_HEX = re.compile(r"%[0-9a-fA-F]{2}")
OX_HEX = re.compile(r"(?<![0-9a-zA-Z])0x[0-9a-fA-F]{2,}")
HEX_RUN = re.compile(r"[0-9a-fA-F]{16,}")
REPEAT_RE = re.compile(r"(.)\1{2,}")   # same character 3+ times in a row


def suspicious_count(lower: str) -> int:
    n = 0
    for tok in TOKEN_RE.findall(lower):
        for w in SUSPICIOUS_WORDS:
            if tok == w or (len(w) >= 5 and tok != w and (tok.startswith(w) or tok.endswith(w))):
                n += 1
                break          # one count per token
    return n


def has_hex(raw: str) -> bool:
    if PCT_HEX.search(raw) or OX_HEX.search(raw):
        return True
    return any(re.search(r"\d", m) and re.search(r"[a-fA-F]", m) for m in HEX_RUN.findall(raw))


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _ensure_scheme(url: str) -> str:
    return url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract(url: str) -> dict:
    raw = str(url).strip()
    parsed = urlparse(_ensure_scheme(raw))
    host = (parsed.hostname or "").lower()
    ext = _extract(host)

    registered = ext.top_domain_under_public_suffix or host   # e.g. example.co.uk
    suffix = ext.suffix or ""                    # e.g. co.uk
    last_label = suffix.split(".")[-1] if suffix else ""
    sub = ext.subdomain
    lower = raw.lower()

    digits = sum(ch.isdigit() for ch in raw)
    letters = sum(ch.isalpha() for ch in raw)
    path = parsed.path or ""
    dirs = [p for p in path.split("/") if p]

    sw_count = suspicious_count(lower)
    no_scheme = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", raw)
    no_scheme = re.sub(r"^www\.", "", no_scheme, flags=re.I)

    return {
        # basic
        "url_length": len(raw),
        "num_digits": digits,
        "num_special_chars": sum((not ch.isalnum()) for ch in raw),
        "num_slashes": raw.count("/"),
        "num_params": len([p for p in parsed.query.split("&") if p]) if parsed.query else 0,
        "num_fragments": raw.count("#"),
        "num_subdomains": len([s for s in sub.split(".") if s]) if sub else 0,
        # security
        "has_https": int(raw.lower().startswith("https://")),
        "is_shortened_url": int(registered in SHORTENERS or host in SHORTENERS),
        "has_suspicious_words": int(sw_count > 0),
        "suspicious_word_count": sw_count,
        "repeated_chars": int(bool(REPEAT_RE.search(no_scheme))),
        "has_hex": int(has_hex(raw)),
        # domain / TLD
        "tld_length": len(last_label),
        "is_common_tld": int(last_label in COMMON_TLDS),
        "has_ip": int(is_ip(host)),
        # path
        "num_directories": len(dirs),
        "path_length": len(path),
        # statistical
        "entropy": round(shannon_entropy(ext.domain or host), 4),
        "letter_digit_ratio": round(letters / max(digits, 1), 4),
    }


FEATURES = list(extract("http://example.com").keys())


def main(inp: str, out: str) -> None:
    df = pd.read_csv(inp)
    if not {"url", "label"} <= set(df.columns):
        sys.exit("Input needs 'url' and 'label' columns.")

    df["url"] = df["url"].astype(str).str.strip()
    blank = df["url"].isin(["", "nan"])
    dup = df["url"].duplicated(keep=False)
    if blank.any() or dup.any():
        sys.exit(f"STOP: {int(blank.sum())} blank and {int(dup.sum())} duplicated URLs in {inp}. "
                 "Fix the input first - rows are never dropped silently, so raw and "
                 "feature files always have the same rows.")

    feats = pd.DataFrame([extract(u) for u in df["url"]], index=df.index)
    result = pd.concat([df, feats], axis=1)
    result.to_csv(out, index=False, encoding="utf-8-sig")

    counts = result["label"].value_counts().to_dict()
    print(f"Saved {len(result)} rows, {len(FEATURES)} features -> {out}")
    print(f"Malicious (1): {counts.get(1, 0)}  |  Legitimate (0): {counts.get(0, 0)}  "
          f"(paper target: 800 / 800)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
