"""
URL features for dataset v4 (16 Nov 2026). Everything is computed from the URL text only
(no web requests), so the features are reproducible and safe to compute for phishing URLs.

Group A - "paper" (20): the 20 features named in Dirchaoui et al. (2026), Sec. 2.2, as implemented
          in extract_features.py (kept unchanged as the reference baseline).
Group B - "new" (30): added in this thesis to cover weaknesses found in v3:
          brand impersonation and look-alike brands, hosting platforms, hacked-CMS paths,
          redirect / e-mail parameters, host-name structure, character statistics, TLD type.
Character n-grams of the URL (Group C) are built in the ML notebook (TF-IDF), not here.
"""
import math
import re
from collections import Counter
from urllib.parse import urlsplit, unquote

import common as C
import extract_features as ef

PAPER = list(ef.FEATURES)

# Brands most often impersonated (global + Arab region). Token match on the URL.
BRANDS = [
    # global
    "paypal", "microsoft", "office365", "outlook", "hotmail", "onedrive", "sharepoint", "apple", "icloud",
    "amazon", "netflix", "facebook", "instagram", "whatsapp", "google", "gmail", "linkedin", "yahoo",
    "dhl", "fedex", "ups", "usps", "chase", "wellsfargo", "bankofamerica", "citi", "hsbc", "coinbase",
    "binance", "metamask", "trustwallet", "steam", "roblox", "docusign", "adobe", "dropbox", "ebay",
    "walmart", "spotify", "att", "verizon", "telegram", "tiktok", "discord", "booking", "airbnb",
    # Arab region (from the v3 brand list)
    "alrajhi", "alahli", "riyadbank", "albilad", "alinma", "absher", "tawakkalna", "nafath", "aramco",
    "stc", "mobily", "saudipost", "sadad", "emiratesnbd", "adcb", "mashreq", "etisalat", "salik",
    "dubaipolice", "emiratespost", "uaepass", "banquemisr", "cibeg", "instapay", "fawry", "egyptpost",
    "vodafonecash", "attijariwafa", "cihbank", "barid", "inwi", "baridimob", "algerieposte", "qnb",
    "ooredoo", "qatarpost", "kfh", "nbk", "knet", "omanpost", "bankmuscat", "arabbank", "aramex",
    "talabat", "careem", "smsa", "naqel", "jumia", "namshi",
    # added 16 Nov: more Egyptian / regional brands
    "vodafone", "egyptair", "saudia", "alexbank", "bosta", "qatarairways",
]
BRAND_SET = set(BRANDS)
LONG_BRANDS = [b for b in BRANDS if len(b) >= 5]

# hacked content-management-system paths (very common on compromised Arab ccTLD sites in v3)
CMS_PATH = re.compile(r"/(wp-content|wp-includes|wp-admin|wp-json|plugins|themes|uploads|cgi-bin|administrator|components|modules|vendor)/", re.I)
REDIRECT_KEYS = re.compile(r"(^|&)(url|redirect|redirect_uri|redir|next|return|returnurl|continue|dest|destination|goto|target)=", re.I)
EMAIL_IN = re.compile(r"[a-z0-9._%+-]+(@|%40)[a-z0-9.-]+\.[a-z]{2,}", re.I)
LEGACY_TLDS = {"com", "net", "org", "edu", "gov", "mil", "int", "info", "biz"}
TOKEN = re.compile(r"[a-z0-9]+")
HOMO = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a"})


def entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def edit1(a, b):
    """True if a and b differ by exactly one edit (substitution, insertion or deletion)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def extract_new(url):
    raw = str(url)
    s = urlsplit(raw)
    host = (s.hostname or "").lower()
    ext = C.EXT(host)
    ext_priv = C.EXT_PRIV(host)
    label = ext.domain                                   # registered label, e.g. "cibeg" in cibeg.com
    sub = ext.subdomain
    path, query = s.path or "", s.query or ""
    low = raw.lower()
    tokens = TOKEN.findall(unquote(low))
    host_tokens = TOKEN.findall(host)
    label_tokens = TOKEN.findall(label)

    # brand impersonation: a brand name appears in the HOST NAME, but the registered label is not
    # exactly that brand (cib-egypt-secure.com.eg, paypal.com.account-update.top). Brands in the path
    # are counted separately, because legitimate pages also mention brands there (news articles, shops).
    host_flat = host.replace("-", "")
    brand_in_host = any(t in BRAND_SET for t in host_tokens) or any(b in host_flat for b in LONG_BRANDS)
    brand_mismatch = int(brand_in_host and label.replace("-", "") not in BRAND_SET)
    path_tokens_all = TOKEN.findall(unquote((path + " " + query).lower()))
    brand_in_path = int(any(t in BRAND_SET for t in path_tokens_all))
    lookalike = int(any(edit1(t, b) or (t.translate(HOMO) == b and t != b)
                        for t in host_tokens for b in LONG_BRANDS if len(t) >= 4))

    last = host.split(".")[-1] if host else ""
    tld_type = 0 if last in LEGACY_TLDS else (1 if len(last) == 2 else 2)   # 0 legacy gTLD, 1 country code, 2 new gTLD

    digits = sum(ch.isdigit() for ch in raw)
    letters = sum(ch.isalpha() for ch in raw)
    upper = sum(ch.isupper() for ch in raw)
    path_tokens = TOKEN.findall(path.lower())
    return {
        # host-name structure
        "host_length": len(host),
        "domain_label_length": len(label),
        "host_hyphens": host.count("-"),
        "host_digits": sum(ch.isdigit() for ch in host),
        "host_dots": host.count("."),
        "domain_label_entropy": round(entropy(label), 4),
        "is_hosting_platform": int(ext_priv.suffix != ext.suffix),           # x.vercel.app, y.github.io ...
        "punycode": int("xn--" in host),
        "tld_type": tld_type,
        "tld_in_subdomain_or_path": int(bool(re.search(r"\.(com|net|org)([.\-/]|$)", (sub + "." + path).lower()))),
        # brands
        "brand_mismatch": brand_mismatch,
        "brand_in_path": brand_in_path,
        "brand_lookalike": lookalike,
        # path / query
        "is_homepage": int(path in ("", "/") and not query),
        "cms_path": int(bool(CMS_PATH.search(path))),
        "php_page": int(path.lower().endswith(".php")),
        "html_page": int(bool(re.search(r"\.html?$", path.lower()))),
        "double_slash_in_path": int("//" in path),
        "at_symbol": int("@" in raw),
        "tilde": int("~" in raw),
        "has_port": int(s.port is not None),
        "redirect_param": int(bool(REDIRECT_KEYS.search(query))),
        "url_in_query": int("http" in unquote(query).lower()),
        "email_in_url": int(bool(EMAIL_IN.search(unquote(path + "?" + query)))),
        "query_length": len(query),
        # character statistics
        "digit_ratio": round(digits / max(len(raw), 1), 4),
        "uppercase_ratio": round(upper / max(letters, 1), 4),
        "longest_token": max((len(t) for t in tokens), default=0),
        "path_tokens": len(path_tokens),
        "path_entropy": round(entropy(path), 4),
    }


NEW = list(extract_new("https://example.com/a").keys())
ALL = PAPER + NEW


def extract_all(url):
    return {**ef.extract(url), **extract_new(url)}


if __name__ == "__main__":
    for u in ["https://www.cibeg.com/en/personal/online-banking",
              "https://cib-egypt-secure.com.eg/login/verify",
              "http://paypa1-support.xyz/wp-includes/signin.php?email=a%40b.com",
              "https://my-wallet-sync.pages.dev",
              "https://secure.paypal.com.account-update.top/?redirect=http://x"]:
        f = extract_new(u)
        print(u, {k: v for k, v in f.items() if v and k in ("brand_mismatch", "brand_in_path", "brand_lookalike",
                                                              "is_hosting_platform", "cms_path", "tld_type", "redirect_param",
                                                              "email_in_url", "tld_in_subdomain_or_path", "is_homepage")})
    print(len(PAPER), "paper +", len(NEW), "new =", len(ALL))
