"""Shared helpers for dataset v4 (16 Nov 2026)."""
import re
from urllib.parse import urlsplit
import tldextract

EXT = tldextract.TLDExtract(suffix_list_urls=())                                   # public suffixes only
EXT_PRIV = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)  # + vercel.app, github.io ...

STATIC_FILE = re.compile(r"\.(pdf|jpe?g|png|gif|svg|webp|zip|rar|docx?|xlsx?|pptx?|mp[34]|css|js|xml|txt|json|ico|woff2?)$", re.I)
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly",
              "shorturl.at", "tiny.cc", "rb.gy", "s.id", "t.ly", "shorturl.asia", "v.gd", "qrco.de", "bitly.com",
              "linktr.ee", "lnkd.in", "tr.ee", "short.gy", "urlz.fr", "u.to", "cli.re", "shorter.gg", "tinyurl.ph",
              "shrtco.de", "rebrandly.com", "bl.ink", "short.io", "wa.me",
              # added in the 17 Nov audit (found in the manual check)
              "surl.li", "flow.page", "ead.me", "qr-codes.io", "encurtador.dev", "qrco.de", "msha.ke"}
# host names that just encode an IP address (1.2.3.4.nip.io): treated like an IP address
IP_DNS = ("nip.io", "sslip.io", "xip.io")
# v4.2: temporary tunnels and dynamic-DNS / self-hosting names (used by developers and home servers - and by phishers)
DYNAMIC_HOSTS = ("duckdns.org", "ddns.net", "zapto.org", "dpdns.org", "no-ip.org", "no-ip.com", "no-ip.biz", "noip.me", "hopto.org",
                 "myftp.org", "myftp.biz", "sytes.net", "serveftp.com", "redirectme.net", "synology.me", "myds.me", "dynu.net",
                 "freeddns.org", "dynv6.net", "dyndns.org", "mooo.com", "afraid.org", "strangled.net", "chickenkiller.com", "us.to",
                 "ngrok.io", "ngrok-free.app", "ngrok.app", "ngrok.dev", "trycloudflare.com", "loca.lt", "localto.net",
                 "serveo.net", "gradio.live", "plesk.page", "webredirect.org")
ARAB_CC = {"eg", "sa", "ae", "ma", "qa", "kw", "jo", "dz", "tn", "bh", "om", "lb", "iq", "ly", "ps", "sy", "ye", "sd", "mr"}

def normalize_url(url):
    """Same rule for both classes: trim, drop #fragment, drop '/' at the very end."""
    return str(url).strip().split("#")[0].rstrip("/")

def host_of(url):
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""

def site_of(url):
    """Website identity used for grouping: registered domain, treating hosting platforms
    (x.vercel.app, y.github.io) as separate sites."""
    h = host_of(url)
    r = EXT_PRIV(h)
    return r.top_domain_under_public_suffix or h

def reg_domain(url):
    h = host_of(url)
    return EXT(h).top_domain_under_public_suffix or h

def is_ip(host):
    import ipaddress
    try:
        ipaddress.ip_address(host.strip("[]")); return True
    except ValueError:
        return False

def eligible(url):
    """A web-page URL with a real host name: http(s), not an IP, not a shortener, not a static file, <= 2000 chars."""
    if not re.match(r"^https?://", url, re.I) or len(url) > 2000:
        return False
    h = host_of(url)
    if not h or "." not in h or is_ip(h) or reg_domain(url) in SHORTENERS or h.endswith(IP_DNS):
        return False
    try:
        path = urlsplit(url).path
    except ValueError:
        return False
    return not STATIC_FILE.search(path)


def platform_of(url):
    """Hosting platform of the URL ('' if the site has its own registered domain)."""
    h = host_of(url)
    a, b = EXT_PRIV(h).suffix, EXT(h).suffix
    return a if a != b else ""

# Out of scope in BOTH classes (labels are ambiguous):
#  - gambling / betting / lottery sites: neither clearly phishing nor clearly legitimate
#  - student "clone" / practice projects on hosting platforms (e.g. github.io/netflix-clone): OpenPhish lists
#    many of them because they copy a brand's design, but most are coursework, not attacks
GAMBLING = re.compile(r"(casino|togel|gacor|jackpot|maxwin|betting|sportsbook|baccarat|roulette|lottery|lotto|lotere|satta|"
                      r"matka|poker|1xbet|1xlite|galabet|taruhan|judol|bet\d|\d+bet|slot\d|slot-|\d{3}bmw|toto\d|totoof|turfvip)", re.I)
GAMBLING_TOKENS = {"toto", "turf", "slot", "slots", "judi", "jitu", "bandar", "bet", "bets"}
# adult sites: legitimate but outside a marketing-security scope (removed from both classes, 17 Nov audit)
ADULT = re.compile(r"(porn|xxx|xnxx|xvideo|redtube|youporn|onlyfans|hentai|camgirl|livecam|desitales|bokep|rule34)", re.I)
ADULT_TOKENS = {"sex", "nude", "nudes", "escort", "escorts"}

def _host_tokens(h):
    return set(re.split(r"[^a-z]+", h.lower())) - {""}
DEMO_WORDS = re.compile(r"clone|cloning|tutorial|practice|project|projeto|assignment|portfolio|frontend|"
                        r"landing-page|homework|exercise|-ui\b|ui-", re.I)

# v4.2 (23 Nov): gaps found in the error analysis - Indonesian/Turkish/Indian betting sites that the host-only
# rule missed (slot/casino words in the path, "...bet" brand names, "rtp" slot sites, "4d" lottery sites)
GAMBLING_HOST_V42 = re.compile(r"(togel|gacor|maxwin|pagcor|idnpoker|sbobet|habanero|pgsoft|pragmaticplay|mahjongways|"
                               r"scatter|lotre|kalyan|kalyaan|dpboss|case-?battle|slot(?![h-z])|toto(?!r)|hoki(?=\d|-|\.|$)|bola\d|jitu\d)")
GAMBLING_PATH_V42 = re.compile(r"(^|[^a-z])(lotto|lottery|casino|togel|jackpot|baccarat|poker|idnpoker|sportsbook|"
                               r"habanero|pgsoft|sabung|tembak-ikan|live-?casino)([^a-z]|$)")
# words that are gambling only as a WHOLE path part (/slot/nxt) - not inside e.g. /slots-booking
GAMBLING_SEGMENT_V42 = {"slot", "slots", "keno", "jili", "spaceman", "mahjong", "playtech"}
BET_WORDS_OK = ("alphabet", "tibet", "sherbet", "sorbet", "sohbet", "abet", "quodlibet", "gibbet", "barbet", "debet")

def _gambling_v42(h, path):
    if GAMBLING_HOST_V42.search(h) or GAMBLING_PATH_V42.search(path):
        return True
    if {seg.rsplit(".", 1)[0] for seg in path.split("/")} & GAMBLING_SEGMENT_V42:
        return True
    for t in re.split(r"[.\-]", h):
        if t != "rtp" and re.search(r"^rtp|rtp\d|rtp$", t):
            return True
        if t.rstrip("s").endswith("bet") and len(t) > 4 and not t.rstrip("s").endswith(BET_WORDS_OK):
            return True
        if re.fullmatch(r"[a-z]{3,}4d\d*[a-z]{0,3}\d*", t):            # papa4d2, kapal4d, instal4dpro
            return True
    return False

def out_of_scope(url):
    h = host_of(url)
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        path = ""
    if GAMBLING.search(h) or _host_tokens(h) & GAMBLING_TOKENS or reg_domain(url) == "pmu.fr" or _gambling_v42(h, path):
        return "gambling"
    if ADULT.search(h) or _host_tokens(h) & ADULT_TOKENS:
        return "adult"
    if platform_of(url) and DEMO_WORDS.search(url.split("://", 1)[-1]):
        return "demo/clone project"
    return ""


def dedup_key(url):
    """Same page written differently (http vs https, with or without 'www.'): one key."""
    h = host_of(url)
    h = h[4:] if h.startswith("www.") else h
    rest = url.split("://", 1)[-1].split("/", 1)
    return h + "/" + (rest[1] if len(rest) > 1 else "")
