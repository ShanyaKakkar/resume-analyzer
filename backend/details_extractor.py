import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")

# Indian mobile numbers: optional +91, then 10 digits starting with 6-9.
# Allows a space or dash in the middle, like "90000 00000".
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)")

LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-_]+", re.I)

NOT_A_NAME = {"resume", "curriculum vitae", "cv", "biodata"}


def first_match(regex, *texts):
    """Search each text in order and return the first match found."""
    for text in texts:
        match = regex.search(text or "")
        if match:
            return match.group(0)
    return None


def clean_phone(raw):
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits


def add_https(url):
    if not url:
        return None
    return url if url.lower().startswith("http") else "https://" + url


def guess_name(header_text):
    """The name is usually one of the first lines: 1-4 words, letters only."""
    for line in (header_text or "").splitlines()[:5]:
        line = line.strip()
        lowered = line.lower()
        if not line or lowered in NOT_A_NAME:
            continue
        if "@" in line or re.search(r"\d", line):
            continue
        if any(word in lowered for word in ("http", "linkedin", "github")):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z.'\-]+", w) for w in words):
            return line.title() if line.isupper() else line
    return None


def extract_contact_details(header_text, full_text="", links=None):
    """Look in the header first, then the whole resume, and finally in the
    hidden link addresses (clickable words like "LinkedIn")."""
    links_text = "\n".join(links or [])
    return {
        "name": guess_name(header_text),
        "email": first_match(EMAIL_RE, header_text, full_text, links_text),
        "phone": clean_phone(first_match(PHONE_RE, header_text, full_text, links_text)),
        "linkedin": add_https(first_match(LINKEDIN_RE, header_text, full_text, links_text)),
        "github": add_https(first_match(GITHUB_RE, header_text, full_text, links_text)),
    }