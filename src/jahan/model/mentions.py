"""Does a piece of text name a public figure?  Alias matching across scripts.

Latin script: accents removed and case folded, whole words only, so "Trump's" names Trump but
"Trumpism" does not, and "Trump" does not match inside the handle "@realDonaldTrump" (handles are
aliases of their own).  Arabic script (Persian, Arabic, Urdu): Arabic and Persian forms of yeh and kaf
unified, zero-width non-joiners treated as spaces, tatweel and short-vowel marks removed, then a
substring match.  Chinese and Devanagari: normalised (NFC) substring match; marks are never removed
because Devanagari vowel signs are combining characters.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Literal

Script = Literal["latin", "arabic", "cjk", "devanagari", "other"]
_ARABIC_MARKS = re.compile("[ً-ٰٟـ]")  # short vowels, superscript alef, tatweel


def script_of(text: str) -> Script:
    """The script most letters of ``text`` are written in."""
    counts: dict[Script, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        s: Script
        if name.startswith(("ARABIC", "PERSIAN")):
            s = "arabic"
        elif name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            s = "cjk"
        elif name.startswith("DEVANAGARI"):
            s = "devanagari"
        elif name.startswith("LATIN"):
            s = "latin"
        else:
            s = "other"
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "latin"
    return max(counts, key=lambda k: counts[k])


def normalise(text: str, script: Script) -> str:
    if script == "latin" or script == "other":
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
    t = unicodedata.normalize("NFC", text)
    if script == "arabic":
        t = t.replace("ي", "ی").replace("ك", "ک").replace("‌", " ")
        t = _ARABIC_MARKS.sub("", t)
    return t


@lru_cache(maxsize=4096)
def _pattern(alias: str) -> tuple[Script, str, re.Pattern[str] | None]:
    script = script_of(alias)
    norm = normalise(alias.strip(), script)
    if script in ("latin", "other"):
        return script, norm, re.compile(rf"(?<![\w@]){re.escape(norm)}(?!\w)")
    return script, norm, None


def mentions(text: str, aliases: list[str] | tuple[str, ...]) -> bool:
    """True when ``text`` names the figure by any of ``aliases``."""
    if not text:
        return False
    cache: dict[Script, str] = {}
    for alias in aliases:
        if not alias.strip():
            continue
        script, norm, pat = _pattern(alias)
        if script not in cache:
            cache[script] = normalise(text, script)
        t = cache[script]
        if pat is not None:
            if pat.search(t):
                return True
        elif norm and norm in t:
            return True
    return False
