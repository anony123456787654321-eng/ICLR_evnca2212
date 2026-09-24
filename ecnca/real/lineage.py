"""Lineage identity for real evidence.

PRIMARY, preregistered definition:

    root_id = deterministic canonicalisation of the cached-document identity

Every chunk, QA pair, summary and paraphrase derived from one document keeps
that root id.  Two different URLs are NOT assumed to be different sources --
they may mirror the same original reporting -- so the paper says
*lineage-distinct*, never *independent*.

A CONTENT-FAMILY id clustering near-identical copies is provided as a
sensitivity analysis only.  It never replaces the document root.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import parse_qsl, urlsplit, urlunsplit

# query parameters that never change which document is served
_TRACKING = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "ref", "referrer",
    "sh", "s", "amp", "at_medium", "at_campaign", "ito", "smid", "partner",
}
_AMP_SUFFIX = re.compile(r"(/amp|\.amp|/amp/)$", re.I)
# Archive wrappers embed the ORIGINAL url after a timestamp:
#   https://web.archive.org/web/20230323135844/https://en.wikipedia.org/wiki/X
# 70% of AVeriTeC answers are archived this way.  Without unwrapping, a page and
# its own snapshot -- or two snapshots at different timestamps -- become
# DIFFERENT roots, which manufactures evidence from a single document.
_ARCHIVE_WRAPPERS = re.compile(
    r"^https?://(?:web\.archive\.org/web/[^/]*/|"
    r"archive\.(?:ph|today|is|li|vn)/[^/]*/|"
    r"timetravel\.mementoweb\.org/memento/[^/]*/|"
    r"webcache\.googleusercontent\.com/search\?q=cache:[^/]*/)"
    r"(?P<inner>https?://.+)$", re.I)
_WS = re.compile(r"\s+")


def unwrap_archive(url: str, max_depth: int = 4) -> str:
    """Recover the original url from an archive wrapper.

    Applied before canonicalisation so an archived snapshot and the live page
    resolve to ONE document root.  Recursive, because archives occasionally wrap
    archives; bounded so a malformed url cannot loop.
    """
    u = (url or "").strip()
    for _ in range(max_depth):
        m = _ARCHIVE_WRAPPERS.match(u)
        if not m:
            break
        u = m.group("inner")
    return u


def canonical_url(url: str) -> str:
    """Stable, idempotent canonical form of a document URL.

    Deliberately conservative: it strips things that provably do not change the
    document served (scheme, www, tracking parameters, AMP suffixes, fragments,
    trailing slash) and nothing else.  Over-aggressive canonicalisation would
    merge genuinely different documents into one root and silently DESTROY
    evidence, which is the more dangerous error here.
    """
    if not url:
        return ""
    u = unwrap_archive(unicodedata.normalize("NFKC", url.strip()))
    if "://" not in u:
        u = "http://" + u
    parts = urlsplit(u)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m.") and len(host) > 2:
        host = host[2:]
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    path = _AMP_SUFFIX.sub("", path)
    if len(path) > 1:
        path = path.rstrip("/")
    keep = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in _TRACKING]
    query = "&".join(f"{k}={v}" for k, v in sorted(keep))
    return urlunsplit(("http", host, path or "/", query, ""))


def document_root_id(url: Optional[str] = None, cached_url: Optional[str] = None,
                     store_id: Optional[str] = None) -> str:
    """Root identity, preferring the most stable available handle.

    Order: knowledge-store id, then cached URL, then live URL.  The store id is
    preferred because a cached snapshot is the artefact actually used, and it
    does not drift when the live page changes.
    """
    for tag, value in (("store", store_id), ("cached", cached_url), ("url", url)):
        if value:
            base = value if tag == "store" else canonical_url(value)
            if base:
                digest = hashlib.blake2b(base.encode("utf-8"), digest_size=12).hexdigest()
                return f"doc::{digest}"
    return ""


def normalise_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", (text or "").lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return _WS.sub(" ", t).strip()


def content_hash(text: str) -> str:
    return hashlib.blake2b(normalise_text(text).encode("utf-8"),
                           digest_size=12).hexdigest()


def shingles(text: str, n: int = 5) -> set:
    words = normalise_text(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def content_family_ids(texts: Sequence[str], roots: Sequence[str],
                       threshold: float = 0.80) -> List[str]:
    """SENSITIVITY ONLY: cluster near-identical copies across different roots.

    Union-find over 5-shingle Jaccard.  Deterministic: clusters are named after
    the lexicographically smallest member root, so the labelling does not depend
    on input order.  This is reported alongside the document-root result, never
    instead of it.
    """
    n = len(texts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    sh = [shingles(t) for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            if roots[i] != roots[j] and jaccard(sh[i], sh[j]) >= threshold:
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = [""] * n
    for members in groups.values():
        name = min(roots[i] for i in members) or f"fam::{members[0]}"
        for i in members:
            out[i] = f"fam::{name}"
    return out
