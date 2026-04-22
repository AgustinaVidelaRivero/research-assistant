"""Mock web search for development; swap for a real search API in production."""

from __future__ import annotations

from research_assistant.core.state import Source

# Plausible domains and path patterns (templates indexed deterministically from the query).
_URL_TEMPLATES: list[str] = [
    "https://arxiv.org/abs/2301.{mod:04d}v1",
    "https://en.wikipedia.org/wiki/{slug}",
    "https://{sub}.stanford.edu/papers/{mod}-research",
    "https://mit.edu/research/publications/{year}/{mod}",
    "https://academic.oup.com/journal/article/{mod}/overview",
    "https://distill.pub/{year}/topic-{mod}",
    "https://blog.jupyter.org/notes-on-{slug}-{mod}",
    "https://medium.com/towards-data-science/{slug}-{mod}",
    "https://www.cs.berkeley.edu/~research/{mod}/index.html",
    "https://research.google/blog/{slug}-{mod}/",
    "https://openreview.net/forum?id=x{mod}{mod2}z",
    "https://semanticscholar.org/paper/{mod}{mod2}",
]

_TITLE_PREFIXES: list[str] = [
    "A survey of",
    "Recent advances in",
    "Critical review:",
    "Toward understanding",
    "Empirical study of",
    "Theoretical limits of",
    "Practical applications of",
    "State of the art in",
]

_TITLE_SUFFIXES: list[str] = [
    "— methods and benchmarks",
    ": a research synthesis",
    "in modern systems",
    "for practitioners",
    "and open questions",
    "(extended analysis)",
    "— literature overview",
    "across multiple domains",
]


def _slugify(q: str) -> str:
    return "-".join(w for w in q.lower().replace("/", " ").split() if w)[:48] or "topic"


def _score_for_index(i: int, total: int) -> float:
    if total <= 1:
        return 0.95
    return round(0.95 - (0.95 - 0.6) * (i / (total - 1)), 4)


def mock_web_search(query: str, max_results: int = 3) -> list[Source]:
    """Return fake but plausible Source objects for a query.

    This is a deterministic mock for development/testing.
    Production would call Tavily, Serper, DuckDuckGo, etc.
    """
    n = max(0, min(max_results, 8))
    if n == 0:
        return []

    h0 = abs(hash((query, "mock_search_v1")))
    slug = _slugify(query)
    out: list[Source] = []
    for i in range(n):
        hi = abs(hash((query, i, "row")))
        url_t = _URL_TEMPLATES[hi % len(_URL_TEMPLATES)]
        mod = h0 % 9000 + 1000
        mod2 = hi % 9000 + 1000
        year = 2019 + (hi % 6)
        sub = ("nlp", "ai", "ml", "hlt")[hi % 4]
        url = url_t.format(
            mod=mod,
            mod2=mod2,
            slug=slug,
            year=year,
            sub=sub,
        )
        tpre = _TITLE_PREFIXES[hi % len(_TITLE_PREFIXES)]
        tsuf = _TITLE_SUFFIXES[hi % len(_TITLE_SUFFIXES)]
        title = f"{tpre} {query.strip()[:80]} {tsuf}"
        snippet = (
            f"This paper and related work discuss «{query.strip()}» with emphasis on "
            f"methodological rigor, evaluation metrics, and trade-offs. Key themes include "
            f"reproducibility, datasets from {year}, and comparison with prior art; "
            f"implications for applied research and engineering practice are drawn."
        )[:200]
        out.append(
            Source(
                url=url,
                title=title[:200],
                snippet=snippet,
                relevance_score=_score_for_index(i, n),
            )
        )
    return out
