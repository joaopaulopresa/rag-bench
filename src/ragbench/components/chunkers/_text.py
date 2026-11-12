"""Pure text helpers shared by the chunkers: token counting, boundaries, headings."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

from ragbench.core.interfaces import Tokenizer
from ragbench.core.types import Span

_BLANK_LINE = re.compile(r"\n[ \t]*\n\s*")
_NEWLINE = re.compile(r"\n\s*")
_SENTENCE_END = re.compile(r"(?<=[.!?;])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_MARKDOWN = re.compile(r"^(#{1,6})\s+(\S.*)$")
_ARTICLE = re.compile(r"^(ARTICLE|Article|SECTION|Section|PART|Part)\s+([IVXLC]+|\d+)\b.*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]*)$")
_CAPITALS = re.compile(r"^[A-Z][A-Z0-9 ,&/'()\-]{3,}$")
_MAX_HEADING_CHARS = 90


class TokenIndex:
    """Token positions of one text, for fast token counts of any span.

    Args:
        text: The text.
        tokenizer: The tokenizer that defines what a token is.
    """

    def __init__(self, text: str, tokenizer: Tokenizer) -> None:
        self.text = text
        self.spans = tokenizer.token_spans(text)
        self._starts = [span.start for span in self.spans]

    def first_token_at(self, position: int) -> int:
        """Return the index of the first token that starts at or after ``position``."""
        return bisect_left(self._starts, position)

    def count(self, span: Span) -> int:
        """Return how many tokens start inside ``span``."""
        return self.first_token_at(span.end) - self.first_token_at(span.start)


def strip_span(text: str, span: Span) -> Span | None:
    """Shrink ``span`` so that it neither starts nor ends with whitespace.

    Returns:
        The shrunk span, or ``None`` if it holds only whitespace.
    """
    piece = text[span.start : span.end]
    stripped = piece.strip()
    if not stripped:
        return None
    start = span.start + (len(piece) - len(piece.lstrip()))
    return Span(start, start + len(stripped))


def cut(span: Span, points: Sequence[int]) -> list[Span]:
    """Cut ``span`` at every point that lies strictly inside it."""
    inside = sorted({point for point in points if span.start < point < span.end})
    edges = [span.start, *inside, span.end]
    return [Span(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


@dataclass(frozen=True, slots=True)
class Heading:
    """A heading line found in a text.

    Attributes:
        start: Position of the first character of the heading line.
        title: Heading text without markup, at most 90 characters.
        level: Depth, 1 for top-level headings.
    """

    start: int
    title: str
    level: int


def find_headings(text: str) -> list[Heading]:
    """Find heading lines: Markdown ``#``, ``ARTICLE IV``, numbered ``2.1 Title``, ALL CAPS.

    The rules are heuristics for contracts, policies and Markdown. A line is a heading only if
    it is short, so numbered sentences are not mistaken for headings.

    Example:
        >>> [(h.title, h.level) for h in find_headings("# A\\n\\ntext\\n\\n2.1 Scope\\nmore")]
        [('A', 1), ('2.1 Scope', 2)]
    """
    headings = []
    position = 0
    for line in text.splitlines(keepends=True):
        content = line.strip()
        start = position + (len(line) - len(line.lstrip()))
        position += len(line)
        if not content or len(content) > _MAX_HEADING_CHARS:
            continue
        markdown = _MARKDOWN.match(content)
        if markdown:
            headings.append(Heading(start, markdown.group(2).strip(), len(markdown.group(1))))
            continue
        article = _ARTICLE.match(content)
        if article:
            level = 1 if article.group(1).lower() in {"article", "part"} else 2
            headings.append(Heading(start, content, level))
            continue
        numbered = _NUMBERED.match(content)
        if numbered and not content.endswith((",", ";", ":")):
            headings.append(Heading(start, content, numbered.group(1).count(".") + 1))
            continue
        if _CAPITALS.match(content) and any(character.isalpha() for character in content):
            headings.append(Heading(start, content, 1))
    return headings


def section_path(headings: Sequence[Heading], position: int) -> list[str]:
    """Return the titles of the headings that enclose ``position``, outermost first.

    A heading encloses every position from its own start up to the next heading of the same or
    a shallower level.
    """
    path: list[Heading] = []
    for heading in headings:
        if heading.start > position:
            break
        while path and path[-1].level >= heading.level:
            path.pop()
        path.append(heading)
    return [heading.title for heading in path]


def boundary_levels(text: str, headings: Sequence[Heading]) -> list[list[int]]:
    """Return candidate cut points from the strongest boundary kind to the weakest.

    Levels: headings, blank lines, line breaks, sentence ends. Each point is the position where
    a new piece would start.
    """
    return [
        [heading.start for heading in headings],
        [match.end() for match in _BLANK_LINE.finditer(text)],
        [match.end() for match in _NEWLINE.finditer(text)],
        [match.end() for match in _SENTENCE_END.finditer(text)],
    ]


def sentence_spans(text: str) -> list[Span]:
    """Split ``text`` into sentences that together cover it without gaps.

    A sentence ends at ``.``, ``!``, ``?`` or ``;`` followed by whitespace and a capital letter
    or digit, or at a line break.
    """
    points = [match.end() for match in _NEWLINE.finditer(text)]
    points += [match.end() for match in _SENTENCE_END.finditer(text)]
    return [span for span in cut(Span(0, len(text)), points) if text[span.start : span.end].strip()]


def token_windows(count: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Return ``[first, last)`` token ranges of ``size`` tokens that overlap by ``overlap``.

    The last window may be shorter. No window is fully contained in the one before it.

    Example:
        >>> token_windows(10, 4, 1)
        [(0, 4), (3, 7), (6, 10)]
    """
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("size must be positive and 0 <= overlap < size")
    windows = []
    first = 0
    while first < count:
        last = min(first + size, count)
        windows.append((first, last))
        if last == count:
            break
        first = last - overlap
    return windows


def recursive_split(
    index: TokenIndex, span: Span, levels: Sequence[Sequence[int]], max_tokens: int
) -> list[Span]:
    """Split ``span`` into pieces of at most ``max_tokens`` at the strongest boundaries.

    The span is cut at the first boundary level that divides it. Pieces that are still too
    large are split again at the weaker levels, and finally by token count. Neighbouring pieces
    are then merged while they fit, so chunks are as large as allowed.

    Returns:
        Pieces in reading order that together cover ``span`` without gaps or overlap.
    """
    if index.count(span) <= max_tokens:
        return [span]
    for depth, points in enumerate(levels):
        pieces = cut(span, points)
        if len(pieces) > 1:
            split: list[Span] = []
            for piece in pieces:
                split.extend(recursive_split(index, piece, levels[depth + 1 :], max_tokens))
            return _merge(index, split, max_tokens)
    first = index.first_token_at(span.start)
    count = index.count(span)
    pieces = []
    for window_first, window_last in token_windows(count, max_tokens, 0):
        start = span.start if window_first == 0 else index.spans[first + window_first].start
        end = span.end if window_last == count else index.spans[first + window_last].start
        pieces.append(Span(start, end))
    return pieces


def _merge(index: TokenIndex, pieces: Sequence[Span], max_tokens: int) -> list[Span]:
    merged: list[Span] = []
    for piece in pieces:
        if merged and index.count(Span(merged[-1].start, piece.end)) <= max_tokens:
            merged[-1] = Span(merged[-1].start, piece.end)
        else:
            merged.append(piece)
    return merged
