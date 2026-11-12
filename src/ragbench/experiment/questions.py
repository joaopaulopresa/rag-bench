"""Draft gold questions for your own documents, for a person to review."""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ragbench.components.chunkers import split_recursive
from ragbench.core.errors import ProviderError
from ragbench.core.interfaces import Generator
from ragbench.core.registries import tokenizers
from ragbench.core.types import Corpus, Document, Message, Span, Usage

_SYSTEM = (
    "You write evaluation questions for a document search system. "
    "Given a passage, write one question that a real user could ask and that the passage "
    "answers, and copy the shortest part of the passage that answers it, character for "
    "character. Do not mention 'the passage' in the question. "
    'Reply with one JSON object and nothing else: {"question": "...", "quote": "..."}'
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DraftQuestion:
    """A generated question with its evidence span. Not gold until a person reviews it.

    Attributes:
        query: The question.
        document_id: The document that answers it.
        span: Where the quoted evidence lies in the document.
        answer: The quoted evidence, equal to ``document.text[span.start:span.end]``.
    """

    query: str
    document_id: str
    span: Span
    answer: str


def locate_quote(text: str, quote: str, within: Span) -> Span | None:
    """Find ``quote`` in ``text`` inside ``within``, tolerating differences in whitespace.

    Returns:
        The span of the quote, or ``None`` if it does not occur exactly once.

    Example:
        >>> locate_quote("Rent is due\\non the first day.", "due on the first", Span(0, 29))
        Span(start=8, end=24)
    """
    words = quote.split()
    if not words:
        return None
    pattern = re.compile(r"\s+".join(re.escape(word) for word in words))
    matches = list(pattern.finditer(text, within.start, within.end))
    if len(matches) != 1:
        return None
    return Span(matches[0].start(), matches[0].end())


def parse_draft(reply: str, document: Document, passage: Span) -> DraftQuestion | None:
    """Turn a model reply into a draft question, or ``None`` if it is unusable.

    A reply is unusable when it is not the expected JSON or when its quote is not found exactly
    once in the passage; a question whose evidence cannot be located is worthless as gold.
    """
    match = _JSON_OBJECT.search(reply)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    query, quote = data.get("question"), data.get("quote")
    if not isinstance(query, str) or not isinstance(quote, str) or not query.strip():
        return None
    span = locate_quote(document.text, quote, passage)
    if span is None:
        return None
    return DraftQuestion(
        query=query.strip(),
        document_id=document.id,
        span=span,
        answer=document.text[span.start : span.end],
    )


async def generate_questions(
    corpus: Corpus,
    generator: Generator,
    *,
    count: int = 20,
    seed: int = 0,
    passage_tokens: int = 300,
) -> tuple[list[DraftQuestion], Usage]:
    """Draft up to ``count`` questions from randomly chosen passages of ``corpus``.

    Args:
        corpus: The documents to write questions about.
        generator: The model that writes them.
        count: How many passages to try. Passages whose reply is unusable give no question.
        seed: Seed of the passage choice.
        passage_tokens: Size of the passages shown to the model, in words.

    Returns:
        The drafts and the total usage of the model calls.
    """
    tokenizer = tokenizers.create("words", None, {})
    passages = [
        (document, span)
        for document in corpus.documents
        for span in split_recursive(document.text, tokenizer, passage_tokens)
        if len(span) >= 200
    ]
    random.Random(seed).shuffle(passages)
    drafts: list[DraftQuestion] = []
    usage = Usage(cached=True)
    for document, span in passages[:count]:
        passage = document.text[span.start : span.end]
        prompt = f"Document: {document.title or document.id}\n\nPassage:\n{passage}"
        try:
            generation = await generator.generate(
                [Message("system", _SYSTEM), Message("user", prompt)]
            )
        except ProviderError as error:
            logger.warning("no question for %s: %s", document.id, error)
            continue
        usage += generation.usage
        draft = parse_draft(generation.text, document, span)
        if draft is None:
            logger.warning("unusable reply for a passage of %s", document.id)
        else:
            drafts.append(draft)
    return drafts, usage


def write_question_file(drafts: Sequence[DraftQuestion], path: str | Path) -> Path:
    """Write drafts in the format the ``folder`` loader reads, marked as not reviewed.

    A person then checks each question, fixes or deletes it, and sets ``"reviewed": true``.
    Only reviewed questions are loaded as gold.
    """
    target = Path(path)
    tests = [
        {
            "query": draft.query,
            "reviewed": False,
            "snippets": [
                {
                    "file_path": draft.document_id,
                    "span": [draft.span.start, draft.span.end],
                    "answer": draft.answer,
                }
            ],
        }
        for draft in drafts
    ]
    target.write_text(
        json.dumps({"tests": tests}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target
