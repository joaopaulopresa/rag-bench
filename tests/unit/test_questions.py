from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ragbench.components.loaders import FolderLoader
from ragbench.core.types import Document, Generation, Message, Span, Usage
from ragbench.experiment.questions import (
    generate_questions,
    locate_quote,
    parse_draft,
    write_question_file,
)

HANDBOOK = Path(__file__).parent.parent.parent / "examples" / "handbook"
TEXT = "Rent is due\non the first day. Rent is paid by bank transfer. A deposit of three months applies."
DOCUMENT = Document(id="lease.txt", text=TEXT, title="Lease")


def test_quotes_are_located_despite_whitespace_but_only_when_unique() -> None:
    whole = Span(0, len(TEXT))
    assert locate_quote(TEXT, "due on  the first day", whole) == Span(8, 28)
    assert locate_quote(TEXT, "Rent is", whole) is None, "occurs twice"
    assert locate_quote(TEXT, "not in the text", whole) is None
    assert locate_quote(TEXT, "  ", whole) is None
    assert locate_quote(TEXT, "A deposit", Span(0, 30)) is None, "outside the passage"


def test_model_replies_become_drafts_only_when_the_quote_is_found() -> None:
    whole = Span(0, len(TEXT))
    good = 'Here you go: {"question": "When is rent due?", "quote": "due on the first day"}'
    draft = parse_draft(good, DOCUMENT, whole)
    assert draft is not None
    assert (draft.query, draft.answer) == ("When is rent due?", "due\non the first day")
    assert TEXT[draft.span.start : draft.span.end] == draft.answer
    assert (
        parse_draft('{"question": "When?", "quote": "on the second day"}', DOCUMENT, whole) is None
    )
    assert parse_draft('{"question": "", "quote": "A deposit"}', DOCUMENT, whole) is None
    assert parse_draft("no json here", DOCUMENT, whole) is None
    assert parse_draft("{broken json}", DOCUMENT, whole) is None


class QuotingGenerator:
    """Asks about the first sentence of whatever passage it is shown."""

    key = "quoting"

    async def generate(self, messages: Sequence[Message]) -> Generation:
        passage = messages[-1].content.split("Passage:\n", 1)[1]
        sentence = passage.split(". ")[0].lstrip("# ").split("\n")[-1]
        reply = json.dumps(
            {"question": f"What does the policy say about: {sentence[:30]}?", "quote": sentence}
        )
        return Generation(text=reply, model="quoting", usage=Usage(cost_usd=0.001))


async def test_generated_questions_load_as_gold_only_after_review(tmp_path: Path) -> None:
    for file in HANDBOOK.glob("*.txt"):
        (tmp_path / file.name).write_text(file.read_text())
    corpus = FolderLoader(None, str(tmp_path)).load()
    assert corpus.questions == ()

    drafts, usage = await generate_questions(
        corpus, QuotingGenerator(), count=4, seed=1, passage_tokens=80
    )
    path = write_question_file(drafts, tmp_path / "questions.json")

    assert 1 <= len(drafts) <= 4
    assert usage.cost_usd > 0
    assert FolderLoader(None, str(tmp_path)).load().questions == (), (
        "unreviewed drafts are not gold"
    )
    assert len(FolderLoader(None, str(tmp_path), include_unreviewed=True).load().questions) == len(
        drafts
    )

    data = json.loads(path.read_text())
    data["tests"][0]["reviewed"] = True
    path.write_text(json.dumps(data))
    reviewed = FolderLoader(None, str(tmp_path)).load()
    assert len(reviewed.questions) == 1
    question = reviewed.questions[0]
    assert reviewed.gold_text(question) == [drafts[0].answer]
