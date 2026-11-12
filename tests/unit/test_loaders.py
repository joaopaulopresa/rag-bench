"""Corpus loaders, offline: a tiny fake dataset stands in for the Hugging Face repository."""

from __future__ import annotations

import hashlib
import json
import random
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from ragbench.components.caches import MemoryCache
from ragbench.components.loaders import FolderLoader, LegalBenchRagLoader, mini_selection
from ragbench.core.errors import DatasetError
from ragbench.core.registry import Registry
from ragbench.core.resources import Resources
from ragbench.core.types import Corpus, Document, Question, Span

POLICY = "﻿We collect your email.\r\nWe never sell it.\r\n"
REQUESTED: list[str] = []
POLICY_AS_READ = "﻿We collect your email.\nWe never sell it.\n"


def fake_repository() -> dict[str, bytes]:
    """Two sources. The offsets index the text as Python reads it: BOM kept, \\r\\n -> \\n."""
    files = {
        "corpus/privacy_qa/App.txt": POLICY.encode(),
        "corpus/cuad/A%20B.txt": b"Governing law: Delaware. Term: five years.",
        "corpus/cuad/Café.txt".encode().decode(): b"Renewal is automatic.",
    }
    privacy = {
        "tests": [
            {
                "query": "Do you sell my email?",
                "snippets": [
                    {
                        "file_path": "privacy_qa/App.txt",
                        "span": [24, 41],
                        "answer": "We never sell it.",
                    },
                    {
                        "file_path": "privacy_qa/App.txt",
                        "span": [1, 23],
                        "answer": "We collect your email.",
                    },
                ],
            }
        ]
    }
    cuad = {
        "tests": [
            {
                "query": "Which law governs?",
                "snippets": [
                    {
                        "file_path": "cuad/A%20B.txt",
                        "span": [0, 24],
                        "answer": "Governing law: Delaware.",
                    }
                ],
            },
            {
                "query": "Does it renew?",
                # NFD spelling of the path, as one upstream CUAD question has it
                "snippets": [
                    {
                        "file_path": "cuad/Café.txt",
                        "span": [0, 21],
                        "answer": "Renewal is automatic.",
                    }
                ],
            },
            {
                "query": "How long is the term?",
                "snippets": [
                    {"file_path": "cuad/A%20B.txt", "span": [25, 42], "answer": "Term: five years."}
                ],
            },
        ]
    }
    files["benchmarks/privacy_qa.json"] = json.dumps(privacy).encode()
    files["benchmarks/cuad.json"] = json.dumps(cuad).encode()
    sums = "".join(f"{hashlib.sha256(body).hexdigest()}  {name}\n" for name, body in files.items())
    files["SHA256SUMS"] = sums.encode()
    return files


@pytest.fixture
def resources(tmp_path: Path) -> Resources:
    return Resources(cache=MemoryCache(), data_dir=tmp_path / "data")


@pytest.fixture
def repository(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    files = fake_repository()
    requested = REQUESTED

    def download(self: LegalBenchRagLoader, client: object, relative: str) -> bytes:
        requested.append(relative)
        if relative not in files:
            raise DatasetError(f"cannot download {relative}: 404")
        return files[relative]

    monkeypatch.setattr(LegalBenchRagLoader, "_download", download)
    REQUESTED.clear()
    return files


def test_loads_verifies_and_keeps_only_documents_with_questions(
    resources: Resources, repository: dict[str, bytes]
) -> None:
    corpus = LegalBenchRagLoader(
        resources, sources=["privacy_qa", "cuad"], mini=2, revision="rev1"
    ).load()

    assert corpus.name == "legalbench-rag-mini"
    assert corpus.revision == "rev1"
    assert corpus.licences == {"privacy_qa": "MIT", "cuad": "CC BY 4.0", "queries": "MIT"}
    assert [question.source for question in corpus.questions] == ["privacy_qa", "cuad", "cuad"]
    assert len({question.id for question in corpus.questions}) == 3
    policy = corpus.document("privacy_qa/App.txt")
    assert policy.text == POLICY_AS_READ, "read in text mode: BOM kept, line endings normalised"
    assert policy.title == "App"
    first = corpus.questions[0]
    assert first.id == "privacy_qa-0000"
    assert first.spans == (Span(1, 23), Span(24, 41)), "spans are sorted"
    assert corpus.gold_text(first) == ["We collect your email.", "We never sell it."]
    for question in corpus.questions:
        assert question.document_id in {document.id for document in corpus.documents}
    assert {document.source for document in corpus.documents} == {"privacy_qa", "cuad"}


def test_files_are_downloaded_once(resources: Resources, repository: dict[str, bytes]) -> None:
    loader = LegalBenchRagLoader(resources, sources=["cuad"], mini=None, revision="rev1")
    first = loader.load()
    requested = list(REQUESTED)
    second = LegalBenchRagLoader(resources, sources=["cuad"], mini=None, revision="rev1").load()

    assert first == second
    assert first.name == "legalbench-rag"
    assert len(first.questions) == 3
    assert {document.id for document in first.documents} == {"cuad/A%20B.txt", "cuad/Café.txt"}
    assert first.document("cuad/A%20B.txt").title == "A B"
    assert requested.count("SHA256SUMS") == 1
    assert requested == REQUESTED, "the second load reads the local copies"


def test_a_changed_file_is_rejected(resources: Resources, repository: dict[str, bytes]) -> None:
    repository["corpus/cuad/A%20B.txt"] = b"Governing law: Delaware. Term: nine years."
    with pytest.raises(
        DatasetError, match=r"corpus/cuad/A%20B.txt: sha256 .* differs from SHA256SUMS"
    ):
        LegalBenchRagLoader(resources, sources=["cuad"], mini=None).load()


def test_a_span_that_does_not_match_its_answer_is_rejected(
    resources: Resources, repository: dict[str, bytes]
) -> None:
    broken = json.loads(repository["benchmarks/privacy_qa.json"])
    broken["tests"][0]["snippets"][0]["span"] = [23, 40]
    body = json.dumps(broken).encode()
    repository["benchmarks/privacy_qa.json"] = body
    sums = repository["SHA256SUMS"].decode().splitlines()
    sums = [line for line in sums if not line.endswith("benchmarks/privacy_qa.json")]
    sums.append(f"{hashlib.sha256(body).hexdigest()}  benchmarks/privacy_qa.json")
    repository["SHA256SUMS"] = ("\n".join(sums) + "\n").encode()
    with pytest.raises(
        DatasetError, match="span \\[23, 40\\) of privacy_qa/App.txt does not match"
    ):
        LegalBenchRagLoader(resources, sources=["privacy_qa"]).load()


def test_maud_and_unknown_sources_are_refused(resources: Resources) -> None:
    with pytest.raises(DatasetError, match="maud is excluded because its licence is not stated"):
        LegalBenchRagLoader(resources, sources=["maud"])
    with pytest.raises(DatasetError, match="source 'wiki' is not supported; allowed: privacy_qa"):
        LegalBenchRagLoader(resources, sources=["wiki"])


def test_mini_selection_is_the_literal_upstream_rule() -> None:
    tests = [
        {
            "query": f"q{number}",
            "snippets": [{"file_path": f"cuad/doc{number % 7}.txt", "span": [0, 1], "answer": "x"}],
        }
        for number in range(60)
    ]
    # legalbenchrag/benchmark.py, SORT_BY_DOCUMENT = True
    upstream = sorted(
        tests,
        key=lambda test: (random.seed(test["snippets"][0]["file_path"]), random.random())[1],  # type: ignore[index]
    )[:20]

    selected = mini_selection(tests, 20)  # type: ignore[arg-type]

    assert [test for _, test in selected] == upstream
    assert [position for position, _ in selected] == [tests.index(test) for test in upstream]
    documents = {test["snippets"][0]["file_path"] for _, test in selected}
    assert len(documents) <= 3, "questions about one document stay together"
    assert mini_selection(tests[:5], 20) == list(enumerate(tests[:5])), "small files are kept whole"  # type: ignore[arg-type]


def test_folder_loader_reads_documents_and_reviewed_questions(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Alpha beta gamma.")
    (tmp_path / "sub" / "b_notes.md").write_text("# Notes\n\nDelta.")
    (tmp_path / "ignored.csv").write_text("x,y")
    questions = {
        "tests": [
            {
                "query": "What comes second?",
                "reviewed": True,
                "snippets": [{"file_path": "a.txt", "span": [6, 10], "answer": "beta"}],
            },
            {
                "query": "Unreviewed",
                "snippets": [{"file_path": "a.txt", "span": [0, 5], "answer": "Alpha"}],
            },
        ]
    }
    (tmp_path / "questions.json").write_text(json.dumps(questions))

    corpus = FolderLoader(None, str(tmp_path), name="docs").load()

    assert [document.id for document in corpus.documents] == ["a.txt", "sub/b_notes.md"]
    assert corpus.document("sub/b_notes.md").title == "b notes"
    assert [question.text for question in corpus.questions] == ["What comes second?"]
    assert corpus.questions[0].id == "docs-0000"
    assert len(FolderLoader(None, str(tmp_path), include_unreviewed=True).load().questions) == 2


def test_folder_loader_reports_bad_input(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="is not a folder"):
        FolderLoader(None, str(tmp_path / "missing")).load()
    with pytest.raises(DatasetError, match="holds no .txt, .md or .pdf files"):
        FolderLoader(None, str(tmp_path)).load()
    (tmp_path / "a.txt").write_text("Alpha beta.")
    (tmp_path / "questions.json").write_text("[1, 2]")
    with pytest.raises(DatasetError, match="is not a question file"):
        FolderLoader(None, str(tmp_path)).load()
    wrong_span = {
        "tests": [
            {
                "query": "q",
                "reviewed": True,
                "snippets": [{"file_path": "a.txt", "span": [0, 4], "answer": "Alpha"}],
            }
        ]
    }
    (tmp_path / "questions.json").write_text(json.dumps(wrong_span))
    with pytest.raises(DatasetError, match="question 0: span \\[0, 4\\) does not match"):
        FolderLoader(None, str(tmp_path)).load()
    unknown = {
        "tests": [
            {
                "query": "q",
                "reviewed": True,
                "snippets": [{"file_path": "zzz.txt", "span": [0, 4], "answer": "Alph"}],
            }
        ]
    }
    (tmp_path / "questions.json").write_text(json.dumps(unknown))
    with pytest.raises(DatasetError, match="points to unknown 'zzz.txt'"):
        FolderLoader(None, str(tmp_path)).load()
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "questions.json").unlink()
    with pytest.raises(DatasetError, match="cannot read"):
        FolderLoader(None, str(tmp_path)).load()


def test_corpus_rejects_inconsistent_data() -> None:
    document = Document(id="a", text="text")
    with pytest.raises(DatasetError, match="duplicate document ids"):
        Corpus("c", (document, document), ())
    with pytest.raises(DatasetError, match="points to unknown document 'b'"):
        Corpus("c", (document,), (Question("q", "?", "b", (Span(0, 1),)),))
    with pytest.raises(DatasetError, match="has no document 'z'"):
        Corpus("c", (document,), ()).document("z")
    with pytest.raises(ValueError, match="invalid span"):
        Span(5, 2)


class PluginChunker:
    def __init__(self, resources: object, size: int = 1) -> None:
        self.size = size


def test_third_party_implementations_are_found_through_entry_points(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    good = EntryPoint("plugin", f"{__name__}:PluginChunker", "ragbench.test_plugins")
    broken = EntryPoint("broken", "no_such_module:Nothing", "ragbench.test_plugins")
    monkeypatch.setattr(
        "ragbench.core.registry.entry_points",
        lambda group: [good, broken] if group == "ragbench.test_plugins" else [],
    )
    registry: Registry[PluginChunker] = Registry("thing", "ragbench.test_plugins")

    assert registry.names() == ["plugin"]
    assert registry.create("plugin", None, {"size": 4}).size == 4
    assert "could not load thing 'broken'" in caplog.text
