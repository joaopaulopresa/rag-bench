"""LegalBench-RAG, from the Hugging Face dataset ``awinml/legalbench-rag``.

Files are downloaded once at a pinned revision, checked against the repository's
``SHA256SUMS`` and every gold span is checked against its stored answer text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote

import httpx

from ragbench.core.errors import DatasetError
from ragbench.core.registries import register_loader
from ragbench.core.resources import Resources
from ragbench.core.types import Corpus, Document, Question, Span

DEFAULT_REPOSITORY = "awinml/legalbench-rag"
DEFAULT_REVISION = "2b9c248bc8179ef0908fd6ba01d50b156facd48b"
MINI_SIZE = 194

#: Licence of each source as stated in the dataset repository's LICENSE file. MAUD is absent
#: on purpose: its licence is not stated upstream, so this loader refuses to load it.
SOURCE_LICENCES = {
    "privacy_qa": "MIT",
    "contractnli": "CC BY 4.0",
    "cuad": "CC BY 4.0",
}
QUERIES_LICENCE = "MIT"

logger = logging.getLogger(__name__)


class _Snippet(TypedDict):
    file_path: str
    span: list[int]
    answer: str


class _Test(TypedDict):
    query: str
    snippets: list[_Snippet]


def mini_selection(tests: Sequence[_Test], size: int = MINI_SIZE) -> list[tuple[int, _Test]]:
    """Reproduce the upstream "mini" sample of one benchmark file.

    This is the rule of ``legalbenchrag/benchmark.py`` (``SORT_BY_DOCUMENT = True``): questions
    are sorted by a random number seeded with the path of their first snippet's document, and
    the first ``size`` are kept. Questions about one document stay together, so the sample
    touches few documents. The sort is stable, so file order breaks ties.

    Args:
        tests: The ``tests`` list of a benchmark file, in file order.
        size: Number of questions to keep.

    Returns:
        ``(position in the file, test)`` pairs of the selected questions, in selection order.
    """
    indexed = list(enumerate(tests))
    if len(indexed) <= size:
        return indexed

    def sort_key(item: tuple[int, _Test]) -> float:
        # A private generator gives the same numbers as upstream's `random.seed(path)` followed
        # by `random.random()`, without touching the global random state.
        return random.Random(item[1]["snippets"][0]["file_path"]).random()

    return sorted(indexed, key=sort_key)[:size]


@register_loader("legalbench_rag")
class LegalBenchRagLoader:
    """Loads LegalBench-RAG sources, by default the official mini sample of each.

    Only the documents that the selected questions point to are downloaded and returned.

    Args:
        resources: Shared dependencies; provides the download directory.
        sources: Benchmark files to load. Allowed: ``privacy_qa``, ``contractnli``, ``cuad``.
        mini: Questions kept per source with the upstream mini rule; ``None`` keeps all.
        revision: Commit of the dataset repository. Pinned so that results are reproducible.
        repository: Hugging Face dataset id.

    Raises:
        DatasetError: If a source is unknown or has no stated licence (``maud``).
    """

    def __init__(
        self,
        resources: Resources,
        sources: Sequence[str] = ("privacy_qa", "contractnli", "cuad"),
        mini: int | None = MINI_SIZE,
        revision: str = DEFAULT_REVISION,
        repository: str = DEFAULT_REPOSITORY,
    ) -> None:
        for source in sources:
            if source not in SOURCE_LICENCES:
                allowed = ", ".join(SOURCE_LICENCES)
                raise DatasetError(
                    f"source {source!r} is not supported; allowed: {allowed} "
                    "(maud is excluded because its licence is not stated upstream)"
                )
        self._sources = tuple(sources)
        self._mini = mini
        self._revision = revision
        self._repository = repository
        self._root = resources.data_dir / "legalbench-rag" / revision
        self._transport = resources.transport
        self._checksums: dict[str, str] | None = None

    def load(self) -> Corpus:
        """Download what is missing, verify it and return the corpus.

        Raises:
            DatasetError: If a download fails, a checksum differs or a gold span does not match
                its stored answer text.
        """
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            documents: dict[str, Document] = {}
            questions: list[Question] = []
            for source in self._sources:
                raw = self._fetch(client, f"benchmarks/{source}.json").read_text(encoding="utf-8")
                tests: list[_Test] = json.loads(raw)["tests"]
                selected = (
                    mini_selection(tests, self._mini)
                    if self._mini is not None
                    else list(enumerate(tests))
                )
                for position, test in selected:
                    question = self._question(client, source, position, test, documents)
                    questions.append(question)
        logger.info(
            "loaded LegalBench-RAG %s: %d questions over %d documents",
            "+".join(self._sources),
            len(questions),
            len(documents),
        )
        return Corpus(
            name="legalbench-rag" + ("-mini" if self._mini is not None else ""),
            documents=tuple(documents[key] for key in sorted(documents)),
            questions=tuple(questions),
            revision=self._revision,
            licences={
                **{source: SOURCE_LICENCES[source] for source in self._sources},
                "queries": QUERIES_LICENCE,
            },
        )

    def _question(
        self,
        client: httpx.Client,
        source: str,
        position: int,
        test: _Test,
        documents: dict[str, Document],
    ) -> Question:
        paths = {unicodedata.normalize("NFC", snippet["file_path"]) for snippet in test["snippets"]}
        if len(paths) != 1:
            raise DatasetError(f"{source} question {position} spans several documents: {paths}")
        document_id = paths.pop()
        if document_id not in documents:
            # Default text mode, as the dataset card requires: "\r\n" becomes "\n" and a
            # leading byte order mark is kept, otherwise the gold offsets shift.
            text = self._fetch(client, f"corpus/{document_id}").read_text(encoding="utf-8")
            documents[document_id] = Document(
                id=document_id, text=text, title=_title(document_id), source=source
            )
        text = documents[document_id].text
        spans = []
        for snippet in test["snippets"]:
            start, end = snippet["span"]
            if text[start:end] != snippet["answer"]:
                raise DatasetError(
                    f"{source} question {position}: span [{start}, {end}) of {document_id} "
                    "does not match its answer text"
                )
            spans.append(Span(start, end))
        return Question(
            id=f"{source}-{position:04d}",
            text=test["query"],
            document_id=document_id,
            spans=tuple(sorted(spans)),
            source=source,
        )

    def _fetch(self, client: httpx.Client, relative: str) -> Path:
        target = self._root / relative
        if target.exists():
            return target
        expected = self._expected_checksum(client, relative)
        content = self._download(client, relative)
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise DatasetError(f"{relative}: sha256 {actual} differs from SHA256SUMS ({expected})")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(content)
        temporary.replace(target)
        return target

    def _expected_checksum(self, client: httpx.Client, relative: str) -> str:
        if self._checksums is None:
            listing = self._download(client, "SHA256SUMS").decode("utf-8")
            self._checksums = {}
            for line in listing.splitlines():
                digest, _, name = line.partition("  ")
                if name:
                    self._checksums[unicodedata.normalize("NFC", name)] = digest
        try:
            return self._checksums[relative]
        except KeyError:
            raise DatasetError(f"{relative} is not listed in SHA256SUMS") from None

    def _download(self, client: httpx.Client, relative: str) -> bytes:
        url = (
            f"https://huggingface.co/datasets/{self._repository}/resolve/{self._revision}/"
            f"{quote(relative)}"
        )
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise DatasetError(f"cannot download {relative}: {error}") from error
        return response.content


def _title(document_id: str) -> str:
    stem = Path(document_id).stem
    return " ".join(stem.replace("%20", " ").replace("_", " ").split())
