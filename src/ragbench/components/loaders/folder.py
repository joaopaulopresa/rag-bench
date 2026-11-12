"""Bring your own documents: a folder of text files with an optional gold question file."""

from __future__ import annotations

import json
from pathlib import Path

from ragbench.core.errors import DatasetError
from ragbench.core.registries import register_loader
from ragbench.core.types import Corpus, Document, Question, Span

TEXT_SUFFIXES = (".txt", ".md")
QUESTIONS_FILE = "questions.json"


def read_document(path: Path) -> str:
    """Return the text of a ``.txt``, ``.md`` or ``.pdf`` file.

    Raises:
        DatasetError: If the file cannot be read, or it is a PDF and the ``pdf`` extra is not
            installed.
    """
    try:
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader  # noqa: PLC0415
            except ImportError as error:
                raise DatasetError(
                    f"{path.name}: reading PDF needs the extra: pip install 'ragbench[pdf]'"
                ) from error
            return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetError(f"cannot read {path}: {error}") from error


@register_loader("folder")
class FolderLoader:
    """Loads every document below a folder, plus reviewed gold questions if present.

    Questions come from ``questions.json`` in the folder, in the format written by
    ``ragbench questions generate``::

        {"tests": [{"query": "...", "reviewed": true,
                    "snippets": [{"file_path": "a.txt", "span": [10, 42], "answer": "..."}]}]}

    Only questions marked ``"reviewed": true`` are loaded: generated questions do not count as
    gold until a person has checked them.

    Args:
        resources: Unused; present so the class fits the factory signature.
        path: The folder.
        name: Corpus name; defaults to the folder name.
        include_unreviewed: Also load questions that nobody reviewed (for trial runs only).
    """

    def __init__(
        self,
        resources: object,
        path: str,
        name: str | None = None,
        include_unreviewed: bool = False,
    ) -> None:
        del resources
        self._path = Path(path)
        self._name = name or self._path.name
        self._include_unreviewed = include_unreviewed

    def load(self) -> Corpus:
        """Read the folder.

        Raises:
            DatasetError: If the folder does not exist, holds no documents, or a question's
                span does not match its answer text.
        """
        if not self._path.is_dir():
            raise DatasetError(f"{self._path} is not a folder")
        documents = {}
        for file in sorted(self._path.rglob("*")):
            if file.is_file() and file.suffix.lower() in (*TEXT_SUFFIXES, ".pdf"):
                document_id = file.relative_to(self._path).as_posix()
                documents[document_id] = Document(
                    id=document_id,
                    text=read_document(file),
                    title=file.stem.replace("_", " ").replace("-", " "),
                    source=self._name,
                )
        if not documents:
            raise DatasetError(f"{self._path} holds no .txt, .md or .pdf files")
        return Corpus(
            name=self._name,
            documents=tuple(documents.values()),
            questions=tuple(self._questions(documents)),
        )

    def _questions(self, documents: dict[str, Document]) -> list[Question]:
        file = self._path / QUESTIONS_FILE
        if not file.exists():
            return []
        try:
            tests = json.loads(file.read_text(encoding="utf-8"))["tests"]
        except (ValueError, KeyError, TypeError) as error:
            raise DatasetError(f"{file} is not a question file: {error}") from error
        questions = []
        for position, test in enumerate(tests):
            if not (test.get("reviewed") or self._include_unreviewed):
                continue
            snippets = test["snippets"]
            document_id = snippets[0]["file_path"]
            if document_id not in documents:
                raise DatasetError(f"{file}: question {position} points to unknown {document_id!r}")
            text = documents[document_id].text
            spans = []
            for snippet in snippets:
                start, end = snippet["span"]
                if snippet["file_path"] != document_id or text[start:end] != snippet["answer"]:
                    raise DatasetError(
                        f"{file}: question {position}: span [{start}, {end}) does not match "
                        "its answer text"
                    )
                spans.append(Span(start, end))
            questions.append(
                Question(
                    id=f"{self._name}-{position:04d}",
                    text=test["query"],
                    document_id=document_id,
                    spans=tuple(sorted(spans)),
                    source=self._name,
                )
            )
        return questions
