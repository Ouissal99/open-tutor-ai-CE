"""Agentic SKG document loader.

OpenTutorAI-native location:
    ai/retrieval/knowledge/

This loader is intentionally lightweight for Phase 9A:
- Markdown/TXT supported by default
- PDF supported if PyMuPDF is installed
"""

from pathlib import Path
from typing import Dict, List


class AgenticSKGLoader:
    """Load course documents for the agentic Static Knowledge Grounding index."""

    SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}

    def load_directory(self, documents_dir: str | Path) -> List[Dict]:
        documents_dir = Path(documents_dir)
        documents = []

        if not documents_dir.exists():
            return documents

        for path in sorted(documents_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                continue

            documents.extend(self.load_file(path))

        return documents

    def load_file(self, path: str | Path) -> List[Dict]:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md", ".markdown"}:
            return [
                {
                    "document_id": path.stem,
                    "source_file": str(path),
                    "page": None,
                    "content": path.read_text(encoding="utf-8", errors="ignore"),
                    "metadata": {
                        "loader": "AgenticSKGLoader",
                        "file_type": suffix.lstrip("."),
                    },
                }
            ]

        if suffix == ".pdf":
            return self._load_pdf(path)

        return []

    def _load_pdf(self, path: Path) -> List[Dict]:
        try:
            import fitz  # PyMuPDF
        except Exception:
            return [
                {
                    "document_id": path.stem,
                    "source_file": str(path),
                    "page": None,
                    "content": "",
                    "metadata": {
                        "loader": "AgenticSKGLoader",
                        "file_type": "pdf",
                        "error": "PyMuPDF is not installed. Install with: pip install pymupdf",
                    },
                }
            ]

        pages = []
        doc = fitz.open(path)

        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                pages.append(
                    {
                        "document_id": path.stem,
                        "source_file": str(path),
                        "page": page_index,
                        "content": text,
                        "metadata": {
                            "loader": "AgenticSKGLoader",
                            "file_type": "pdf",
                        },
                    }
                )

        return pages
