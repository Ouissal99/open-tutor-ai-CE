"""Build the local Agentic SKG index.

Usage:
    python scripts/build_agentic_skg_index.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.retrieval.knowledge.agentic_skg_indexer import AgenticSKGIndexer


def main():
    indexer = AgenticSKGIndexer()
    metadata = indexer.build()

    print("Agentic SKG index built.")
    print("Documents:", metadata["document_count"])
    print("Chunks:", metadata["chunk_count"])
    print("Chunks path:", metadata["chunks_path"])


if __name__ == "__main__":
    main()
