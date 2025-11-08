"""PDF helpers for CUDAway tooling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict

from pypdf import PdfReader


@dataclass
class SymbolData:
    """Represents occurrences of an API symbol within a PDF."""

    name: str
    pages: list[int]
    hits: int = 0

    def add_hit(self, page: int) -> None:
        if not self.pages or self.pages[-1] != page:
            self.pages.append(page)
        self.hits += 1


def extract_prefixed_symbols(pdf_path: Path | str, prefix: str) -> Dict[str, SymbolData]:
    """Scan the PDF and collect symbols starting with *prefix*.

    Parameters
    ----------
    pdf_path: Path | str
        PDF file to scan.
    prefix: str
        Identifier prefix to search for (e.g. ``"cuda"`` or ``"hip"``).

    Returns
    -------
    Dict[str, SymbolData]
        Mapping from symbol name to aggregated metadata.
    """

    prefix_pattern = re.compile(rf"\b({re.escape(prefix)}[A-Za-z0-9_]+)\b")
    reader = PdfReader(str(pdf_path))
    matches: Dict[str, SymbolData] = {}
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for match in prefix_pattern.finditer(text):
            symbol = match.group(1)
            data = matches.get(symbol)
            if data is None:
                data = SymbolData(name=symbol, pages=[], hits=0)
                matches[symbol] = data
            data.add_hit(index)
    return matches
