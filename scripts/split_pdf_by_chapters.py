#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from pypdf import PdfReader, PdfWriter

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


def safe_filename(name: str, max_len: int = 120) -> str:
    """Make safe filenames from chapter titles."""
    name = re.sub(r"[\\/:*?\"<>|]+", " ", name).strip()
    name = re.sub(r"\s+", " ", name)
    return (name or "chapter")[:max_len]


@dataclass
class Chapter:
    title: Optional[str]
    start_page: int
    end_page: Optional[int] = None


# Replace your extract_top_level_bookmarks with this:

from typing import List, Tuple

def _iter_outline_items(reader):
    """
    Return a list of (level, title, page_index) for all outline items.
    Items that don't resolve to a page are skipped.
    Supports both newer `outline` and older `outlines`.
    """
    tree = getattr(reader, "outline", None)
    if not tree:
        tree = getattr(reader, "outlines", None)
    if not tree:
        return []

    items = []

    def walk(nodes, level=0):
        for node in nodes:
            if isinstance(node, list):
                walk(node, level + 1)
            else:
                try:
                    title = (getattr(node, "title", None) or str(node) or "").strip()
                    dest = getattr(node, "destination", node)
                    # Some nodes don't resolve to a page -> catch and skip
                    try:
                        page_idx = reader.get_destination_page_number(dest)
                    except Exception:
                        page_idx = None
                    if page_idx is not None:
                        items.append((level, title, page_idx))
                except Exception:
                    # Ignore weird/unsupported nodes
                    pass

    walk(tree, 0)
    # Filter out any Nones defensively, then sort by page index
    items = [(lvl, title, pg) for (lvl, title, pg) in items if pg is not None]
    items.sort(key=lambda t: t[2])
    return items

def extract_top_level_bookmarks(reader, start_level: int = 0) -> List[Tuple[str, int]]:
    all_items = _iter_outline_items(reader)
    if not all_items:
        return []
    # Prefer the requested level
    filtered = [(title, page) for (lvl, title, page) in all_items if lvl == start_level]
    # If that yields nothing or only a single TOC item, try the next level automatically
    if len(filtered) <= 1:
        next_level = start_level + 1
        filtered = [(title, page) for (lvl, title, page) in all_items if lvl == next_level]
    # Dedup by page
    seen = {}
    for title, pg in filtered:
        seen[pg] = title
    return [(title, pg) for pg, title in sorted(seen.items(), key=lambda x: x[0])]


def detect_headings_regex(pdf_path: str, heading_pattern: str, min_font_size: Optional[float]):
    """Detect chapter starts using regex (optionally with font size threshold)."""
    pat = re.compile(heading_pattern, flags=re.IGNORECASE | re.MULTILINE)
    starts = []

    if not HAS_PDFPLUMBER:
        # fallback: plain text
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if pat.search(text):
                starts.append((pat.search(text).group(0).strip(), i))
        return starts

    # pdfplumber for font-aware detection
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            m = pat.search(text)
            if m and min_font_size is None:
                starts.append((m.group(0).strip(), i))
            elif min_font_size:
                for ch in page.chars:
                    if float(ch.get("size", 0)) >= min_font_size:
                        if pat.search(ch.get("text", "")):
                            starts.append((ch["text"], i))
                            break
    return starts


def write_chapters(reader: PdfReader, chapters: List[Chapter], outdir: str, prefix: str = "chapter"):
    os.makedirs(outdir, exist_ok=True)
    for i, ch in enumerate(chapters, start=1):
        writer = PdfWriter()
        for p in range(ch.start_page, ch.end_page + 1):
            writer.add_page(reader.pages[p])
        title = ch.title or f"{prefix}_{i}"
        filename = f"{i:02d}_{safe_filename(title)}.pdf"
        path = Path(outdir) / filename
        with open(path, "wb") as f:
            writer.write(f)
        print(f"✅ Wrote {path} ({ch.end_page - ch.start_page + 1} pages)")


def main():
    parser = argparse.ArgumentParser(description="Split a PDF into chapters using bookmarks or regex.")
    parser.add_argument("--input", "-i", required=True, help="Path to the PDF")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory")
    parser.add_argument("--strategy", choices=["bookmarks", "regex"], default="bookmarks")
    parser.add_argument("--start-level", type=int, default=0, help="Bookmark level to use (default: 0)")
    parser.add_argument("--heading-pattern", default=r"^(Chapter)\s+\d+\b", help="Regex to detect headings")
    parser.add_argument("--min-font-size", type=float, default=None, help="Min font size for heading detection")
    parser.add_argument("--prefix", default="chapter", help="Prefix for filenames")
    args = parser.parse_args()

    reader = PdfReader(args.input)
    num_pages = len(reader.pages)
    chapters = []

    if args.strategy == "bookmarks":
        starts = extract_top_level_bookmarks(reader, start_level=args.start_level)
    else:
        starts = detect_headings_regex(args.input, args.heading_pattern, args.min_font_size)

    if not starts:
        print("⚠️ No chapters detected!")
        sys.exit(1)

    # Build chapter ranges
    starts = sorted(starts, key=lambda x: x[1])
    for idx, (title, pg) in enumerate(starts):
        end = starts[idx + 1][1] - 1 if idx < len(starts) - 1 else num_pages - 1
        chapters.append(Chapter(title=title, start_page=pg, end_page=end))

    write_chapters(reader, chapters, args.outdir, prefix=args.prefix)


if __name__ == "__main__":
    main()
