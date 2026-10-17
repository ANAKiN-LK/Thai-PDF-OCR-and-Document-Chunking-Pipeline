from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
import unicodedata
from pythainlp.tokenize import word_tokenize

class TableHTMLParser(HTMLParser):
    """Small HTML table parser; keeps table cells in row/column order."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self.current_cell is not None:
            value = re.sub(r"\s+", " ", "".join(self.current_cell)).strip()
            if self.current_row is not None:
                self.current_row.append(value)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if any(cell for cell in self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None


def html_table_to_schema(html: str) -> tuple[list[str], list[dict[str, str]]]:
    parser = TableHTMLParser()
    try:
        parser.feed(html or "")
    except Exception:
        pass
    matrix = parser.rows
    if not matrix:
        return [], []
    width = max(len(row) for row in matrix)
    matrix = [row + [""] * (width - len(row)) for row in matrix]
    headers = [cell or f"column_{i + 1}" for i, cell in enumerate(matrix[0])]
    # If the first row has no meaningful header cells, still keep it as headers
    # so the original table remains structured and no data is discarded.
    rows: list[dict[str, str]] = []
    for row in matrix[1:]:
        rows.append({headers[i]: row[i] for i in range(width)})
    return headers, rows


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def get_result_json(result: Any) -> dict[str, Any]:
    """PP-StructureV3 Result objects expose a .json dictionary."""
    data = getattr(result, "json", None)
    if callable(data):
        data = data()
    if isinstance(data, dict):
        # Some versions wrap the actual result in a top-level 'res'.
        if isinstance(data.get("res"), dict):
            return data["res"]
        return data
    if isinstance(result, dict):
        if isinstance(result.get("res"), dict):
            return result["res"]
        return result
    return {}


def bbox_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        flat = np.asarray(value).reshape(-1).tolist()
        if len(flat) >= 4:
            return [float(v) if isinstance(v, (float, np.floating)) else int(v) for v in flat[:4]]
    except Exception:
        return None
    return None


def block_type(label: str) -> str:
    label = (label or "").lower().replace("_", " ").strip()
    if "table" in label:
        return "table"
    if any(word in label for word in ("figure", "image", "chart", "graph", "picture")):
        return "image"
    if "formula" in label or "equation" in label:
        return "equation"
    return "text"


def extract_elements(result_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize PP-StructureV3 layout blocks into the requested elements schema."""
    parsing = result_json.get("parsing_res_list") or result_json.get("layout_parsing_result") or []
    elements: list[dict[str, Any]] = []

    for block in parsing:
        if not isinstance(block, dict):
            continue
        label = str(block.get("block_label") or block.get("label") or block.get("type") or "text")
        kind = block_type(label)
        content = block.get("block_content")
        if content is None:
            content = block.get("content") or block.get("text") or ""
        bbox = bbox_list(block.get("block_bbox") or block.get("bbox") or block.get("coordinate"))
        item: dict[str, Any] = {"element_id": len(elements) + 1, "type": kind, "bbox": bbox}

        if kind == "table":
            html = str(content or "")
            headers, rows = html_table_to_schema(html)
            title = block.get("table_title") or block.get("title")
            item.update({"title": title, "headers": headers, "rows": rows, "html": html})
            if not headers and not rows:
                # Keep all recognized table content even if HTML structure is absent.
                item["content"] = re.sub(r"<[^>]+>", " ", html).strip()
        elif kind == "image":
            item["description"] = str(content).strip() if content else None
        else:
            item["content"] = str(content or "").strip()
        elements.append(item)

    # Compatibility fallback: use PP-StructureV3's markdown output when its
    # version doesn't expose parsing_res_list. Markdown tables are still retained
    # as a single table element rather than being flattened into prose chunks.
    if not elements:
        markdown = result_json.get("markdown_texts") or result_json.get("markdown") or ""
        if isinstance(markdown, dict):
            markdown = markdown.get("markdown_texts", "")
        if isinstance(markdown, list):
            markdown = "\n".join(map(str, markdown))
        lines = str(markdown).splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if "|" in line and i + 1 < len(lines) and re.search(r"\|?\s*:?-{3,}", lines[i + 1]):
                table_lines = [line, lines[i + 1]]
                i += 2
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i].strip())
                    i += 1
                matrix = [[c.strip() for c in row.strip().strip("|").split("|")] for row in table_lines if not re.search(r"^\|?\s*:?-{3,}", row)]
                headers = matrix[0] if matrix else []
                rows = [{headers[j] if j < len(headers) else f"column_{j+1}": val for j, val in enumerate(row)} for row in matrix[1:]]
                elements.append({"element_id": len(elements) + 1, "type": "table", "title": None, "headers": headers, "rows": rows, "html": None, "content": "\n".join(table_lines), "bbox": None})
            else:
                elements.append({"element_id": len(elements) + 1, "type": "text", "content": line, "bbox": None})
                i += 1
    return elements


def ocr_pdf(pdf_path: Path, pipeline: Any, dpi: int = 250) -> list[dict[str, Any]]:
    """Run local PaddleOCR PP-StructureV3 page-by-page, retaining layout/tables."""
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as pdf:
        total = len(pdf)
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csRGB, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            predictions = pipeline.predict(input=image)
            result_json = get_result_json(predictions[0]) if predictions else {}
            elements = extract_elements(result_json)
            pages.append({
                "document": pdf_path.name,
                "document_id": pdf_path.stem,
                "source_file": pdf_path.name,
                "page": page_number,
                "elements": elements,
            })
            count_tables = sum(e.get("type") == "table" for e in elements)
            print(f"  Page {page_number}/{total}: {len(elements)} elements, {count_tables} table(s)")
    return pages



def split_text(text: str, chunk_size: int, overlap_ratio: float) -> list[str]:
    """Split normal text using Thai word boundaries; tables never enter this function."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")

    words = word_tokenize(text, engine="newmm")
    overlap = int(chunk_size * overlap_ratio)
    chunks = []
    current_words = []
    current_length = 0
    i = 0

    while i < len(words):
        word = words[i]

        if current_words and current_length + len(word) > chunk_size:
            chunks.append("".join(current_words).strip())

            # Keep approximately the requested character overlap.
            kept_words = []
            kept_length = 0
            for previous_word in reversed(current_words):
                if kept_length + len(previous_word) > overlap:
                    break
                kept_words.insert(0, previous_word)
                kept_length += len(previous_word)

            current_words = kept_words
            current_length = kept_length

            # Prevent an infinite loop if overlap keeps the current word.
            if not current_words:
                continue

        current_words.append(word)
        current_length += len(word)
        i += 1

    if current_words:
        chunks.append("".join(current_words).strip())

    return [chunk for chunk in chunks if chunk]


def table_to_text(element: dict[str, Any]) -> str:
    """Readable table rendering for retrieval; structured data is also preserved."""
    title = element.get("title")
    headers = element.get("headers") or []
    rows = element.get("rows") or []
    pieces = [str(title).strip()] if title else []
    if headers:
        pieces.append(" | ".join(map(str, headers)))
        for row in rows:
            if isinstance(row, dict):
                pieces.append(" | ".join(str(row.get(h, "")) for h in headers))
            else:
                pieces.append(" | ".join(map(str, row)))
    elif element.get("content"):
        pieces.append(str(element["content"]))
    elif element.get("html"):
        pieces.append(re.sub(r"<[^>]+>", " ", str(element["html"])))
    return "\n".join(p for p in pieces if p)


def chunk_document(
    pages: list[dict[str, Any]], document_id: str, source_file: str,
    domain: str = "unknown", chunk_size: int = 1000, overlap_ratio: float = 0.30,
) -> list[dict[str, Any]]:
    """Chunk text normally, but emit each detected table as exactly one whole chunk."""
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    def add_chunk(kind: str, text: str, page: int, element: dict[str, Any] | None = None) -> None:
        nonlocal chunk_index
        if not text.strip():
            return
        chunk_index += 1
        record: dict[str, Any] = {
            "chunk_id": f"{document_id}_chunk_{chunk_index:04d}",
            "chunk_index": chunk_index,
            "document_id": document_id,
            "source_file": source_file,
            "domain": domain,
            "page_start": page,
            "page_end": page,
            "content_type": kind,
            "text": text,
        }
        if element is not None and kind == "table": 
            record["table"] = {
                "title": element.get("title"),
                "headers": element.get("headers", []),
                "rows": element.get("rows", []),
                "html": element.get("html"),
                "bbox": element.get("bbox"),
            }
        chunks.append(record)

    for page_record in pages:
        page_number = int(page_record["page"])
        # Join adjacent normal-text elements on the same page before chunking.
        pending_text: list[str] = []
        def flush_text() -> None:
            nonlocal pending_text
            if pending_text:
                combined = "\n".join(pending_text).strip()
                for part in split_text(combined, chunk_size, overlap_ratio):
                    add_chunk("text", part, page_number)
                pending_text = []

        for element in page_record.get("elements", []):
            kind = element.get("type", "text")
            if kind == "table":
                flush_text()
                # Important: a table is not split, even when larger than chunk_size.
                add_chunk("table", table_to_text(element), page_number, element)
            elif kind == "image":
                flush_text()
                description = element.get("description") or ""
                add_chunk("image", str(description) or "[ตรวจพบรูปภาพในเอกสาร]", page_number, element)
            else:
                content = str(element.get("content") or "").strip()
                if content:
                    pending_text.append(content)
        flush_text()
    return chunks


def load_domain_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("domain_map.json must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(jsonable(data), file, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local PDF layout OCR + table-preserving JSONL chunking")
    parser.add_argument("--input-dir", default="input_pdfs")
    parser.add_argument("--output-dir", default="output_json")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap-ratio", type=float, default=0.30)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--domain", default="unknown")
    parser.add_argument("--domain-map", default=None)
    parser.add_argument("--expected-files", type=int, default=20)
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input folder not found: {input_dir.resolve()}", file=sys.stderr)
        return 1
    pdf_files = sorted((p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"), key=lambda p: p.name.lower())
    if not pdf_files:
        print(f"ERROR: no PDF files found in {input_dir.resolve()}", file=sys.stderr)
        return 1
    if args.chunk_size < 1 or not 0 <= args.overlap_ratio < 1:
        print("ERROR: chunk-size must be positive and overlap-ratio must be in [0, 1)", file=sys.stderr)
        return 1
    if len(pdf_files) != args.expected_files:
        print(f"WARNING: found {len(pdf_files)} PDF(s); expected {args.expected_files}; processing all found PDFs.")
    try:
        domain_map = load_domain_map(Path(args.domain_map) if args.domain_map else None)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR reading domain map: {exc}", file=sys.stderr)
        return 1

    try:
        from paddleocr import PPStructureV3
    except ImportError as exc:
        print('Missing PaddleOCR document parser. Install with: python -m pip install "paddleocr[doc-parser]"', file=sys.stderr)
        return 1
    try:
        # Initialize once; reuse the local models across all PDFs.
      pipeline = PPStructureV3(
        text_recognition_model_name="th_PP-OCRv5_mobile_rec",
        text_detection_model_name="PP-OCRv5_server_det",  # ตัวตรวจจับแม่นกว่า mobile
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        device=args.device,
        )
        
    except Exception as exc:
        print(f"ERROR initializing PP-StructureV3: {exc}", file=sys.stderr)
        return 1

    all_ocr: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{index}/{len(pdf_files)}] {pdf_path.name}")
        try:
            page_records = ocr_pdf(pdf_path, pipeline=pipeline, dpi=args.dpi)
            all_ocr.extend(page_records)
            chunks = chunk_document(page_records, pdf_path.stem, pdf_path.name,
                                    domain_map.get(pdf_path.stem, args.domain), args.chunk_size, args.overlap_ratio)
            all_chunks.extend(chunks)
            print(f"  Chunks created: {len(chunks)} (tables kept whole: {sum(c['content_type'] == 'table' for c in chunks)})")
        except Exception as exc:
            failed.append({"source_file": pdf_path.name, "error": str(exc)})
            print(f"  FAILED: {exc}", file=sys.stderr)

    write_jsonl(output_dir / "all_ocr.jsonl", all_ocr)
    write_jsonl(output_dir / "all_chunks.jsonl", all_chunks)
    write_json(output_dir / "run_summary.json", {
        "pdf_files_found": len(pdf_files), "pdf_files_succeeded": len(pdf_files) - len(failed),
        "pdf_files_failed": len(failed), "ocr_page_records": len(all_ocr),
        "total_chunks": len(all_chunks), "table_chunks": sum(c["content_type"] == "table" for c in all_chunks),
        "chunk_size_characters": args.chunk_size, "overlap_ratio": args.overlap_ratio,
        "domain_default": args.domain, "ocr_engine": "PaddleOCR PP-StructureV3",
        "failed_files": failed,
    })
    print("\n========== DONE ==========")
    print(f"PDF found: {len(pdf_files)} | succeeded: {len(pdf_files) - len(failed)} | failed: {len(failed)}")
    print(f"OCR JSONL: {(output_dir / 'all_ocr.jsonl').resolve()}")
    print(f"Chunks JSONL: {(output_dir / 'all_chunks.jsonl').resolve()}")
    print(f"Summary: {(output_dir / 'run_summary.json').resolve()}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
