# Thai PDF OCR and Document Chunking Pipeline

A Python-based pipeline for extracting text and table information from PDF documents using OCR, then organizing the results into JSONL files for downstream processing and Retrieval-Augmented Generation (RAG).

## Features

- **PDF Processing:** Process PDF documents from an input directory.
- **OCR:** Extract text from scanned PDF documents.
- **Table Extraction:** Preserve detected tables as structured data, including headers and rows when available.
- **Text Chunking:** Split normal text into manageable chunks with configurable chunk size and overlap.
- **Thai Text Processing:** Normalize Unicode text and use Thai word segmentation to improve chunk boundaries.
- **JSONL Export:** Save OCR results and chunks in JSON Lines format.
- **Batch Processing:** Process multiple PDF files in one run.
- **Local Execution:** Run the pipeline locally without requiring a hosted OCR API.

## Project Structure

```text
pdf_ocr_project/
├── input_pdfs/             # Input PDF documents (local only)
├── output_json/            # Generated OCR and chunking results
│   ├── all_ocr.jsonl
│   ├── all_chunks.jsonl
│   └── run_summary.json
├── pdf_ocr_chunking.py     # Main processing script
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
```

## Requirements

- Python
- Dependencies listed in `requirements.txt`
- Sufficient disk space for OCR models and generated output files

The OCR pipeline may download model files on its first run, depending on the installed libraries and model configuration.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd pdf_ocr_project
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Add PDF documents

Place the PDF files you want to process in:

```text
input_pdfs/
```

For scanned PDFs, OCR is required to extract text from page images.

### 2. Run the pipeline

```bash
python pdf_ocr_chunking.py
```

If the script supports command-line options, check its argument definitions or run:

```bash
python pdf_ocr_chunking.py --help
```

### 3. Review the output

The pipeline writes its results to `output_json/`.

| File | Description |
|---|---|
| `all_ocr.jsonl` | OCR results organized as JSON objects, typically per page |
| `all_chunks.jsonl` | Text chunks and structured table chunks for downstream retrieval |
| `run_summary.json` | Processing summary, including success and failure information when provided by the script |

The exact fields depend on the output schema implemented in the processing script.

## Chunking Strategy

Normal text is divided into chunks using a configurable chunk size and overlap ratio. Thai word segmentation is used to help avoid splitting text in the middle of words.

Detected tables are handled separately from normal text chunks so that each table can be preserved as a whole structured unit.

## Output Format

The output uses JSON Lines (JSONL), where each line contains one independent JSON object.

Example of a normal text chunk:

```json
{
  "chunk_id": "doc001_chunk_0001",
  "document_id": "doc001",
  "source_file": "example.pdf",
  "page_start": 1,
  "page_end": 1,
  "content_type": "text",
  "text": "Extracted document text..."
}
```

Example of a structured table chunk:

```json
{
  "chunk_id": "doc001_chunk_0002",
  "document_id": "doc001",
  "source_file": "example.pdf",
  "page_start": 2,
  "page_end": 2,
  "content_type": "table",
  "text": "Table content...",
  "table": {
    "title": "Example Table",
    "headers": ["Date", "Activity", "Responsible Person"],
    "rows": [
      ["01/01/2026", "Document review", "Auditor"]
    ]
  }
}
```

*The examples above are illustrative. Actual field names and table row structures may vary depending on the OCR output and implementation.*

## Potential Applications

- Digitization of scanned documents
- Thai document processing
- Structured extraction of tables from PDFs
- Preparing document data for RAG systems
- Building searchable document collections

## Limitations

- OCR accuracy depends on scan quality, resolution, document layout, and the selected OCR models.
- Complex, borderless, or merged-cell tables may require additional validation.
- OCR output should be reviewed before being used for critical decisions.
- Chunking organizes extracted content but does not itself implement a complete RAG retrieval system.

## Privacy

Input documents and generated outputs should be reviewed before committing to a public repository. Do not upload confidential documents, personal information, credentials, or sensitive OCR results.

## License

No license has been specified yet. Add a license file if you intend to permit others to use, modify, or redistribute this project.