# Cleanup Notes

## What was changed
- Renamed and simplified project to top-level folders:
  - `notebooks/`
  - `scripts/`
  - `datasets/`
  - `submissions/`
  - `materials/`

## Main source code identified
- `notebooks/*.ipynb`
- `scripts/*.py`

## Files that should not be committed
- Course readings/textbook PDFs in `materials/`
- Slides, generated images, recordings, and zip archives in `materials/`
- Temporary/system artifacts (checkpoint notebooks, local DB) in `materials/`
- Raw datasets in `datasets/`

## Duplicate and temp artifacts found
- Duplicate by hash: `Group RAG Module 4.docx` (multiple copies)
- Placeholder doc files: `New DOCX 文档.docx`

## Deletion policy used
- No files were permanently deleted during cleanup.
- Everything was moved into explicit folders for easy final review.
