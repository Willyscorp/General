# General

Author: William

This project was created using a Copier template.

## Project Structure
- `data/` — raw, interim, and processed datasets
- `notebooks/` — Jupyter notebooks
- `src/` — source code
- `tests/` — test scripts
- `scripts/` — utility scripts

## About
This file have all the general tool which I have built and will be pushed in the github going forward.
Brief details of the tool is given below

# Feature 1:
Split the pdf by chapters. 
Environment: general-env 
File name: split_pdf_by_chapters.py

Split by bookmarks (if PDF has TOC/bookmarks):
python scripts/split_pdf_by_chapters.py --input "data/raw/Making sense of data.pdf" --outdir data/processed --strategy bookmarks
python scripts/split_pdf_by_chapters.py --input "data/raw/Ayn Rand_ Atlas Shrugged_ the Fountainhead -- Rand, Ayn -- New York, 2009.pdf" --outdir data/processed --strategy bookmarks --start-level 1
python scripts/split_pdf_by_chapters.py --input "data/raw/Ayn Rand_ Atlas Shrugged_ the Fountainhead -- Rand, Ayn -- New York, 2009.pdf" --outdir data/processed --strategy regex --heading-pattern "^(?:Pattern|Chapter|Part)\\s+(?:\\d+|[IVXLCDM]+)\\b" --min-font-size 14

# Feature 2:
Customized data analysis
Environment: general-env 
File name: data_analysis_report.py
Output file: Saved in analysis_outputs

# Feature 3:
Data analysis report created by ydata_profiling and sweetviz
Environment: general-env 
File name: data_profilling_nb
Output file: HTML file saved in analysis_outputs


