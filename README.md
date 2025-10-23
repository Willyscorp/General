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
Environment: conda activate prof-py311-conda-env
File name: data_analysis_report.py
Output file: Saved in analysis_outputs
run: scripts\data_analysis_report.py data\raw\life_style_data.csv

# Feature 3:
Data analysis report created by ydata_profiling and sweetviz
Environment: conda activate prof-py311-conda-env
File name: data_profilling_nb
Output file: HTML file saved in analysis_outputs

# Feature 4:
Data analysis report created using langchain framework
ollama pull llama3
python scripts\langchain_pandas_agent.py data\raw\life_style_data.csv

# Feature 5
Streamlit audio-driven LangChain data analysis app
Environment: general-env 
File Name: streamlit_app_langchain.py
python -m streamlit run streamlit_app_langchain.py
Data: Can be selected as needed
Output: Will be shown on the screen
It uses Ollama llama3 which is installed locally. This need to be installed first in your environment. Run ollama pull llama3 first which takes some time
Example Audio: Describe Age or Correlation
