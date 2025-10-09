# Research Navigator - Intelligent Research Paper Analyzer

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.50.0-orange)](https://streamlit.io/)

## Overview
**Research Navigator** is a Streamlit-based application that allows users to **analyze research papers intelligently**. Users can upload PDF or DOCX files, extract key phrases, identify topics, and even ask questions about the content. The app can optionally fetch related research from **Semantic Scholar** and **arXiv**.

---

## Features
- Upload PDF or DOCX research papers.
- Text cleaning and preprocessing.
- Extract key phrases using **TF-IDF**.
- Identify topics using **LDA**.
- Question answering on the uploaded content.
- Optional web retrieval from **Semantic Scholar** and **arXiv**.
- Interactive and user-friendly **Streamlit interface**.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Shreyapatidar/Research-navigator.git
cd Research-navigator
# Research-navigator
Research navigator is a chatbot that used to analyzing the research paper and asking related question to paper .

2.Create a virtual environment:
python -m venv venv
3.Activate the virtual environment:

Windows:

.\venv\Scripts\Activate.ps1



Install dependencies:

pip install -r requirements.txt

Usage

Run the Streamlit app:

streamlit run app.py


Upload your PDF or DOCX research paper.

Explore:

Key phrases (TF-IDF)

LDA topics

Ask questions about your document

Optional: Enable web retrieval for external papers.

Configuration

Semantic Scholar API Key (optional) for more accurate web retrieval.

External doc score weight to prioritize local vs external sources.

File Structure
Research-navigator/
├─ app.py                 # Main Streamlit app
├─ requirements.txt       # Python dependencies
├─ .gitignore             # Ignored files
├─ README.md              # Project documentation
└─ venv/                  # Virtual environment (ignored)

License

This project is licensed under the MIT License – see the LICENSE
 file for details.



