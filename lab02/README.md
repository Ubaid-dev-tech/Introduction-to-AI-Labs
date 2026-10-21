# AI-101L Laboratory 02 — Web Scraping, Feature Engineering, and Exploratory Data Analysis (EDA)

This repository contains the complete implementation, datasets, exported figures, and analytical report for **Laboratory 02** (*Introduction to Artificial Intelligence, AI–101L*, Instructor: Dr. Muhammad Awais).

## Deliverables Included

| File | Description |
| :--- | :--- |
| `lab02/scraping_eda.ipynb` | Executed Jupyter Notebook containing Tasks 2.1–2.4, In-Lab Exercises 1–4, and the 20-article Home Assignment with all outputs and plots embedded |
| `lab02/articles_raw.csv` | Cached raw web scrape of `techncruncher.blogspot.com` articles |
| `lab02/articles.csv` | Engineered dataset with `spaCy` linguistic counts (`num_tokens`, `num_sentences`, `num_entities`, `num_nouns`, `title_length`) and `TextBlob` sentiment metrics (`sentiment_polarity`, `sentiment_subjectivity`) |
| `lab02/two_source_corpus.csv` | Home Assignment corpus of 20 articles across two blogs (`techncruncher-blog` vs. `openai-index-blog`) with polarity and noun density metrics |
| `lab02/figures/*.png` | High-resolution PNG exports of all distribution plots, pair plots, TF–IDF bar charts, and the two-source comparison chart |
| `lab02/report.md` | Full written report covering the NumPy vectorization benchmark, EDA findings, TF–IDF interpretation, Exercises 1–4, and Home Assignment statistical analysis |

## Environment Setup

```bash
conda activate ai-lab
pip install -r lab02/requirements.txt
python -m spacy download en_core_web_sm
```
