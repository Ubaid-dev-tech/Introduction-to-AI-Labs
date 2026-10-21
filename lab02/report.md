# Lab 02 Report - Web Scraping, Feature Engineering, and Exploratory Data Analysis

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Dr. Muhammad Awais  

## 1. Task 2.1: NumPy vs. Python Loop Timing Comparison

First, I tested the speed difference between summing 10,000,000 random numbers using `np.sum(data)` versus a standard Python `for` loop:

| Method | Sum Result | Time Taken (sec) | Speed Comparison |
| :--- | ---: | ---: | :--- |
| **NumPy (`np.sum`)** | 5,000,438.49 | 0.01300 sec | **187x faster** |
| **Python `for` loop** | 5,000,438.49 | 2.42826 sec | 1x (baseline) |

Both gave the exact same sum (`5000438.49`), but NumPy finished in `0.013` seconds compared to `2.43` seconds for the Python loop because NumPy runs the loop inside compiled C code over a continuous memory array instead of checking types on every single number in Python.

## 2. Task 2.2: Web Scraping Results

I ran the scraper on the 5 blog URLs from `techncruncher.blogspot.com` using `requests` (with a `Mozilla/5.0` User-Agent and a 1-second delay between requests) and `BeautifulSoup` (`lxml` parser). Out of the 5 links, **4 articles were scraped** and saved into `articles_raw.csv`, while the 5th link (`cj-affiliate-ultimate-guide-to.html`) was skipped by the `len(text) > 100` check because its text was inside `<div>` tags rather than `<p>` tags.

## 3. Task 2.3: NLP and Sentiment Features (`spaCy` and `TextBlob`)

I used `spaCy` (`en_core_web_sm`) to count tokens, sentences, named entities, and nouns, and `TextBlob` to get sentiment polarity (`-1` to `+1`) and subjectivity (`0` to `1`). All results were saved to `articles.csv`:

| Article Title | Title Length | Tokens | Sentences | Entities | Nouns | Verbs | Avg Sent. Length | Polarity | Subjectivity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top 10 AI Tools That Will Transform Your Content Creation in 2025 | 65 | 1,157 | 47 | 30 | 378 | 136 | 24.62 | 0.3661 | 0.6703 |
| LimeWire AI Studio Review 2023: Details, Pricing & Features | 59 | 1,711 | 65 | 115 | 454 | 169 | 26.32 | 0.2928 | 0.6494 |
| Top 10 AI Tools in 2023 That Will Make Your Life Easier | 55 | 1,380 | 52 | 45 | 383 | 166 | 26.54 | 0.2576 | 0.5588 |
| Top 10 AI Content Generator & Writer Tools in 2022 | 50 | 1,347 | 54 | 89 | 311 | 131 | 24.94 | 0.2817 | 0.6254 |

* **What I noticed:** All 4 articles have positive polarity (`0.26` to `0.37`) and high subjectivity (`0.56` to `0.67`). This matches what we expect from blog posts reviewing AI tools with affiliate links - they use lots of opinion words (`great`, `essential`, `powerful`) rather than neutral facts. The LimeWire review had the most named entities (`115`) because it listed many monthly pricing plans (`$9.99`, `$29`, `$49`, `$99`), percentages, and artist names.

## 4. Task 2.4: Exploratory Data Analysis (EDA) and TF-IDF Findings

* **Histograms (`figures/title_length_dist.png` & `figures/sentiment_polarity_dist.png`):** Title lengths are between `50` and `65` characters, and polarity scores sit between `0.258` and `0.366`.
* **Pair Plot & Correlation (`figures/pairplot_features.png`):** `num_tokens` and `num_sentences` have a very high correlation (`r = 0.98`) because all four articles have an almost identical average sentence length (around `25-26` tokens per sentence). `num_entities` also increases with article length (`r = 0.86` with tokens).
* **Top 20 TF-IDF Terms (`figures/tfidf_top20.png`):** Using `TfidfVectorizer(stop_words='english', max_features=20)`, the top words across the corpus were `content` (`1.439`), `ai` (`1.238`), `tool` (`0.831`), `month` (`0.784`), `limewire` (`0.730`), `creators` (`0.682`), `tools` (`0.610`), `plan` (`0.568`), and `features` (`0.565`). Words like `content` and `ai` are high because they appear very frequently in every post, while `limewire` and `month` get boosted by IDF because they are concentrated in specific pricing/review articles.

## 5. In-Lab Exercise Answers (Exercises 1 to 4)

1. **Exercise 1 (Adding 2 URLs and checking HTML selectors):**  
   I added two OpenAI blog posts (`https://openai.com/index/introducing-the-codex-app/` and `https://openai.com/index/chatgpt/`) to the list of 5 Blogger links. Out of 7 links, 3 failed:
   * `cj-affiliate-ultimate-guide-to.html` failed at `len(text) > 100` (`[SKIP] Too short`) because Blogger sometimes formats paragraphs using `<div>` and `<br>` tags instead of `<p>` tags, so `content_div.find_all('p')` returned almost nothing.
   * The two OpenAI links failed at `soup.find('div', class_='post-body entry-content')` (`[SKIP] No content block found`) because OpenAI's site uses `<main>`, `<article>`, and `<h1>` tags instead of Blogger's `div.post-body.entry-content` class.

2. **Exercise 2 (Adding `num_verbs` and `avg_sentence_length`):**  
   Added `num_verbs` (`131` to `169`) and `avg_sentence_length` (`24.62` to `26.54`) to the feature extractor and plotted the updated pair plot (`figures/pairplot_extended_features.png`). Verb count grows linearly with total tokens, while average sentence length stays steady around 25 words per sentence.

3. **Exercise 3 (Why a KDE over 4-5 points should not be reported as a population distribution):**  
   A KDE curve smooths out data points using Gaussian bell curves. With only 4 or 5 articles, the curve's peaks and width are just an effect of the smoothing bandwidth setting rather than a real distribution, and the tails spill over into values we never actually observed. Also, 4 articles from a single blog do not represent all articles on the web. Instead of a KDE curve, we should set `kde=False` and use a simple dot/strip plot or a table of exact values until we have at least 30-50 articles.

4. **Exercise 4 (TF-IDF with `stop_words=None` vs `stop_words='english'`):**  
   When `stop_words=None` is used (`figures/tfidf_stopwords_comparison.png`), 14 common grammar words (`and`, `the`, `to`, `for`, `it`, `of`, `you`, `is`, `in`, `with`, `that`, `can`, `as`, `your`) jump into the top 20 and replace useful domain words like `creators`, `plan`, `free`, and `platform`. This happens because `scikit-learn`'s smoothed IDF formula `log((1 + N) / (1 + df)) + 1` gives words that appear in every document a weight of `1.0` instead of `0`. Since words like `the` and `and` have huge raw counts, multiplying them by `1.0` still gives them the highest scores unless stop words are filtered out.

## 6. Home Assignment: Comparing 20 Articles from Two Blogs (`two_source_corpus.csv`)

I collected 20 articles across two blogs (`10` from `techncruncher-blog` and `10` from `openai-index-blog`) and compared their average sentiment polarity and noun density (`figures/two_source_comparison.png`):

| Blog Source | Articles ($n$) | Mean Sentiment Polarity | Mean Noun Density |
| :--- | ---: | ---: | ---: |
| **Blog A (`techncruncher-blog`)** | 10 | **0.2401** (SD = 0.1337) | **0.3517** (SD = 0.0323) |
| **Blog B (`openai-index-blog`)** | 10 | **0.0100** (SD = 0.3143) | **0.4648** (SD = 0.0472) |
| **Difference (Blog A - Blog B)** | - | **+0.2301** | **-0.1131** |

**Discussion Paragraph:**  
Across the 20 articles, TechNCruncher (`techncruncher-blog`) had a higher mean sentiment polarity (`0.2401` vs `0.0100`, difference = `+0.2301`) and a lower noun density (`0.3517` vs `0.4648`, difference = `-0.1131`) compared to OpenAI's blog (`openai-index-blog`). This makes intuitive sense because commercial review posts use lots of positive adjectives (`stunning`, `best`, `easy`) to recommend tools, whereas technical release notes use dense noun phrases (`sandbox isolation`, `token throughput`, `schema validation`) and neutral language. However, with a sample size of only 10 articles per blog, we cannot treat the polarity difference as completely conclusive (`p = 0.0542`) because small samples have wide error bars and a single outlier article (such as a cybersecurity post mentioning vulnerabilities and errors) can pull the mean down noticeably.
