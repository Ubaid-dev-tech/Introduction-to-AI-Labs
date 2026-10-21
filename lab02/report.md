# Laboratory 02 Report — Web Scraping, Feature Engineering, and Exploratory Data Analysis (EDA)

**Course:** Introduction to Artificial Intelligence (AI–101L)  
**Instructor:** Dr. Muhammad Awais  
**Block:** Phase A — Environment and Data (Mapped CLO: CLO–2)  

## 1. Vectorized NumPy Operations Achieve a 187x Speedup Over Native Python Loops (Task 2.1)

To quantify why scientific computing libraries rely on compiled array operations rather than interpreted Python loops, a uniform random array of $N = 10,000,000$ double-precision floating-point values (`np.random.rand(10_000_000)`) was summed using both `np.sum(data)` and a native `for` loop.

| Implementation Method | Computed Sum | Execution Time (sec) | Relative Speed | Architectural Mechanism |
| :--- | ---: | ---: | ---: | :--- |
| **NumPy Vectorized (`np.sum`)** | 5,000,438.49 | 0.01300 | **187x faster** | Contiguous C memory buffer processed via compiled SIMD routines without per-element type checking |
| **Native Python Loop (`for val in data`)** | 5,000,438.49 | 2.42826 | 1.0x baseline | Ten million interpreted bytecode iterations with dynamic boxing/unboxing of Python `float` objects |

Both methods agree to the second decimal place (`5000438.49`), confirming numerical equivalence, while execution time drops by more than two orders of magnitude (`0.01300 sec` vs. `2.42826 sec`). Every downstream library in this pipeline (`pandas`, `spaCy`, `scikit-learn`) relies on this principle by pushing token, matrix, and vector operations into compiled C/Cython extensions.

## 2. Polite Web Scraping Acquires Four Structured Articles and Caches Raw Data (Task 2.2)

Five target blog URLs on `techncruncher.blogspot.com` were queried using `requests.get()` with a browser-mimicking `User-Agent` header (`Mozilla/5.0`), a 10-second timeout, and a polite 1-second inter-request delay (`time.sleep(1)`). The HTML responses were parsed with `BeautifulSoup(..., 'lxml')` targeting `<h3 class="post-title">` for titles and `<div class="post-body entry-content">` `<p>` tags for body text, subject to a minimum length threshold of `len(text) > 100` characters.

| URL | Scraping Status | Extracted Title | Body Length (Chars) |
| :--- | :--- | :--- | ---: |
| [top-10-ai-tools-that-will-transform.html](https://techncruncher.blogspot.com/2025/01/top-10-ai-tools-that-will-transform.html) | **Scraped** | Top 10 AI Tools That Will Transform Your Content Creation in 2025 | 6,982 |
| [limewire-ai-studio-review-2023-details.html](https://techncruncher.blogspot.com/2023/12/limewire-ai-studio-review-2023-details.html) | **Scraped** | LimeWire AI Studio Review 2023: Details, Pricing & Features | 9,556 |
| [top-10-ai-tools-in-2023-that-will-make.html](https://techncruncher.blogspot.com/2023/01/top-10-ai-tools-in-2023-that-will-make.html) | **Scraped** | Top 10 AI Tools in 2023 That Will Make Your Life Easier | 7,307 |
| [top-10-ai-content-generator-writer.html](https://techncruncher.blogspot.com/2022/11/top-10-ai-content-generator-writer.html) | **Scraped** | Top 10 AI Content Generator & Writer Tools in 2022 | 6,350 |
| [cj-affiliate-ultimate-guide-to.html](https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html) | **Skipped (`len <= 100`)** | *(Skipped by quality gate)* | $\le 100$ in `<p>` tags |

Four of the five articles passed quality control and were persisted immediately to `articles_raw.csv` (`shape: (4, 3)`) so that downstream feature engineering and exploratory data analysis remain 100% reproducible offline.

## 3. Linguistic and Sentiment Feature Engineering Quantifies Promotional Blog Style (Task 2.3)

Using `spaCy` (`en_core_web_sm`) and `TextBlob`, each unstructured article was transformed into quantitative linguistic and sentiment features (`articles.csv`, 10 columns total, plus `num_verbs` and `avg_sentence_length` from Exercise 2):

| Article Title | Title Length | Tokens | Sentences | Entities | Nouns | Verbs | Avg Sent. Length | Polarity `[-1, 1]` | Subjectivity `[0, 1]` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top 10 AI Tools That Will Transform Your Content Creation in 2025 | 65 | 1,157 | 47 | 30 | 378 | 136 | 24.62 | +0.3661 | 0.6703 |
| LimeWire AI Studio Review 2023: Details, Pricing & Features | 59 | 1,711 | 65 | 115 | 454 | 169 | 26.32 | +0.2928 | 0.6494 |
| Top 10 AI Tools in 2023 That Will Make Your Life Easier | 55 | 1,380 | 52 | 45 | 383 | 166 | 26.54 | +0.2576 | 0.5588 |
| Top 10 AI Content Generator & Writer Tools in 2022 | 50 | 1,347 | 54 | 89 | 311 | 131 | 24.94 | +0.2817 | 0.6254 |
| **Corpus Mean** | **57.25** | **1,398.75** | **54.50** | **69.75** | **381.50** | **150.50** | **25.61** | **+0.2996** | **0.6260** |

### Key Takeaways from Feature Extraction
- **Named Entity Density Reflects Pricing & Product Mentions:** The LimeWire AI Studio review (`1,711` tokens) and the 2022 AI Content Generator roundup (`1,347` tokens) contain `115` and `89` named entities respectively (`ORG`, `MONEY`, `CARDINAL`, `PERCENT`, `DATE`, `PERSON`), driven by dense subscription tier tables (`$9.99/month`, `$29/month`, `$99/month`, `50%`, `70%`) and celebrity creator references (`Deadmau5`, `Soulja Boy`, `Sean Kingston`).
- **Mildly Positive Polarity Paired with High Subjectivity:** All four articles score mildly positive (`+0.2576` to `+0.3661`, mean `+0.2996`) and markedly subjective (`0.5588` to `0.6703`, mean `0.6260`). As highlighted in the lab manual, a positive polarity combined with high subjectivity indicates persuasive opinion and affiliate marketing enthusiasm (`secret weapon`, `powerhouse`, `game-changing`) rather than objective benchmark evidence.

## 4. Exploratory Data Analysis Reveals Length Scaling and Domain Vocabulary (Task 2.4)

### A. Feature Distributions (`figures/title_length_dist.png` & `figures/sentiment_polarity_dist.png`)
- **Title Length Distribution:** Article titles span `50` to `65` characters (`mean = 57.25`, `SD = 6.34`), adhering closely to standard SEO headline guidelines (`50–65` characters) designed to prevent truncation in search engine result pages.
- **Sentiment Polarity Distribution:** Polarity values cluster tightly in the `[+0.258, +0.366]` band, reflecting a uniform promotional review tone across all four posts on the blog.

### B. Pairwise Feature Correlations (`figures/pairplot_features.png`)

| Feature | `title_length` | `num_tokens` | `num_sentences` | `num_entities` | `sentiment_polarity` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`title_length`** | 1.00 | -0.21 | -0.21 | -0.42 | +0.82 |
| **`num_tokens`** | -0.21 | 1.00 | **+0.98** | **+0.86** | -0.51 |
| **`num_sentences`** | -0.21 | **+0.98** | 1.00 | **+0.92** | -0.43 |
| **`num_entities`** | -0.42 | **+0.86** | **+0.92** | 1.00 | -0.41 |
| **`sentiment_polarity`** | +0.82 | -0.51 | -0.43 | -0.41 | 1.00 |

- **Strong Collinearity Among Volume Metrics:** `num_tokens` and `num_sentences` exhibit near-perfect linear correlation ($r = +0.98$), and both correlate strongly with `num_entities` ($r = +0.86$ and $r = +0.92$). Because the author maintains a remarkably stable sentence length (`24.6` to `26.5` tokens/sentence), total token count is a direct linear multiple of sentence count.
- **Negative Association Between Length/Entities and Polarity:** Longer articles with detailed pricing breakdowns and `Pros/Cons` caveats (such as the 2023 AI tools article, which explicitly notes `biased, offensive, or factually incorrect` outputs) have slightly lower polarity ($r = -0.51$ with `num_tokens`), whereas the shorter 2025 overview omits pricing drawbacks and scores highest in polarity (`+0.3661`).

### C. Top 20 TF–IDF Terms (`figures/tfidf_top20.png`)
Fitting `TfidfVectorizer(stop_words='english', max_features=20)` yields a `(4, 20)` document-term matrix whose column sums rank global term importance across the corpus:

| Rank | Term | TF–IDF Sum | Rank | Term | TF–IDF Sum |
| ---: | :--- | ---: | ---: | :--- | ---: |
| 1 | `content` | 1.4390 | 11 | `free` | 0.4838 |
| 2 | `ai` | 1.2378 | 12 | `creation` | 0.4830 |
| 3 | `tool` | 0.8310 | 13 | `use` | 0.4149 |
| 4 | `month` | 0.7837 | 14 | `generate` | 0.3327 |
| 5 | `limewire` | 0.7297 | 15 | `users` | 0.2952 |
| 6 | `creators` | 0.6815 | 16 | `quality` | 0.2931 |
| 7 | `tools` | 0.6104 | 17 | `creative` | 0.2636 |
| 8 | `plan` | 0.5676 | 18 | `platform` | 0.2617 |
| 9 | `features` | 0.5650 | 19 | `create` | 0.2564 |
| 10 | `makes` | 0.5365 | 20 | `generation` | 0.2418 |

- **Interpretation of TF–IDF Ranking:** Corpus-wide theme words (`content`, `ai`, `tool`, `tools`, `creators`, `creation`) top the ranking because they appear dozens of times per article, while document-specific terms like `limewire` (Rank 5, `0.7297`, appearing exclusively in Article 2) and subscription terms like `month` (Rank 4, `0.7837`) and `plan` (Rank 8, `0.5676`) achieve high scores because higher Inverse Document Frequency (IDF) boosts terms that distinguish specific articles within the collection.

## 5. In-Lab Exercises (Exercises 1–4)

### Exercise 1 — HTML Selector Diagnosis Across Seven URLs
Two additional AI blog URLs ([Introducing the Codex app](https://openai.com/index/introducing-the-codex-app/) and [Introducing ChatGPT](https://openai.com/index/chatgpt/)) were appended to the original 5 URLs (7 total). Out of the 7 pages, **4 succeeded** and **3 failed**:
1. **`https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html` (`[SKIP] Too short`):** While this Blogger page contains `<div class="post-body entry-content">`, Blogger's visual editor frequently formats body paragraphs using nested `<div>`, `<span>`, and `<br>` tags rather than `<p>` tags. Because our scraper specifically runs `paragraphs = content_div.find_all('p')`, it misses text outside `<p>` elements and fails the `len(text) > 100` check. Calling `content_div.get_text(separator=' ', strip=True)` resolves this fragile selector assumption.
2. **`https://openai.com/index/introducing-the-codex-app/` and `https://openai.com/index/chatgpt/` (`[SKIP] No content block found`):** Inspecting the HTML DOM reveals a Next.js architecture that uses semantic `<main>`, `<article>`, and `<h1>` tags with Tailwind CSS classes rather than Blogger's theme-specific `<h3 class="post-title">` and `<div class="post-body entry-content">`. Because `soup.find('div', class_='post-body entry-content')` returns `None`, both pages are skipped immediately.

### Exercise 2 — Extended Linguistic Features (`num_verbs` and `avg_sentence_length`)
Extending `extract_features` with `num_verbs` (`[t for t in doc if t.pos_ == 'VERB']`) and `avg_sentence_length` (`len(doc) / len(sentences)`) shows that verb counts (`131` to `169`) scale linearly with total tokens ($r = +0.88$), whereas `avg_sentence_length` remains invariant between `24.62` and `26.54` tokens/sentence (see `figures/pairplot_extended_features.png`).

### Exercise 3 — Why a Kernel Density Estimate (KDE) Over 4–5 Points Must Not Be Reported as a Population Distribution
A Gaussian KDE places a smooth bell curve of bandwidth $h$ over each observation and sums them to estimate a continuous probability density function $f(x)$. Computing a KDE over $n = 4$ or $n = 5$ articles is methodologically invalid as a population distribution for three reasons:
1. **Bandwidth dominance and high estimation variance:** With $n \le 5$, the apparent modes, valleys, and spread of the KDE curve are artifacts of the bandwidth heuristic (Scott's rule $h \propto n^{-1/5}$) rather than genuine population structure.
2. **Unbounded support leakage:** Gaussian kernels extend to $\pm\infty$, assigning positive density to values outside the observed range (and potentially beyond bounded intervals like $[-1, 1]$).
3. **Single-source sampling bias:** Four articles from one Blogger site represent a convenience sample of a single author, not the broader population of web articles.
- **Recommended Alternative:** Set `kde=False` and plot discrete observations directly via a **dot/strip plot (`sns.stripplot`)**, a **rug plot**, or report exact order statistics (**median and IQR**), reserving continuous KDE curves for sample sizes of at least $n \ge 30\text{–}50$.

### Exercise 4 — TF–IDF With and Without Stop-Word Removal (`figures/tfidf_stopwords_comparison.png`)
Re-running `TfidfVectorizer(stop_words=None, max_features=20)` causes **14 English function words** to enter the Top 20 (`['and', 'as', 'can', 'for', 'in', 'is', 'it', 'of', 'that', 'the', 'to', 'with', 'you', 'your']`), taking the top four spots (`and`: `1.6855`, `the`: `1.5756`, `to`: `1.4787`, `for`: `0.9786`) and pushing out 14 informative domain words (`creators`, `plan`, `makes`, `free`, `creation`, `use`, `generate`, `users`, `quality`, `creative`, `platform`, `create`, `generation`, `tool`). Why doesn't IDF eliminate words that appear in every document? Because `scikit-learn` uses **smoothed IDF with an additive constant**:
$$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
When a word appears in all $N = 4$ documents ($\text{df}(t) = 4$), $\text{IDF}(t) = \ln(5/5) + 1 = 1.0 > 0$. Consequently, ubiquitous stop words retain an IDF weight of `1.0`, allowing their massive raw term frequencies ($\text{TF}$) to dominate the L2-normalized TF–IDF vectors unless `stop_words='english'` or a `max_df` threshold (e.g., `max_df=0.80`) is applied.

## 6. Home Assignment — Two-Blog Comparison Across 20 Articles (`two_source_corpus.csv`)

Twenty articles were compiled and labeled across two contrasting sources (`n = 10` per source):
- **Blog A (`techncruncher-blog`):** 10 commercial AI tool reviews and affiliate listicles on `techncruncher.blogspot.com`.
- **Blog B (`openai-index-blog`):** 10 technical engineering and system release articles from `openai.com/index/`.

| Blog Source | Sample Size ($n$) | Mean Sentiment Polarity ($\pm$ SD) | Mean Noun Density ($\pm$ SD) | Mean Subjectivity ($\pm$ SD) | Mean Tokens |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Blog A (`techncruncher-blog`)** | 10 | **+0.2401** ($\pm 0.1337$) | **0.3517** ($\pm 0.0323$) | 0.6290 ($\pm 0.0445$) | 606.2 |
| **Blog B (`openai-index-blog`)** | 10 | **+0.0100** ($\pm 0.3143$) | **0.4648** ($\pm 0.0472$) | 0.4539 ($\pm 0.2415$) | 57.2 |
| **Difference ($\Delta = \text{Blog A} - \text{Blog B}$)** | — | **+0.2301** ($t = 2.13, p = 0.0542$) | **-0.1131** ($t = -6.26, p < 0.0001$) | **+0.1751** | +549.0 |

### Statistical Interpretation Paragraph (Meaningfulness at $n = 10$ per Group)
Across the 20-article corpus, **Blog A (`techncruncher-blog`)** exhibits a substantially higher mean sentiment polarity (`+0.2401` vs. `+0.0100`, $\Delta = +0.2301$, Welch's $t = 2.131, p = 0.0542$) and lower mean noun density (`0.3517` vs. `0.4648` nouns per token, $\Delta = -0.1131$, Welch's $t = -6.260, p < 0.0001$) than **Blog B (`openai-index-blog`)** (see `figures/two_source_comparison.png`). Qualitatively, these differences reflect genre conventions: commercial review blogs rely on evaluative adjectives (`stunning`, `invaluable`, `seamless`) and conversational verbs/pronouns (`you`, `your`) to persuade readers, whereas engineering release notes use dense technical noun phrases (`sandbox isolation`, `schema validation`, `token throughput`) and neutral or cautionary safety terminology (`vulnerability`, `incorrect`, `limitations`). However, **with a sample size of only $n = 10$ articles per blog ($N = 20$ total), statistical inference must be treated with caution**: while the large separation in noun density reaches statistical significance ($p < 0.0001$), the polarity difference sits on the boundary of the $\alpha = 0.05$ threshold ($p = 0.0542$) because small samples suffer from wide standard errors ($s/\sqrt{10}$) and high sensitivity to individual outlier articles (such as a single cybersecurity system card in Blog B with negative polarity). Furthermore, large differences in article length (`606.2` vs. `57.2` tokens) act as a confounding variable because shorter texts exhibit higher variance in both lexicon-based polarity averages and part-of-speech ratios.

## 7. Repository Structure and GitHub Upload Instructions

All required files match the exact layout specified in the Lab 02 Deliverables rubric:

```text
lab02/
├── scraping_eda.ipynb                  # Fully executed notebook with all cell outputs & plots visible
├── articles_raw.csv                    # Raw cached scrape (4 articles)
├── articles.csv                        # Scraped corpus with all engineered linguistic & sentiment features
├── two_source_corpus.csv               # 20-article two-blog corpus for the Home Assignment
├── report.md                           # Complete Lab 02 analytical report
├── README.md                           # Repository overview for GitHub landing page
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Standard Python/Jupyter gitignore
└── figures/
    ├── title_length_dist.png           # Fig 1: Histogram & KDE of article title lengths
    ├── sentiment_polarity_dist.png     # Fig 2: Histogram & KDE of article sentiment polarity
    ├── pairplot_features.png           # Fig 3: Pair plot of derived numerical features
    ├── tfidf_top20.png                 # Fig 4: Top 20 TF-IDF terms (stop_words='english')
    ├── pairplot_extended_features.png  # Fig 5 (Ex 2): Pair plot with num_verbs & avg_sentence_length
    ├── tfidf_stopwords_comparison.png  # Fig 6 (Ex 4): TF-IDF comparison with vs. without stop words
    └── two_source_comparison.png       # Fig 7 (Home Assignment): Two-blog grouped bar chart
```

### Quick Commands to Push `lab02/` to Your GitHub Repository
1. Extract `lab02_github_repo.zip` on your computer.
2. Open a terminal inside the extracted folder and run:

```bash
git init
git add lab02/
git commit -m "Complete Lab 02: Web Scraping, Feature Engineering, and EDA"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
