# AI-101L مکمل وائیوا تیاری گائیڈ (Labs 1 to 8 Complete Viva Preparation Guide)

**کورس:** Introduction to Artificial Intelligence (AI-101L)  
**انسٹرکٹر:** Dr. Muhammad Awais  
**ایویلیوایشن مارکس:** 10 / 20 Lab Marks  

---

## تمام 8 لیبز کا خلاصہ (Quick Master Summary Table)

| لیب نمبر | لیب کا عنوان (Topic) | استعمال شدہ لائبریریز (Libraries) | اہم فنکشنز اور فارمولے (Key Functions & Formulas) | آپ کے پورٹ فولیو کا نتیجہ (Exact Result in Your Code) |
| :--- | :--- | :--- | :--- | :--- |
| **Lab 01** | Environment Setup, Git & GitHub | `conda`, `pandas`, `git` | `df.head()`, `df.info()`, `df.describe()`, `git checkout -b`, `git merge` | Titanic: `891x15` (`deck` میں 688 اور `age` میں 177 missing values)؛ `parch` کا `std/mean = 2.112` سب سے زیادہ۔ |
| **Lab 02** | Web Scraping, NLP & EDA | `numpy`, `requests`, `bs4`, `spacy`, `textblob`, `sklearn` | `np.sum()`, `BeautifulSoup(..., 'lxml')`, `nlp(text)`, `TextBlob(text).sentiment`, `TfidfVectorizer` | NumPy لوپ سے **187 گنا تیز** (`0.013s` بمقابلہ `2.43s`)؛ 4 بلاگز سکریپ ہوئے؛ ٹاپ TF-IDF الفاظ: `content`, `ai`, `tool`۔ |
| **Lab 03** | Text & Image Features | `sklearn`, `PIL (Pillow)`, `numpy` | `CountVectorizer(ngram_range=(1,2))`, `img.convert('L')`, `img.resize((64,64))`, `arr.flatten()` | BoW میٹرکس: `(3, 11)` (`57.6%` sparsity)؛ RGB `100x100x3` (`30,000`) سے Grayscale `100x100` (`10,000` فیچرز، `3x` کمی)۔ |
| **Lab 04** | Linear Regression (Simple & Multi) | `sklearn.linear_model`, `pandas`, `numpy` | $Y = mx + c$ اور $Y = \sum m_i x_i + c$؛ `LinearRegression().fit()`, `r2_score` | **Simple LR (`Salary`):** $m = 9423.82$, $c = 24380.20$, $R^2 = 0.9024$۔ **Multi LR (`House_Price`):** 7 فیچرز، $R^2 = 0.9984$۔ |
| **Lab 05** | Logistic Regression & Classification | `sklearn.linear_model`, `sklearn.svm`, `pickle` | $z = \sum m_i x_i + c$, Sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$؛ `LogisticRegression`, `LabelEncoder`, `confusion_matrix`, `roc_curve` | **Diabetes:** Accuracy = `75.97%`, AUC = `0.8290` (`TN=80, FP=19, FN=18, TP=37`)۔ **Titanic (`sample-data.csv`):** SVC Accuracy = `80%`, RF = `83.24%`۔ |
| **Lab 06** | FastAPI Model Deployment | `fastapi`, `uvicorn`, `pydantic`, `pickle` | `@app.get("/")`, `@app.get("/health")`, `@app.post("/predict")`, `CORSMiddleware`, `fetch()` | `Mymdoel.pkl` کو `main.py` اور `index.html` سے جوڑا۔ Patient 1 (`Age 21, Glucose 112`) $\rightarrow$ `Not Diabetic (0.148)`۔ |
| **Lab 07** | CNN Architecture & Layers | `numpy`, `matplotlib`, `seaborn` | Conv2D $O = \frac{I-K+2P}{S}+1$, ReLU $\max(0,x)$, MaxPool $2\times2$, Softmax $\frac{e^{z_i}}{\sum e^{z_j}}$ | `5x5` امیج $\rightarrow$ `3x3` Conv (`[[8,-4,-2],[4,0,0],[2,1,-2]]`) $\rightarrow$ ReLU $\rightarrow$ `2x2` Pool (`[8,0,4,1]`) $\rightarrow$ **Cat (`81%`)**۔ |
| **Lab 08** | Digit Recognition on MNIST | `tensorflow.keras`, `sklearn.metrics`, `PIL` | `Conv2D(32,(3,3))`, `MaxPooling2D((2,2))`, `Dropout(0.3)`, `sparse_categorical_crossentropy` | `42,000` ٹریننگ امیجز (`28x28x1`)؛ `225,034` پیرامیٹرز؛ 10 Epochs کے بعد **Train Acc = `99.50%`**, **Val Acc = `98.93%`**؛ `images.png` $\rightarrow$ **`5`**۔ |

---

## Lab 1: Environment Setup, Git and GitHub (`lab01/`)

### 1. بنیادی تصور (Core Concept)
* **Anaconda اور Virtual Environment (`conda`):** ڈیٹا سائنس میں مختلف پراجیکٹس کو مختلف لائبریری ورژنز کی ضرورت ہوتی ہے۔ ہم نے `conda create -n ai-lab python=3.11 -y` کمانڈ سے ایک الگ تھلگ (Isolated) ماحول بنایا تاکہ ایک پراجیکٹ کی لائبریریاں دوسرے پراجیکٹ کو خراب نہ کریں۔
* **Pandas Inspection (`df.head()`, `df.info()`, `df.describe()`):** کسی بھی ڈیٹا سیٹ پر ماڈل لگانے سے پہلے اس کا معائنہ کیا جاتا ہے:
  * `df.shape`: قطاروں اور کالموں کی تعداد بتاتا ہے (Titanic میں `891 rows x 15 columns`)۔
  * `df.head()`: پہلی 5 قطاریں دکھاتا ہے تاکہ پتہ چلے ڈیٹا صحیح لوڈ ہوا ہے۔
  * `df.info()`: ہر کالم کی ڈیٹا ٹائپ اور `Non-Null Count` بتاتا ہے جس سے **Missing Values** فوری پکڑی جاتی ہیں (`deck` میں `891 - 203 = 688` خالی اقدار، `age` میں `891 - 714 = 177` خالی اقدار، اور `embarked`/`embark_town` میں `2` خالی اقدار)۔
  * `df.describe()`: عددی کالموں کا `mean`, `std`, `min`, `25%`, `50% (median)`, `75%`, اور `max` بتاتا ہے۔
* **Git اور GitHub:** Git ایک Version Control System ہے جو کوڈ کی ہر تبدیلی کو **Commit** (چیک پوائنٹ) کی صورت میں محفوظ کرتا ہے۔ `.gitignore` فائل میں ہم `.ipynb_checkpoints/`, `__pycache__/`, `.venv/`, اور `*.pkl` لکھتے ہیں تاکہ فالتو یا بھاری فائلیں Git میں اپلوڈ نہ ہوں۔

### 2. متوقع وائیوا سوالات اور جوابات (Viva Q&A - Lab 1)
* **سوال 1: `df.info()` اور `df.describe()` میں کیا فرق ہے؟**  
  **جواب:** `df.info()` تمام کالموں کے نام، ان کی ڈیٹا ٹائپس (`int64`, `float64`, `object`) اور غیر خالی (Non-Null) اندراجات کی تعداد بتاتا ہے، جبکہ `df.describe()` صرف عددی (Numeric) کالموں کا شماریاتی خلاصہ (`mean`, `std`, `min`, `max`, percentiles) دکھاتا ہے۔
* **سوال 2: Exercise 3 میں ہم نے `std / mean` (Coefficient of Variation) کیوں نکالا اور Feature Scaling کیوں ضروری ہے؟**  
  **جواب:** `std / mean` بتاتا ہے کہ کسی کالم میں اس کی اوسط کے لحاظ سے کتنا پھیلاؤ ہے۔ Titanic میں `parch` (`2.112`)، `sibsp` (`2.108`) اور `fare` (`1.543`, جہاں `max = 512.33`) کا پھیلاؤ بہت زیادہ ہے۔ اگر ہم بغیر Scaling کے ڈیٹا ماڈل کو دیں تو KNN، SVM یا Gradient Descent والے ماڈلز بڑی ویلیوز والے فیچر (`fare`) کو زیادہ اہمیت دیں گے اور چھوٹے فیچر (`pclass`) کو نظر انداز کر دیں گے۔
* **سوال 3: `git branch`, `git checkout -b`, اور `git merge` کا کیا کام ہے؟**  
  **جواب:** `git checkout -b branch_name` ایک نئی برانچ بنا کر اس میں سوئچ کرتا ہے تاکہ ہم اصل `main` برانچ کو چھیڑے بغیر نیا تجربہ کر سکیں۔ کام مکمل ہونے پر `git checkout main` کر کے `git merge branch_name` سے وہ کام واپس `main` میں شامل کر لیا جاتا ہے۔

---

## Lab 2: Web Scraping, NLP Feature Engineering & EDA (`lab02/`)

### 1. بنیادی تصور (Core Concept)
* **NumPy Vectorization بمقابلہ Python Loop:** 1 کروڑ (`10,000,000`) نمبروں کو جمع کرنے میں Python `for` loop نے **`2.43` سیکنڈ** لگائے جبکہ `np.sum(data)` نے صرف **`0.013` سیکنڈ** لگائے (**187 گنا تیز**)، کیونکہ NumPy اپنا لوپ کمپائل شدہ C کوڈ میں ایک مسلسل میموری بلاک (Contiguous C Array) پر چلاتا ہے اور ہر نمبر پر Python کی طرح ٹائپ چیکنگ نہیں کرتا۔
* **Web Scraping (`requests` + `BeautifulSoup`):**
  * `requests.get(url, headers={"User-Agent": "Mozilla/5.0 ..."}, timeout=10)` ویب پیج کا خام HTML لاتا ہے۔ `User-Agent` اس لیے بھیجتے ہیں تاکہ ویب سرور ہمیں عام براؤزر سمجھے اور بلاک نہ کرے۔
  * `BeautifulSoup(res.content, 'lxml')` اس HTML کو پارس کر کے ٹائٹل (`soup.find('h3', class_='post-title')`) اور مضمون کا متن (`soup.find('div', class_='post-body entry-content')`) نکالتا ہے۔
* **NLP Feature Engineering (`spaCy` اور `TextBlob`):**
  * **spaCy (`en_core_web_sm`):** متن کو ٹوکنز (`len(doc)`)، جملوں (`len(list(doc.sents))`)، Named Entities (`len(doc.ents)` جیسے نام، کمپنیاں، قیمتیں) اور Parts of Speech (`t.pos_ == 'NOUN'` یا `'VERB'`) میں توڑتا ہے۔
  * **TextBlob:** دو سکور دیتا ہے: **Polarity** (`-1.0` منفی سے `+1.0` مثبت تک) اور **Subjectivity** (`0.0` حقائق/Objective سے `1.0` ذاتی رائے/Opinion تک)۔
* **TF-IDF (`TfidfVectorizer`):**
  * **TF (Term Frequency):** ایک لفظ اس مضمون میں کتنی بار آیا۔
  * **IDF (Inverse Document Frequency):** اگر کوئی لفظ ہر مضمون میں آتا ہے (جیسے `the`, `is`) تو اس کی اہمیت کم کر دیتا ہے، اور جو لفظ کسی خاص مضمون کا خاص موضوع ہو (جیسے `limewire`, `creators`) اس کا سکور بڑھا دیتا ہے۔

### 2. متوقع وائیوا سوالات اور جوابات (Viva Q&A - Lab 2)
* **سوال 1: 7 میں سے 3 لنکس سکریپ کیوں نہیں ہوئے؟**  
  **جواب:** پانچواں Blogger لنک (`cj-affiliate`) اس لیے فیل ہوا کیونکہ اس میں پیراگراف `<p>` ٹیگز کے بجائے `<div>` اور `<br>` ٹیگز میں لکھے گئے تھے، اس لیے `find_all('p')` کو 100 حروف سے کم متن ملا۔ جبکہ OpenAI کے دونوں لنکس اس لیے فیل ہوئے کیونکہ وہ Blogger پر نہیں بنے اور ان کے HTML میں `<div class="post-body entry-content">` موجود نہیں تھا۔
* **سوال 2: اگر `TfidfVectorizer` میں `stop_words=None` رکھیں تو کیا نقصان ہوتا ہے؟**  
  **جواب:** اگر `stop_words=None` رکھیں تو عام گرامر کے الفاظ (`the`, `and`, `to`, `of`, `in`) ٹاپ 20 الفاظ پر قبضہ کر لیتے ہیں کیونکہ ان کی تعدد (TF) بہت زیادہ ہوتی ہے اور `scikit-learn` کے ہموار فارمولے $\text{IDF} = \ln(\frac{1+N}{1+df}) + 1$ میں ہر دستاویز میں آنے والے لفظ کا IDF صفر ہونے کے بجائے `1.0` رہتا ہے۔

---

## Lab 3: Transforming Text and Image Data into Machine-Understandable Format (`lab03/`)

### 1. بنیادی تصور (Core Concept)
مشین لرننگ ماڈلز صرف نمبروں کے یکساں لمبائی والے ویکٹرز (Numerical Feature Vectors) پر ریاضیاتی عمل (Dot Product) کر سکتے ہیں، وہ براہِ راست الفاظ یا تصویر کو نہیں سمجھ سکتے۔
* **Part A - Bag-of-Words (`CountVectorizer`):**
  * تین جملوں سے 11 منفرد الفاظ کی لغت (Vocabulary) بنی اور `(3, 11)` کا میٹرکس بنا۔
  * `CountVectorizer` بائی ڈیفالٹ دو کام خود کرتا ہے: `lowercase=True` (بڑے حروف کو چھوٹا کرنا) اور `token_pattern='(?u)\\b\\w\\w+\\b'` (صرف 2 یا زیادہ حروف والے الفاظ رکھنا، جس کی وجہ سے ایک حرف والا لفظ `'a'` نکل جاتا ہے)۔
  * **Sparsity:** میٹرکس کے 33 خانوں میں سے 19 خانے صفر (`0`) ہیں یعنی `57.6%` sparsity۔ اسی لیے میموری بچانے کے لیے `scipy.sparse.csr_matrix` استعمال ہوتا ہے۔
  * **نقصان (Information Lost):** Bag-of-Words الفاظ کی ترتیب (Word Order) اور گرامر کھو دیتا ہے۔ `ngram_range=(1, 2)` استعمال کرنے سے یک لفظی (Unigrams) کے ساتھ دو الفاظ کے جوڑے (Bigrams جیسے `'the sun'`, `'sunny day'`) بھی شامل ہو جاتے ہیں اور فیچرز 11 سے بڑھ کر 24 ہو جاتے ہیں۔
* **Part B - Image Features (Resize, Grayscale & Flatten):**
  * **RGB سے Grayscale (`convert('L')`):** ایک رنگین تصویر `100x100x3` میں `30,000` نمبرز ہوتے ہیں (ہر پکسل کے لیے R, G, B)۔ Grayscale کرنے سے یہ صرف چمک (Luminance) کا 1 چینل `100x100 = 10,000` رہ جاتا ہے، یعنی فیچرز **3 گنا کم** ہو جاتے ہیں۔
  * **Resize (`resize((64, 64))`):** مختلف سائز کی تصاویر کو ایک ہی سائز (`64x64`) پر لانا ضروری ہے تاکہ ہر تصویر کا فلیٹ ویکٹر بالکل برابر لمبائی (`4,096` فیچرز) کا بنے۔
  * **Flattening (`flatten()`):** 2D میٹرکس `(H, W)` کو 1D ویکٹر `(H * W,)` میں بدلتا ہے تاکہ Logistic Regression یا SVM ہر پکسل $x_j$ کو اس کے وزن $w_j$ سے ضرب دے سکے۔
  * **Curse of Dimensionality:** ہوم اسائنمنٹ میں 20 تصاویر اور `4,096` فیچرز ($D \gg N$) ہیں؛ اتنے زیادہ فیچرز اور اتنے کم سیمپلز پر ماڈل ٹریننگ ڈیٹا کے شور (Noise) کو رٹ لیتا ہے (Overfitting)۔

---

## Lab 4: Linear Regression - Simple & Multiple (`lab04/`)

### 1. بنیادی تصور اور سلائیڈز کے فارمولے (`Regression Linear vs Logistic.pptx`)
Linear Regression ایک **Supervised Learning** الگورتھم ہے جو کسی مسلسل عددی قیمت (**Continuous Numerical Output** جیسے تنخواہ یا گھر کی قیمت) کی پیشگوئی کرتا ہے۔

#### A. Simple Linear Regression (`Salary_dataset.csv` - 30 rows)
* **فارمولا:** $$Y = m x + c$$
  جہاں $x$ ان پٹ فیچر (`YearsExperience`)، $m$ ڈھلوان (`model.coef_`)، اور $c$ انٹرسیپٹ (`model.intercept_`) ہے۔
* **ٹریننگ کے بعد سیکھے گئے پیرامیٹرز (`80/20` split, `random_state=42`):**
  * **Slope ($m$):** `9423.81532303` (ہر 1 سال کے اضافی تجربے سے تنخواہ میں تقریباً `$9,423.82` اضافہ ہوتا ہے)
  * **Intercept ($c$):** `24380.201479473726` (صفر تجربے پر بنیادی تنخواہ)
* **سر کی سلائیڈ نمبر 2 کا حل (Solved Slide 2 Table):**
  1. جب $x = 9.7$: $Y = 9423.81532303 \times 9.7 + 24380.201479473726 = \mathbf{115,791.21}$
  2. جب $x = 5.0$: $Y = 9423.81532303 \times 5.0 + 24380.201479473726 = \mathbf{71,499.28}$
* **Evaluation Metrics:** `MAE = 6,286.45`, `MSE = 49,830,096.86`, `RMSE = 7,059.04`, **`R² Score = 0.9024` (`90.24%`)**۔

#### B. Multiple Linear Regression (`house_price_regression_dataset.csv` - 1,000 rows)
* **فارمولا:** $$Y = m_1 x_1 + m_2 x_2 + m_3 x_3 + m_4 x_4 + m_5 x_5 + m_6 x_6 + m_7 x_7 + c$$
* **7 ان پٹ فیچرز:** `Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`
* **Weights ($m_1 \dots m_7$):** `[199.5132451, 10225.20442601, 8208.43477213, 993.53717098, 14885.38441545, 5146.14838675, 115.06859497]`
* **Intercept ($c$):** `-2010567.4201016556`
* **سر کی سلائیڈ نمبر 3 کا حل (Solved Slide 3 Table):**
  1. پہلا گھر `[4012, 3, 1, 2016, 2.098092, 1, 5]` $\rightarrow$ Output $Y = \mathbf{868,687.10}$
  2. دوسرا گھر `[2310, 3, 1, 1988, 1.369622, 1, 4]` $\rightarrow$ Output $Y = \mathbf{490,337.89}$
* **Evaluation Metrics:** `MAE = 8,174.58`, `RMSE = 10,071.48`, **`R² Score = 0.9984` (`99.84%`)**۔

### 2. متوقع وائیوا سوالات اور جوابات (Viva Q&A - Lab 4)
* **سوال 1: MAE، MSE، RMSE اور $R^2$ Score میں کیا فرق ہے؟**  
  **جواب:**
  * **MAE (Mean Absolute Error):** اصل اور پیشگوئی شدہ قیمت کے فرق کی اوسط ($\frac{1}{n}\sum |y - \hat{y}|$)۔
  * **MSE (Mean Squared Error):** غلطیوں کے مربع (Square) کی اوسط ($\frac{1}{n}\sum (y - \hat{y})^2$)، جو بڑی غلطیوں کو زیادہ سزا دیتا ہے۔
  * **RMSE:** MSE کا جزر ($\sqrt{\text{MSE}}$) تاکہ یونٹ واپس اصل ٹارگٹ (ڈالر/روپے) جیسا ہو جائے۔
  * **$R^2$ Score (Coefficient of Determination):** بتاتا ہے کہ ٹارگٹ $y$ کی کتنی فیصد تبدیلی ہمارے ماڈل نے سمجھی ہے (`1.0` کا مطلب 100% بہترین فٹ ہے)۔

---

## Lab 5: Logistic Regression & Classification Systems (`lab05/`)

### 1. بنیادی تصور اور سلائیڈز کا حل (`Regression Linear vs Logistic.pptx` Slides 4 & 5)
جب ہمارا آؤٹ پٹ مسلسل نمبر نہ ہو بلکہ **کلاس/کیٹیگری** ہو (جیسے ذیابیطس ہے `1` یا نہیں `0`، مسافر بچ گیا `Yes` یا نہیں `No`)، تو ہم **Logistic Regression** استعمال کرتے ہیں۔
* **مرحلہ 1 (Linear Logit $z$):** پہلے لینیئر مساوات سے $z$ نکالتے ہیں:
  $$z = m_1 x_1 + m_2 x_2 + m_3 x_3 + m_4 x_4 + m_5 x_5 + m_6 x_6 + c$$
* **مرحلہ 2 (Sigmoid Activation Function):** چونکہ $z$ منفی یا مثبت کوئی بھی نمبر ہو سکتا ہے، اس لیے اسے `0` اور `1` کے درمیان Probability میں بدلنے کے لیے **Sigmoid فنکشن** میں سے گزارتے ہیں:
  $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  اگر $\sigma(z) \ge 0.5$ ہو تو کلاس **`1` (Diabetic)** اور اگر $< 0.5$ ہو تو کلاس **`0` (Non-Diabetic)** ہوتی ہے۔

#### A. Diabetes Logistic Regression (`diabetes.csv` - 768 rows, 6 features)
* **6 فیچرز کی ترتیب (`model.feature_names_in_`):** `['Age', 'BMI', 'BloodPressure', 'Insulin', 'Glucose', 'DiabetesPedigreeFunction']`
* **Weights ($m_1 \dots m_6$):** `[0.04559683, 0.09394991, -0.00964773, 0.00442035, 0.03010364, 0.48521205]`
* **Intercept ($c$):** `-9.2358725`
* **سر کی سلائیڈ نمبر 4 اور 5 کا حل (Solved Slide 4 & 5 Table):**
  1. **پہلا مریض `[43, 34.0, 58.0, 190.0, 98, 0.430]`:**
     * $z = \mathbf{-0.641816}$
     * $\sigma(z) = \frac{1}{1 + e^{0.641816}} = \mathbf{0.3448}$ (`34.48%` $< 0.5 \implies$ **Non-Diabetic `0`**)
  2. **دوسرا مریض `[49, 30.4, 88.0, 169.5, 165, 0.302]`:**
     * $z = \mathbf{+0.868333}$
     * $\sigma(z) = \frac{1}{1 + e^{-0.868333}} = \mathbf{0.7044}$ (`70.44%` $\ge 0.5 \implies$ **Diabetic `1`**)
  3. **نوٹ بک کا ٹیسٹ مریض `[21, 35.7, 75.0, 102.5, 112, 0.148]`:**
     * $z = \mathbf{-1.751402} \implies \sigma(z) = \mathbf{0.1479}$ (`14.79%` $\implies$ **Non-Diabetic `0`**)
* **Test Results:** Accuracy = **`75.97%`**، Confusion Matrix = `[[80, 19], [18, 37]]`، ROC AUC = **`0.8290`**۔

#### B. Titanic Survival Prediction System (`Titanic Passenger Survival Prediction System...ipynb`)
* **Machine Learning Cycle کے 4 مراحل:**
  1. **Training Phase:** ڈیٹا کو سمجھنا، `LabelEncoder` سے الفاظ (`First/Second/Third`, `Male/Female`) کو نمبروں میں بدلنا، `train_test_split` کرنا، ماڈل (`svm.SVC` / `LogisticRegression` / `RandomForest`) کو ٹرین کرنا اور `pickle.dump` سے `.pkl` فائل میں محفوظ کرنا۔
  2. **Testing Phase:** سیو کیے ہوئے ماڈل کو `pickle.load` سے لوڈ کرنا، ٹیسٹ ڈیٹا پر `model.predict()` چلانا اور `accuracy_score` نکالنا (`sample-data.csv` کے 20 ٹیسٹ مسافروں پر **`80%` accuracy**)۔
  3. **Application Phase:** نئے ان دیکھے مسافر (Unseen User Input) کا ڈیٹا لینا، محفوظ شدہ `*_encoder.pkl` سے اسے بالکل اسی طرح Encode کرنا اور نتیجہ دکھانا۔
  4. **Feedback Phase:** ماہرین اور صارفین کے فیڈ بیک سے ماڈل کو مزید بہتر بنانا۔

### 2. متوقع وائیوا سوالات اور جوابات (Viva Q&A - Lab 5)
* **سوال 1: Linear Regression اور Logistic Regression میں سب سے بڑا فرق کیا ہے؟**  
  **جواب:** Linear Regression مسلسل عددی قیمت (Continuous value، جیسے تنخواہ) نکالتا ہے اور اس کی سیدھی لائن $(-\infty, +\infty)$ تک جاتی ہے اور اس میں MSE loss استعمال ہوتا ہے۔ جبکہ Logistic Regression کلاسیفیکیشن (Classification، جیسے `0` یا `1`) کے لیے استعمال ہوتا ہے، یہ لینیئر آؤٹ پٹ $z$ پر **Sigmoid فنکشن** لگا کر اسے `0` اور `1` کے درمیان Probability بنا دیتا ہے اور اس میں Log-Loss (Binary Cross-Entropy) استعمال ہوتا ہے۔
* **سوال 2: ٹریننگ اور ٹیسٹنگ میں `fit_transform()` اور `transform()` کا کیا اصول ہے؟**  
  **جواب:** ہم صرف ٹریننگ ڈیٹا پر `fit_transform()` چلاتے ہیں اور ٹیسٹ یا نئے یوزر ڈیٹا پر صرف `transform()` چلاتے ہیں تاکہ ٹیسٹ ڈیٹا کی معلومات ٹریننگ میں لیک نہ ہوں (**Data Leakage** سے بچاؤ)۔
* **سوال 3: Confusion Matrix کے چاروں خانے کیا بتاتے ہیں؟**  
  **جواب:**
  * **True Positive (`TP = 37`):** مریض کو شوگر تھی اور ماڈل نے بھی صحیح پکڑا کہ شوگر ہے۔
  * **True Negative (`TN = 80`):** مریض صحت مند تھا اور ماڈل نے بھی صحت مند بتایا۔
  * **False Positive (`FP = 19`, Type I Error):** مریض صحت مند تھا لیکن ماڈل نے غلطی سے شوگر بتا دی۔
  * **False Negative (`FN = 18`, Type II Error):** مریض کو شوگر تھی لیکن ماڈل نے غلطی سے صحت مند کہہ دیا (طبی لحاظ سے یہ سب سے خطرناک غلطی ہے)۔

---

## Lab 6: Model Deployment with FastAPI (`lab06/`)

### 1. بنیادی تصور (Core Concept)
ٹرین کیے گئے ماڈل (`Mymdoel.pkl`) کو عام صارفین تک پہنچانے کے لیے ہم نے **FastAPI** بیک اینڈ (`main.py`) اور **HTML/JavaScript** فرنٹ اینڈ (`index.html`) بنایا:
* **`Mymdoel.pkl`:** Lab 5 میں ٹرین کیا گیا Diabetes Logistic Regression ماڈل جسے `pickle.load()` سے لوڈ کیا گیا۔
* **`PredictRequest(BaseModel)` (Pydantic Schema):** چیک کرتا ہے کہ فرنٹ اینڈ سے آنے والے چھ کے چھ فیچرز (`Glucose`, `BloodPressure`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) درست عددی (`float`) فارمیٹ میں ہیں۔
* **`FEATURES = list(model.feature_names_in_)`:** جب فرنٹ اینڈ سے JSON آتا ہے تو `df = pd.DataFrame([payload])[FEATURES]` کالموں کی ترتیب کو بالکل ٹریننگ والی ترتیب (`['Age', 'BMI', 'BloodPressure', 'Insulin', 'Glucose', 'DiabetesPedigreeFunction']`) پر لاتا ہے تاکہ غلط کالم غلط وزن سے ضرب نہ کھائے۔
* **CORS (`CORSMiddleware`):** براؤزر کی سیکیورٹی پالیسی کے تحت `index.html` کو `http://localhost:8000/predict` پر کال کرنے کی اجازت دیتا ہے۔
* **تین Endpoints:**
  1. `GET /`: سرور چلنے کا پیغام دیتا ہے۔
  2. `GET /health`: `{"status": "ok"}` لوٹاتا ہے۔
  3. `POST /predict`: `model.predict(df)` اور `model.predict_proba(df)` چلا کر `predicted_label`, `prediction` (`"Diabetic"` یا `"Not Diabetic"`) اور `probability` واپس بھیجتا ہے۔

### 2. متوقع وائیوا سوالات اور جوابات (Viva Q&A - Lab 6)
* **سوال 1: سرور کو کس کمانڈ سے چلاتے ہیں اور Swagger UI کہاں کھلتا ہے؟**  
  **جواب:** `python -m uvicorn main:app --reload --port 8000` سے چلاتے ہیں، اور `http://localhost:8000/docs` کھولنے پر FastAPI کا خودکار انٹرایکٹو Swagger UI کھل جاتا ہے جہاں سے ہم بغیر فرنٹ اینڈ کے بھی `/predict` ٹیسٹ کر سکتے ہیں۔
* **سوال 2: `/predict` کے لیے `GET` کے بجائے `POST` میتھڈ کیوں استعمال کیا گیا؟**  
  **جواب:** کیونکہ ہم کلائنٹ سے سرور کو 6 فیچرز پر مشتمل ایک مکمل JSON Body (`req.body`) بھیج رہے ہیں جس پر ماڈل نے حساب لگا کر نتیجہ دینا ہے۔ ڈیٹا پے لوڈ بھیجنے کے لیے ہمیشہ `POST` استعمال ہوتا ہے۔

---

## Lab 7: CNN Architecture & Layers (`lab07/` & `CNN Architecture.pptx`)

### 1. بنیادی تصور اور سلائیڈز کی مکمل ریاضی (`CNN Architecture.pptx`)
روایتی ایم ایل میں تصویر کو Flatten کرنے سے اس کی 2D ساخت ختم ہو جاتی ہے۔ **Convolutional Neural Network (CNN)** تصویر کی 2D ساخت کو برقرار رکھتا ہے اور 6 مراحل میں کام کرتا ہے:

1. **Stage 1 - Input Image (`5x5` Grayscale):** کمپیوٹر تصویر کو پکسل نمبروں (`0` = کالا، `255` = سفید) کے گرڈ کے طور پر دیکھتا ہے۔ رنگین تصویر میں 3 گرڈ (R, G, B) ہوتے ہیں۔
2. **Stage 2 - Convolution Layer (`3x3` Kernel, Stride=1, Padding=0):**
   * ایک چھوٹا فلٹر (Kernel $3\times3$) تصویر کے اوپر سلائیڈ کرتا ہے اور ہر جگہ آمنے سامنے والے نمبروں کو ضرب دے کر جمع کرتا ہے (Dot Product)۔
   * پہلا خانہ: $3(1) + 1(0) + 0(-1) + 2(1) + 0(0) + 1(-1) + 4(1) + 1(0) + 0(-1) = \mathbf{8}$
   * آؤٹ پٹ سائز کا فارمولا: $$O = \frac{I - K + 2P}{S} + 1 = \frac{5 - 3 + 0}{1} + 1 = \mathbf{3 \times 3}$$
   * حاصل شدہ Feature Map: $\begin{bmatrix} 8 & -4 & -2 \\ 4 & 0 & 0 \\ 2 & 1 & -2 \end{bmatrix}$
3. **Stage 3 - ReLU Activation ($f(x) = \max(0, x)$):**
   * تمام منفی نمبروں کو `0` کر دیتا ہے اور مثبت نمبروں کو ویسا ہی رکھتا ہے: $\begin{bmatrix} 8 & 0 & 0 \\ 4 & 0 & 0 \\ 2 & 1 & 0 \end{bmatrix}$۔ اس سے نیٹ ورک میں **Non-linearity** آتی ہے تاکہ وہ پیچیدہ پیٹرن سیکھ سکے۔
4. **Stage 4 - Max Pooling Layer ($2\times2$ Window):**
   * ہر $2\times2$ کھڑکی میں سے صرف سب سے بڑا نمبر رکھتا ہے (پہلی کھڑکی: $\max(8, 0, 4, 0) = 8$)۔ نتیجہ: $\begin{bmatrix} 8 & 0 \\ 4 & 1 \end{bmatrix}$۔
   * **فائدہ:** سائز چھوٹا ہو جاتا ہے، حساب کتاب (Computation) کم ہو جاتا ہے، اور اگر تصویر میں چیز تھوڑی سی دائیں بائیں ہل بھی جائے تب بھی جواب یکساں رہتا ہے (**Translation Invariance**)۔
5. **Stage 5 - Flatten + Fully Connected (`Dense`) Layer:**
   * $2\times2$ گرڈ کو سیدھی 1D لسٹ `[8, 0, 4, 1]` میں بدلتے ہیں اور ہر کلاس کے اوزان (Weights) سے ضرب دے کر جمع کرتے ہیں (Logits $z$):
     * **Cat:** $8(0.3) + 0(0.1) + 4(0.2) + 1(0.1) = \mathbf{3.3}$
     * **Dog:** $8(0.1) + 0(0.2) + 4(0.1) + 1(0.3) = \mathbf{1.5}$
     * **Bird:** $8(0.0) + 0(0.4) + 4(0.1) + 1(0.2) = \mathbf{0.6}$
6. **Stage 6 - Softmax Output ($\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$):**
   * جب 2 سے زیادہ کلاسز ہوں (Multi-class) تو آخری لیئر میں Sigmoid کے بجائے **Softmax** لگاتے ہیں جو Logits کو ایسی Probabilities میں بدلتا ہے جن کا مجموعہ `100%` ہو:
     * **Cat = `81%`**، **Dog = `13%`**، **Bird = `5%`** $\implies$ سب سے زیادہ امکان **Cat (`81%`)** کا ہے!

---

## Lab 8: Digit Recognition on MNIST Dataset (`lab08/`)

### 1. بنیادی تصور اور کوڈ کی مکمل تفصیل (`Digits Recognition.ipynb`)
اس لیب میں ہم نے ہاتھ سے لکھے ہوئے ہندسوں (`0` سے `9` تک) کو پہچاننے کے لیے `TensorFlow / Keras` میں مکمل CNN ٹرین کیا:
* **ڈیٹا سیٹ (`train.csv` اور `test.csv`):**
  * `train.csv` کا شیپ `(42000, 785)` ہے: پہلا کالم `label` (`0` سے `9` تک اصلی ہندسہ) ہے اور باقی `784` کالمز (`pixel0` سے `pixel783`) $28 \times 28$ تصویر کے پکسلز ہیں۔
  * `test.csv` کا شیپ `(28000, 784)` ہے جس میں صرف پکسلز ہیں۔
* **پری پروسیسنگ کے 3 اہم قدم:**
  1. **Normalization (`X = X.astype("float32") / 255.0`):** پکسل ویلیوز کو `0-255` سے `0.0-1.0` کے درمیان لاتے ہیں تاکہ نیورل نیٹ ورک کی ٹریننگ تیز اور مستحکم ہو۔
  2. **Reshape (`X = X.reshape(-1, 28, 28, 1)`):** `784` پکسلز کی سیدھی قطار کو واپس `28x28x1` (Height=28, Width=28, Channel=1 برائے Grayscale) میں بدلتے ہیں کیونکہ `Conv2D` کو 2D تصویر چاہیے ہوتی ہے۔ (`-1` کا مطلب ہے کہ Python خود سیمپلز کی تعداد `42,000` رکھ لے)۔
  3. **Stratified Split:** `train_test_split(..., test_size=0.2, stratify=y)` سے `33,600` ٹریننگ اور `8,400` ویلیڈیشن تصاویر الگ کیں۔ `stratify=y` یقینی بناتا ہے کہ `0` سے `9` تک ہر ہندسے کا تناسب ٹریننگ اور ویلیڈیشن دونوں میں برابر رہے۔
* **CNN ماڈل کا ڈھانچہ اور Parameters کا حساب (`Total Params = 225,034`):**
  * **Block 1:**
    * `Conv2D(32, (3, 3))` $\rightarrow$ آؤٹ پٹ شیپ `(None, 26, 26, 32)` کیونکہ $28 - 3 + 1 = 26$۔ پیرامیٹرز: $(3 \times 3 \times 1 + 1\text{ bias}) \times 32 = \mathbf{320}$۔
    * `Activation("relu")` $\rightarrow$ منفی نمبروں کو صفر کرنے کے لیے۔
    * `MaxPooling2D((2, 2))` $\rightarrow$ سائز آدھا کر کے `(None, 13, 13, 32)` کر دیتا ہے۔
  * **Block 2:**
    * `Conv2D(64, (3, 3))` $\rightarrow$ آؤٹ پٹ شیپ `(None, 11, 11, 64)` کیونکہ $13 - 3 + 1 = 11$۔ پیرامیٹرز: $(3 \times 3 \times 32 + 1\text{ bias}) \times 64 = \mathbf{18,496}$۔
    * `Activation("relu")` + `MaxPooling2D((2, 2))` $\rightarrow$ آؤٹ پٹ شیپ `(None, 5, 5, 64)` کیونکہ $\lfloor 11 / 2 \rfloor = 5$۔
  * **Classifier Head:**
    * `Flatten()` $\rightarrow$ $5 \times 5 \times 64 = \mathbf{1,600}$ لمبائی کا 1D ویکٹر بنا دیتا ہے۔
    * `Dense(128)` + `ReLU` $\rightarrow$ پیرامیٹرز: $(1600 + 1) \times 128 = \mathbf{204,928}$۔
    * `Dropout(0.3)` $\rightarrow$ ٹریننگ کے دوران ہر قدم پر 30% نیورونز کو عارضی طور پر بند کر دیتا ہے تاکہ ماڈل رٹا نہ لگائے (**Overfitting روکنے کے لیے**)۔
    * `Dense(10)` + `Softmax` $\rightarrow$ 10 ہندسوں (`0-9`) کے لیے 10 احتمالات (Probabilities) نکالتا ہے۔ پیرامیٹرز: $(128 + 1) \times 10 = \mathbf{1,290}$۔
* **Compile اور Training:**
  * `loss="sparse_categorical_crossentropy"`: اس لیے استعمال کیا کیونکہ ہمارے لیبلز `y` سیدھے انٹیجرز (`0, 1, ..., 9`) ہیں؛ اگر لیبلز One-Hot Encoded ہوتے تو `categorical_crossentropy` لگاتے۔
  * 10 Epochs کے بعد **Train Accuracy = `99.50%`** اور **Validation Accuracy = `98.93%`** حاصل ہوئی!
* **ٹیسٹ اور کسٹم امیج (`images.png`) کی پیشگوئی:**
  * `X_test[5]` پر `np.argmax(probs)` سے ماڈل نے **`7`** پہچانا۔
  * `images.png` (سفید بیک گراؤنڈ پر کالا **`5`**) کو پہلے Grayscale کیا، پھر **`ImageOps.invert()`** سے الٹایا (کیونکہ MNIST میں کالی سکرین پر سفید ہندسہ ہوتا ہے!)، پھر `28x28` پر Resize اور `/ 255.0` کر کے ماڈل کو دیا تو ماڈل نے **`99.82%` اعتماد کے ساتھ `5`** پہچان لیا!

### 2. متوقع وائیوا سوالات اور جوابات (Viva Q&A - Lab 8)
* **سوال 1: اگر ہم `images.png` کو Invert (`255 - img`) کیے بغیر سیدھا ماڈل کو دیں تو کیا ہوگا؟**  
  **جواب:** MNIST ڈیٹا سیٹ میں پس منظر (Background) کالا (`0`) اور ہندسہ سفید (`255`) ہوتا ہے، جبکہ `images.png` میں پس منظر سفید (`255`) اور ہندسہ کالا (`0`) ہے۔ اس لیے اگر Invert نہ کریں تو ماڈل کو پورا پس منظر روشن نظر آئے گا اور پیشگوئی غلط ہو جائے گی۔
* **سوال 2: `Dropout(0.3)` کا کیا فائدہ ہے اور کیا یہ ٹیسٹنگ (`model.predict`) کے وقت بھی چلتا ہے؟**  
  **جواب:** `Dropout(0.3)` صرف ٹریننگ کے وقت 30% نیورونز کو بے ترتیب طور پر بند کرتا ہے تاکہ کوئی ایک نیورون پورے ڈیٹا کو رٹ نہ سکے (Overfitting کم ہوتی ہے)۔ ٹیسٹنگ یا `model.predict()` کے وقت Dropout خود بخود بند ہو جاتا ہے اور پورے 100% نیورونز کام کرتے ہیں۔
* **سوال 3: `np.argmax(probs)` کیا کرتا ہے؟**  
  **جواب:** `model.predict()` ہمیں 10 ہندسوں (`0` سے `9`) کی 10 Probabilities کی لسٹ دیتا ہے۔ `np.argmax(probs)` اس لسٹ میں سب سے بڑی Probability والے انڈیکس (یعنی جیتنے والے ہندسے `0-9`) کو نکال کر دیتا ہے۔
