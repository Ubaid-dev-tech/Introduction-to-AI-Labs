"""Laboratory 01 NLP Tokenization and Stop-Word Removal Walkthrough (from Lab 01 Manual)."""

import re

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"
}


def run_tokenization_demo():
    text = "Natural Language Processing is a key part of AI and data science."
    if NLTK_AVAILABLE:
        try:
            word_tokens = word_tokenize(text)
            stop_words = set(stopwords.words("english"))
        except LookupError:
            word_tokens = re.findall(r"\w+|[^\w\s]", text)
            stop_words = FALLBACK_STOPWORDS
    else:
        word_tokens = re.findall(r"\w+|[^\w\s]", text)
        stop_words = FALLBACK_STOPWORDS

    print("Original sentence:")
    print(text)
    print("\nTokens list:")
    print(word_tokens)

    print("\nTokens:")
    for token in word_tokens:
        print(token)

    filtered_words = [w for w in word_tokens if w.lower() not in stop_words]
    print("\nWords after removing stopwords:")
    for word in filtered_words:
        print(word)


if __name__ == "__main__":
    run_tokenization_demo()
