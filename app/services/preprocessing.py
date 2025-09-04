"""
Text preprocessing service for email classification.
"""

import re
from typing import List, Optional

import nltk
import structlog
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize logger
logger = structlog.get_logger()


class TextPreprocessor:
    """
    Text preprocessing utility for email classification.
    """

    def __init__(self):
        """Initialize the text preprocessor."""
        self._stop_words: Optional[set] = None
        self._download_nltk_data()

    def _download_nltk_data(self) -> None:
        """Download required NLTK data."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)

    def _get_stop_words(self) -> set:
        """Get English stop words."""
        if self._stop_words is None:
            try:
                self._stop_words = set(stopwords.words("english"))
            except LookupError:
                logger.warning("NLTK stopwords not available, using empty set")
                self._stop_words = set()
        return self._stop_words

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # Remove phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def remove_stop_words(self, text: str) -> str:
        """
        Remove stop words from text.

        Args:
            text: Input text

        Returns:
            Text with stop words removed
        """
        if not text:
            return ""

        try:
            # Tokenize text
            tokens = word_tokenize(text)

            # Remove stop words
            stop_words = self._get_stop_words()
            filtered_tokens = [token for token in tokens if token not in stop_words]

            return " ".join(filtered_tokens)
        except Exception as e:
            logger.warning("Failed to remove stop words", error=str(e))
            return text

    def preprocess(self, text: str, remove_stop_words: bool = True) -> str:
        """
        Complete text preprocessing pipeline.

        Args:
            text: Input text to preprocess
            remove_stop_words: Whether to remove stop words

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Clean text
        cleaned_text = self.clean_text(text)

        # Remove stop words if requested
        if remove_stop_words:
            cleaned_text = self.remove_stop_words(cleaned_text)

        return cleaned_text

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract top keywords from text.

        Args:
            text: Input text
            top_n: Number of top keywords to extract

        Returns:
            List of top keywords
        """
        if not text:
            return []

        try:
            # Preprocess text
            processed_text = self.preprocess(text, remove_stop_words=True)

            # Tokenize and count words
            tokens = word_tokenize(processed_text)
            word_freq = {}

            for token in tokens:
                if len(token) > 2:  # Only consider words longer than 2 characters
                    word_freq[token] = word_freq.get(token, 0) + 1

            # Sort by frequency and return top N
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:top_n]]

        except Exception as e:
            logger.warning("Failed to extract keywords", error=str(e))
            return []

    def get_text_features(self, text: str) -> dict:
        """
        Extract various text features for analysis.

        Args:
            text: Input text

        Returns:
            Dictionary of text features
        """
        if not text:
            return {
                "length": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0.0,
                "has_url": False,
                "has_email": False,
                "has_phone": False,
            }

        # Basic features
        length = len(text)
        word_count = len(text.split())

        # Sentence count
        sentences = re.split(r"[.!?]+", text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Average word length
        words = text.split()
        avg_word_length = (
            sum(len(word) for word in words) / len(words) if words else 0.0
        )

        # Pattern detection
        has_url = bool(re.search(r"http[s]?://", text))
        has_email = bool(
            re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        )
        has_phone = bool(re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text))

        return {
            "length": length,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "has_url": has_url,
            "has_email": has_email,
            "has_phone": has_phone,
        }
