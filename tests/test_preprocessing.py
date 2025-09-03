"""
Tests for text preprocessing service.
"""

import pytest
import numpy as np

from app.services.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""
    
    def test_clean_text_basic(self, text_preprocessor):
        """Test basic text cleaning."""
        text = "Hello World! This is a test email."
        cleaned = text_preprocessor.clean_text(text)
        assert cleaned == "hello world this is a test email"
    
    def test_clean_text_with_url(self, text_preprocessor):
        """Test text cleaning with URL removal."""
        text = "Visit our website at https://example.com for more info."
        cleaned = text_preprocessor.clean_text(text)
        assert "https://example.com" not in cleaned
        assert "visit our website at" in cleaned
    
    def test_clean_text_with_email(self, text_preprocessor):
        """Test text cleaning with email removal."""
        text = "Contact us at test@example.com for support."
        cleaned = text_preprocessor.clean_text(text)
        assert "test@example.com" not in cleaned
        assert "contact us at" in cleaned
    
    def test_clean_text_with_phone(self, text_preprocessor):
        """Test text cleaning with phone number removal."""
        text = "Call us at 123-456-7890 for assistance."
        cleaned = text_preprocessor.clean_text(text)
        assert "123-456-7890" not in cleaned
        assert "call us at" in cleaned
    
    def test_clean_text_empty(self, text_preprocessor):
        """Test cleaning empty text."""
        assert text_preprocessor.clean_text("") == ""
        assert text_preprocessor.clean_text(None) == ""
    
    def test_remove_stop_words(self, text_preprocessor):
        """Test stop words removal."""
        text = "this is a test email for the interview"
        cleaned = text_preprocessor.remove_stop_words(text)
        # Should remove common stop words like 'this', 'is', 'a', 'for', 'the'
        assert "this" not in cleaned
        assert "is" not in cleaned
        assert "a" not in cleaned
        assert "test" in cleaned
        assert "email" in cleaned
        assert "interview" in cleaned
    
    def test_preprocess_complete(self, text_preprocessor):
        """Test complete preprocessing pipeline."""
        text = "Hello! This is a test email with https://example.com and test@email.com"
        processed = text_preprocessor.preprocess(text)
        
        # Should be lowercase, no URLs, no emails, no stop words
        assert processed.islower()
        assert "https://example.com" not in processed
        assert "test@email.com" not in processed
        assert "this" not in processed  # stop word
        assert "is" not in processed    # stop word
        assert "a" not in processed     # stop word
        assert "test" in processed
        assert "email" in processed
    
    def test_preprocess_without_stop_words(self, text_preprocessor):
        """Test preprocessing without stop words removal."""
        text = "This is a test email"
        processed = text_preprocessor.preprocess(text, remove_stop_words=False)
        
        # Should keep stop words
        assert "this" in processed
        assert "is" in processed
        assert "a" in processed
    
    def test_extract_keywords(self, text_preprocessor):
        """Test keyword extraction."""
        text = "We would like to schedule an interview with you for the software engineer position"
        keywords = text_preprocessor.extract_keywords(text, top_n=5)
        
        assert len(keywords) <= 5
        assert "interview" in keywords
        assert "schedule" in keywords
        assert "software" in keywords
    
    def test_get_text_features(self, text_preprocessor):
        """Test text feature extraction."""
        text = "This is a test email with https://example.com and test@email.com and phone 123-456-7890"
        features = text_preprocessor.get_text_features(text)
        
        assert features["length"] > 0
        assert features["word_count"] > 0
        assert features["sentence_count"] > 0
        assert features["avg_word_length"] > 0
        assert features["has_url"] is True
        assert features["has_email"] is True
        assert features["has_phone"] is True
    
    def test_get_text_features_empty(self, text_preprocessor):
        """Test text features for empty text."""
        features = text_preprocessor.get_text_features("")
        
        assert features["length"] == 0
        assert features["word_count"] == 0
        assert features["sentence_count"] == 0
        assert features["avg_word_length"] == 0.0
        assert features["has_url"] is False
        assert features["has_email"] is False
        assert features["has_phone"] is False
