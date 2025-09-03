#!/usr/bin/env python3
"""
Script to download required NLTK data.
"""

import nltk
import sys


def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    
    try:
        # Download punkt tokenizer
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        
        # Download punkt_tab for newer NLTK versions
        print("Downloading punkt_tab...")
        nltk.download('punkt_tab', quiet=True)
        
        # Download stopwords
        print("Downloading stopwords...")
        nltk.download('stopwords', quiet=True)
        
        print("NLTK data download completed successfully!")
        
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_nltk_data()
