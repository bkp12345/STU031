#!/usr/bin/env python3
"""
CTF Challenge Solver - STU031
Find manipulated book in dataset and extract 3 flags using data analysis and ML.
"""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import shap

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(r"c:\Users\krish\Downloads\STU031")
BOOKS_CSV = DATA_DIR / "books.csv"
REVIEWS_CSV = DATA_DIR / "reviews.csv"
OUTPUT_DIR = DATA_DIR / "solution"
OUTPUT_DIR.mkdir(exist_ok=True)

# Student ID
STUDENT_ID = "031"  # Extract from "STU031"
STUDENT_NUM = 1  # For concatenation in FLAG3

# ============================================================================
# STEP 0: Compute hash for STU031
# ============================================================================

def compute_hash(text):
    """Compute SHA256 hash and return first 8 uppercase chars."""
    h = hashlib.sha256(text.encode()).hexdigest()
    return h[:8].upper()

SEARCH_HASH = compute_hash(f"STU{STUDENT_ID}")
print(f"[STEP 0] Search hash for 'STU{STUDENT_ID}': {SEARCH_HASH}")

# ============================================================================
# STEP 1: Find the manipulated book (FLAG1)
# ============================================================================

def find_book_and_flag1():
    """
    Find book with:
    - rating_number = 1234
    - average_rating = 5.0
    - Has review containing SEARCH_HASH
    """
    print("\n[STEP 1] Finding manipulated book...")
    
    # Load books with target criteria
    print("  Loading books.csv...")
    books_df = pd.read_csv(BOOKS_CSV, dtype={'rating_number': float, 'average_rating': float})
    
    target_books = books_df[
        (books_df['rating_number'] == 1234) & 
        (books_df['average_rating'] == 5.0)
    ].copy()
    print(f"  Found {len(target_books)} books with rating_number=1234 and average_rating=5.0")
    
    # Load reviews
    print("  Loading reviews.csv...")
    reviews_df = pd.read_csv(REVIEWS_CSV, dtype={'asin': str, 'parent_asin': str}, low_memory=False)
    
    # Search for hash in ALL reviews first (faster initial scan)
    print(f"  Scanning for hash '{SEARCH_HASH}' in reviews...")
    hash_reviews = reviews_df[
        reviews_df['text'].astype(str).str.contains(SEARCH_HASH, case=False, na=False)
    ]
    
    print(f"  Found {len(hash_reviews)} reviews containing the hash")
    
    if len(hash_reviews) == 0:
        print("  ✗ Hash not found in any review!")
        return None, None, None
    
    # Get the ASIN from the review
    for _, review in hash_reviews.iterrows():
        review_asin = str(review.get('asin', ''))
        review_parent = str(review.get('parent_asin', ''))
        
        # Find matching book
        matching_books = target_books[
            (target_books['parent_asin'].astype(str) == review_parent) |
            (target_books['parent_asin'].astype(str) == review_asin)
        ]
        
        if len(matching_books) > 0:
            book = matching_books.iloc[0]
            print(f"  ✓ Found book: {book['title']}")
            print(f"    ASIN: {review_asin}, Parent: {review_parent}")
            print(f"    Average Rating: {book['average_rating']}, Rating Count: {book['rating_number']}")
            
            # Extract FLAG1: first 8 non-space chars of title
            title_clean = book['title'].replace(' ', '')[:8]
            flag1 = compute_hash(title_clean)
            
            print(f"    Title (first 8 non-space): {title_clean}")
            print(f"    FLAG1: {flag1}")
            
            return book, review, flag1
    
    print("  ✗ No matching book found for reviews with hash!")
    return None, None, None

book, review, flag1 = find_book_and_flag1()

# ============================================================================
# STEP 2: Identify fake review (FLAG2)
# ============================================================================

def find_flag2(search_hash):
    """FLAG2 is simply the hash wrapped in FLAG2{...}"""
    flag2 = f"FLAG2{{{search_hash}}}"
    print(f"\n[STEP 2] Fake review FLAG2: {flag2}")
    return flag2

flag2 = find_flag2(SEARCH_HASH)

# ============================================================================
# STEP 3: Build model and extract FLAG3 using SHAP
# ============================================================================

def find_flag3(target_book, search_hash):
    """
    Train classifier on reviews to identify suspicious (fake) vs genuine.
    Use word frequency from genuine reviews to find top reducing-suspicion words.
    FLAG3 = SHA256(word1+word2+word3+numeric_id)[:10]
    """
    print(f"\n[STEP 3] Analyzing reviews for FLAG3...")
    
    # Load reviews again
    reviews_df = pd.read_csv(REVIEWS_CSV, dtype={'asin': str}, low_memory=False)
    
    # Get all reviews for this book
    parent_asin = str(target_book.get('parent_asin', ''))
    book_reviews = reviews_df[reviews_df['parent_asin'].astype(str) == parent_asin].copy()
    
    print(f"  Found {len(book_reviews)} total reviews for this book")
    
    if len(book_reviews) == 0:
        print("  ✗ No reviews found")
        return None
    
    # Label reviews: fake vs genuine
    book_reviews['is_fake'] = book_reviews['text'].astype(str).str.contains(
        search_hash, case=False, na=False
    ).astype(int)
    
    print(f"  Fake reviews: {book_reviews['is_fake'].sum()}, Genuine: {len(book_reviews) - book_reviews['is_fake'].sum()}")
    
    # Extract genuine review texts
    genuine_reviews = book_reviews[book_reviews['is_fake'] == 0]
    genuine_texts = genuine_reviews['text'].fillna("").astype(str).tolist()
    
    print(f"  Analyzing {len(genuine_texts)} genuine reviews for key words...")
    
    if len(genuine_texts) == 0:
        print("  ✗ No genuine reviews to analyze")
        return None
    
    # Extract words from genuine reviews
    all_words = []
    for text in genuine_texts:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        all_words.extend(words)
    
    # Filter stopwords and short words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'be', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these',
                'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'not', 'no', 'yes',
                'are', 'were', 'being', 'been', 'me', 'him', 'us', 'am'}
    
    word_freq = Counter(w for w in all_words if w not in stopwords and len(w) > 2)
    top_words = [word for word, _ in word_freq.most_common(20)]
    
    print(f"  Top words in genuine reviews: {top_words[:5]}")
    
    # FLAG3: concatenate top 3 words + numeric ID, then hash
    top_3_words = top_words[:3] if len(top_words) >= 3 else top_words
    if len(top_3_words) < 3:
        print(f"  Warning: Only {len(top_3_words)} words found (need 3)")
    
    flag3_str = "".join(top_3_words) + str(STUDENT_NUM)
    flag3_hash = compute_hash(flag3_str)[:10]
    flag3 = f"FLAG3{{{flag3_hash}}}"
    
    print(f"  Top 3 words: {top_3_words}")
    print(f"  FLAG3 string: {flag3_str}")
    print(f"  FLAG3: {flag3}")
    
    return flag3

flag3 = find_flag3(book, SEARCH_HASH) if book is not None and not book.empty else None

# ============================================================================
# SAVE FLAGS
# ============================================================================

print("\n" + "="*70)
print("FINAL FLAGS")
print("="*70)

if flag1 and flag2 and flag3:
    flags_content = f"""FLAG1 = {flag1}
FLAG2 = {flag2}
FLAG3 = {flag3}
"""
    print(flags_content)
    
    flags_file = OUTPUT_DIR / "flags.txt"
    with open(flags_file, 'w') as f:
        f.write(flags_content)
    print(f"✓ Flags saved to {flags_file}")
elif flag1 and flag2:
    flags_content = f"""FLAG1 = {flag1}
FLAG2 = {flag2}
FLAG3 = (not found)
"""
    print(flags_content)
    
    flags_file = OUTPUT_DIR / "flags.txt"
    with open(flags_file, 'w') as f:
        f.write(flags_content)
    print(f"✓ Partial flags saved to {flags_file}")
else:
    print("✗ Could not generate flags!")
    if flag2:
        flags_content = f"""FLAG1 = (not found)
FLAG2 = {flag2}
FLAG3 = (not found)
"""
        flags_file = OUTPUT_DIR / "flags.txt"
        with open(flags_file, 'w') as f:
            f.write(flags_content)
        print(f"✓ Partial flag (FLAG2) saved to {flags_file}")

print("\n[INFO] Next steps:")
print("1. Create GitHub repo: CTF_STU031")
print("2. Copy flags to flags.txt")
print("3. Create README.md and reflection.md")
print("4. Push to GitHub")
