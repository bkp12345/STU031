#!/usr/bin/env python3
"""
ML Model Demo - Demonstrates the classification model predicting suspicious reviews
Shows which words indicate suspicious vs genuine reviews
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

print("="*70)
print("ML CLASSIFICATION MODEL DEMONSTRATION")
print("="*70)

# ============================================================================
# TRAINING DATA: Create synthetic dataset
# ============================================================================

print("\n[TRAINING] Creating synthetic training data...")

# Suspicious reviews (short, lots of superlatives, many exclamation marks)
suspicious_reviews = [
    "AMAZING!!! PERFECT!!! BEST BOOK EVER!!! HIGHLY RECOMMEND!!!",
]

# Genuine reviews (detailed, domain-specific)
genuine_reviews = [
    "Great book with excellent character development. The plot was well-structured and kept me engaged throughout.",
    "The author did a fantastic job with the world-building. Very enjoyable read with meaningful themes.",
]

all_reviews = suspicious_reviews + genuine_reviews
labels = np.array([1] * len(suspicious_reviews) + [2] * len(genuine_reviews))

print(f"  Suspicious reviews: {len(suspicious_reviews)}")
print(f"  Genuine reviews: {len(genuine_reviews)}")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

print("\n[FEATURES] Extracting numerical features...")

def extract_features(text):
    """Extract 14 numerical features from review text"""
    features = {}
    features['char_length'] = len(text)
    features['word_count'] = len(text.split())
    words = text.split()
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    features['punctuation_count'] = sum(1 for c in text if c in '!?.,:;-')
    
    positive_words = ['amazing', 'excellent', 'great', 'wonderful', 'perfect', 'awesome', 'love', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'waste', 'poor', 'hate', 'boring']
    
    features['positive_count'] = sum(1 for w in words if w.lower() in positive_words)
    features['negative_count'] = sum(1 for w in words if w.lower() in negative_words)
    features['superlative_count'] = sum(1 for w in words if w.lower().endswith(('est', 'ly')))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['all_caps_ratio'] = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
    unique_words = len(set(w.lower() for w in words))
    features['repetition_ratio'] = 1 - (unique_words / max(len(words), 1))
    
    return features

# Extract numerical features
numerical_features = pd.DataFrame([extract_features(text) for text in all_reviews])

# Extract TF-IDF features
print("[FEATURES] Computing TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
tfidf_features = vectorizer.fit_transform(all_reviews).toarray()

# Combine features
X_combined = np.hstack([numerical_features.values, tfidf_features])

print(f"  Total features: {X_combined.shape[1]} (14 numerical + 50 TF-IDF)")
print(f"  Training samples: {X_combined.shape[0]}")

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n[TRAINING] Training Random Forest classifier...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_scaled, labels)

accuracy = model.score(X_scaled, labels)
print(f"  Accuracy: {accuracy*100:.1f}%")
print(f"  Classes: 1=Suspicious, 2=Genuine")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n[FEATURES] Top 10 most important features for detection:")

feature_importance = model.feature_importances_
feature_names = list(numerical_features.columns) + list(vectorizer.get_feature_names_out())
sorted_idx = np.argsort(feature_importance)[::-1][:10]

for rank, idx in enumerate(sorted_idx, 1):
    fname = feature_names[idx]
    importance = feature_importance[idx]
    print(f"  {rank:2d}. {fname:25s} {importance:.4f}")

# ============================================================================
# PREDICTIONS ON TEST REVIEWS
# ============================================================================

print("\n" + "="*70)
print("PREDICTIONS ON TEST REVIEWS")
print("="*70)

test_reviews = [
    ("This book is absolutely AMAZING!!! PERFECT!!! Best ever!!!!", "TEST 1 - Suspicious Pattern"),
    ("Great book with excellent characters and compelling plot.", "TEST 2 - Genuine Pattern"),
    ("Interesting read. The story had interesting elements.", "TEST 3 - Neutral Review"),
]

class_map = {1: "SUSPICIOUS", 2: "GENUINE"}

for test_text, label in test_reviews:
    print(f"\n[{label}]")
    print(f"Review: {test_text}")
    
    # Extract features
    test_features = extract_features(test_text)
    test_tfidf = vectorizer.transform([test_text]).toarray()
    X_test = np.hstack([pd.DataFrame([test_features]).values, test_tfidf])
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    prediction = model.predict(X_test_scaled)[0]
    probabilities = model.predict_proba(X_test_scaled)[0]
    confidence = probabilities[prediction - 1] * 100
    
    print(f"Prediction: {class_map[prediction]}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Probabilities - Suspicious: {probabilities[0]*100:.1f}%, Genuine: {probabilities[1]*100:.1f}%")
    
    # Show top words in review
    tfidf_scores = test_tfidf[0]
    top_indices = np.argsort(tfidf_scores)[::-1][:3]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices if tfidf_scores[i] > 0]
    
    if top_words:
        print(f"Key words: {', '.join(top_words)}")

# ============================================================================
# SHOW SUSPICIOUS VS GENUINE INDICATORS
# ============================================================================

print("\n" + "="*70)
print("REVIEW CHARACTERISTICS")
print("="*70)

print("\nSUSPICIOUS REVIEW INDICATORS:")
print("  - Short text length")
print("  - High capitalization ratio")
print("  - Multiple exclamation marks")
print("  - High positive word count")
print("  - Superlatives (amazing, perfect, excellent)")
print("  - Words: AMAZING, PERFECT, BEST, LOVE, MUST, HIGHLY RECOMMEND")

print("\nGENUINE REVIEW INDICATORS:")
print("  - Longer, detailed text")
print("  - Balanced tone")
print("  - Domain-specific words (plot, characters, development, theme)")
print("  - Specific story elements mentioned")
print("  - Words: great, good, interesting, engaging, enjoyed")

print("\n" + "="*70)
print("ML MODEL DEMONSTRATION COMPLETE")
print("="*70)
