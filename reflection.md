# Reflection - CTF Challenge Solution Methodology

## Problem Analysis
The challenge required identifying a manipulated book in a 20,000+ book dataset where someone artificially inflated ratings (5.0 stars with exactly 1234 reviews) and embedded a hidden clue in a fake review.

## Solution Approach

**Step 1 - Hash Computation & Book Finding:**
Computed SHA256("STU031")  979DA9FA. Applied dual-criteria filtering (rating_number=1234, average_rating=5.0) reducing 20,000 books to 150 candidates. Scanned 728,000+ reviews for the hash and located "Four: A Divergent Collection."

**Step 2 - FLAG1 Extraction:**
Extracted first 8 non-space characters from title: "Four:ADi". Computed SHA256("Four:ADi")  70755B97.

**Step 3 - FLAG2 Identification:**
Located the fake review containing embedded hash (979DA9FA) and formatted as FLAG2{979DA9FA}.

**Step 4 - FLAG3 (Model-Based Analysis):**
Trained a Random Forest classifier to distinguish suspicious (short, superlative-heavy) from genuine (detailed, domain-specific) reviews. Engineered 14 numerical features plus 50 TF-IDF features achieving 100% accuracy. Applied SHAP analysis on genuine reviews to identify features reducing suspicion. Top 3 words: "great," "boot," "awesome." Generated FLAG3 via SHA256("greatbootawesome1")  67111029.

## Key Learnings
Data formatting precision is critical—initial spacing error ("STU 031") produced wrong hashes. Efficient filtering achieved 99% search space reduction. SHAP analysis effectively explained model decisions for anomaly detection. Combining traditional data analysis with modern ML created a robust detection pipeline, demonstrating how explainable AI helps understand automated decisions in security applications.
