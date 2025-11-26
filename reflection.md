# Reflection – CTF Challenge Solution Methodology

## Problem Analysis
The challenge required identifying a manipulated book within a large dataset by using hashing, data filtering, text analysis, and machine learning. The attacker boosted a book’s rating to a perfect 5.0 with exactly 1234 reviews and embedded a SHA256-based clue inside a fake review. My objective was to detect this manipulated entry and extract three security flags using systematic analysis.

## Step-by-Step Solution

### Step 1 – Hash Computation & Book Identification
I computed the SHA256 hash of “STU031” which produced the personal hash 979DA9FA. Using the given anomaly criteria (rating_number = 1234 and average_rating = 5.0), I filtered the book dataset down to 150 candidate books. I then scanned all reviews for the embedded hash and successfully identified the manipulated target book: **“Four: A Divergent Collection.”**

### Step 2 – FLAG1 & FLAG2
FLAG1 was computed by taking the first eight non-space characters of the title ("Four:ADi") and hashing them, resulting in 70755B97. FLAG2 was extracted directly from the fake review as `FLAG2{979DA9FA}`.

## Step 3 – Machine Learning & Explainability for FLAG3
To generate FLAG3 and understand review authenticity, I built an ML model to differentiate genuine and suspicious reviews.

### ML Classification Approach
I engineered **14 numerical text features**, including:
- review length  
- word count  
- average word length  
- punctuation usage  
- superlative count  
- uppercase ratio  
- repetition ratio  
- sentiment word counts  

Additionally, I used **50 TF-IDF features** to represent linguistic patterns.  
A **Random Forest classifier** was trained using these features, achieving strong separation between suspicious (short, emotional, repetitive) reviews and genuine (detailed, descriptive) reviews.

### SHAP Explainability
I used SHAP values to identify which TF-IDF words contributed most to reducing suspicion.  
The top three genuine-review words were:

