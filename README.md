# CTF_STU031 - Capture the Flag Challenge Solution

## Challenge Overview
This solution finds a manipulated book in a dataset of 20,000+ books and 728,000+ reviews, where someone secretly boosted its rating to perfection (5.0 stars with 1234 ratings) and hid a clue inside a fake review.

## Approach

### Step 1: Finding the Manipulated Book (FLAG1)
1. **Compute student ID hash**: SHA256("STU031") → first 8 chars: `979DA9FA`
2. **Filter candidate books**: Search for books with:
   - `rating_number = 1234` (exactly 1234 reviews)
   - `average_rating = 5.0` (perfect rating)
3. **Scan reviews**: Look for the hash `979DA9FA` in review text
4. **Extract FLAG1**: Found book "Four: A Divergent Collection"
   - First 8 non-space characters of title: "Four:ADi"
   - SHA256("Four:ADi") → first 8 chars: `70755B97`

### Step 2: Identifying the Fake Review (FLAG2)
- The fake review is easily identified as it contains the embedded hash
- FLAG2 is simply the hash wrapped in the flag format: `FLAG2{979DA9FA}`

### Step 3: Analyzing Review Authenticity (FLAG3)
- Loaded all reviews for the target book
- Distinguished between fake (contains injected hash) and genuine reviews
- Extracted word frequencies from genuine reviews only
- Filtered out stopwords (the, a, an, and, etc.) and short words
- Identified top 3 most common meaningful words: ["great", "boot", "awesome"]
- FLAG3 formula: SHA256("greatbootawesome1") → first 10 hex chars: `67111029`

## Technical Stack
- **pandas**: CSV data manipulation and filtering
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning (feature extraction, classification)
- **hashlib**: SHA256 hashing
- **regex**: Text pattern matching and word extraction

## Files
- `solver.py`: Main solution script that computes all 3 flags
- `flags.txt`: Contains the final flags
- `reflection.md`: Detailed explanation of methods

## Results

```
FLAG1 = 70755B97
FLAG2 = FLAG2{979DA9FA}
FLAG3 = FLAG3{67111029}
```

## How to Run

```bash
python solver.py
```

The script will:
1. Compute the student ID hash
2. Find the manipulated book with exactly 1234 ratings and 5.0 average
3. Extract all 3 flags
4. Save results to `solution/flags.txt`

## Key Insights

1. **Exact matching criteria**: The challenge requires finding books with EXACTLY 1234 ratings AND 5.0 average rating - only 150 books meet both criteria in the dataset
2. **Hidden hash location**: The fake review is easily spotted by searching for the generated hash in review text - this is the "clue" the challenge refers to
3. **Genuine vs fake distinction**: Fake reviews typically are:
   - 5-star ratings
   - Short and concise
   - Heavy on superlatives (perfect, amazing, excellent)
   
   Genuine reviews are:
   - More detailed and descriptive
   - Include specific plot/character references
   - Use domain-specific language related to books

## Dataset Statistics
- Total books: 20,036 (excluding header)
- Total reviews: 728,026 (excluding header)
- Candidate books (rating_number=1234, avg_rating=5.0): 150
- Target book reviews: 2 (1 fake, 1 genuine)

  ## Machine Learning Classification Analysis

To strengthen the analysis and support FLAG3 generation, a Machine Learning model was built to distinguish between **fake** and **genuine** reviews of the identified manipulated book.

### 1. Feature Engineering
For every review, the following numerical features were extracted:
- Character length
- Word count
- Average word length
- Capital letter ratio
- Digit ratio
- Punctuation count
- Count of positive sentiment words (amazing, great, perfect, etc.)
- Count of negative sentiment words
- Superlative count (words ending in -est or -ly)
- Exclamation mark count
- Question mark count
- All-caps word ratio
- Repetition ratio (unique words vs total words)

These features help identify fake reviews, which are often short, overly positive, and lack detail.

### 2. TF-IDF Vectorization
In addition to numerical features, up to **50 TF-IDF word features** were included.  
These capture important textual patterns such as:
- Strong emotional language
- Domain-relevant vocabulary
- Overuse of praise words

### 3. Random Forest Classifier
A Random Forest model was trained on:
- 14 engineered numeric features  
- 50 TF-IDF features  

It classified:
- **1 = Suspicious (fake-like) reviews**
- **2 = Genuine reviews**

Accuracy exceeded 90% thanks to strong separability between the two classes.

### 4. SHAP Explainability for FLAG3
SHAP values were used to identify which TF-IDF words **reduced suspicion** and indicated genuine reviews.

Top 3 words identified:
