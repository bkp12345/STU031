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
