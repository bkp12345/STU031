# Reflection - CTF Challenge Solution Methodology

## Problem Analysis
The Capture the Flag challenge required identifying a manipulated book entry in a large book-review dataset where someone artificially inflated ratings (to exactly 5.0 stars with 1234 reviews) and embedded a hidden clue in a fake review.

## Methodology

### Phase 1: Hash Computation & Verification
**Challenge**: Determine the correct format for computing the student ID hash.
- Initial attempt used "STU 031" (with space) → 5E51D93E (not found in dataset)
- Corrected to "STU031" (no space) → 979DA9FA (found!)
- **Learning**: Precision in input formatting is critical; debug by testing variations against actual data

### Phase 2: Book Identification Strategy
**Challenge**: Filter from 20,000+ books to find the single manipulated entry.

**Approach**:
1. Applied dual-criteria filtering: rating_number=1234 AND average_rating=5.0
   - This reduced ~20,000 books to 150 candidates (0.75%)
2. Scanned 728,000+ reviews for the hash string
   - Found hash in exactly 1 review (false positive free)
3. Matched review's ASIN back to books dataframe

**Why this works**: The challenge designer ensured uniqueness by:
- Setting very specific rating criteria (exactly 1234 reviews at 5.0 stars)
- Embedding the hash where it stands out (only 1 review has it)

### Phase 3: FLAG1 Extraction
**Challenge**: Convert book title to first 8 non-space characters and hash it.
- Title: "Four: A Divergent Collection"
- Cleaned: "Four:ADi" (first 8 chars after removing spaces)
- SHA256("Four:ADi") → 70755B97 (first 8 hex chars)

**Key insight**: "Non-space characters" means literally removing spaces, not splitting by word boundaries.

### Phase 4: FLAG2 (Straightforward)
FLAG2 simply wraps the student hash in the required format: FLAG2{979DA9FA}

### Phase 5: FLAG3 - Genuine Review Analysis
**Challenge**: Distinguish genuine reviews from the injected fake and extract meaningful words.

**Methodology**:
1. **Classification approach**: Instead of building a complex ML model, used word frequency analysis on genuine reviews
2. **Fake vs Genuine distinction**:
   - Fake: "Perfect amazing experience 979DA9FA" (contains hash, superlatives)
   - Genuine: Longer, more descriptive reviews with plot/character references
3. **Word extraction**:
   - Tokenized all text to lowercase words
   - Filtered out 50+ common stopwords (the, and, or, is, etc.)
   - Removed words shorter than 3 characters
   - Computed frequency: {"great": 2, "boot": 1, "awesome": 1, "read": 1, "book": 1}
4. **Top 3 words**: ["great", "boot", "awesome"]
5. **Final FLAG3**: SHA256("greatbootawesome1") → 67111029 (first 10 chars)

## Tools & Libraries
- **pandas**: Efficient CSV reading and filtering (728K rows scanned in <5 seconds)
- **hashlib**: SHA256 implementation (always available, no external deps needed)
- **regex**: Pattern matching for hex strings and word extraction

## Challenges Encountered & Solutions

| Challenge | Solution |
|-----------|----------|
| Hash not found in reviews | Tested all format variations; found "STU031" (no space) was correct |
| Parent ASIN/ASIN confusion | Implemented flexible matching on both fields |
| Pandas Series truthiness error | Used explicit `.empty` check instead of boolean eval |
| Large CSV file scanning | Used streaming approach with `str.contains()` vectorization |

## Key Takeaways

1. **Data precision matters**: Off-by-one errors in criteria (1234 vs 1235 reviews) or string formatting completely change results
2. **Verify assumptions**: Don't assume hash format; test against actual data
3. **Simplicity wins**: Complex ML models weren't needed; basic word frequency on genuine reviews was effective
4. **Efficiency at scale**: Pandas vectorized operations processed 728K records faster than manual iteration

## Time Complexity
- Filter books: O(n) where n = 20,036
- Scan reviews: O(m) where m = 728,026
- Word extraction: O(k) where k = word count in reviews
- Overall: Linear in dataset size (highly scalable)

## Lessons for Future Challenges
1. Always validate assumptions against real data
2. Start simple (frequency analysis) before complex ML
3. Debug systematically (test variations, print intermediate results)
4. Document edge cases (space in "STU 031" was the key differentiator)
