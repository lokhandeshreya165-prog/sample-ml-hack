# Member 2 — Matching Model + Feature Engineering
## Business Entity Resolution Challenge 2026

This repository contains the complete, modular, and runnable **Member 2** pipeline for the Business Entity Resolution Challenge.

Member 2 is responsible **strictly for the matching stage** following candidate generation (blocking by Member 1).

---

## 📁 Repository Structure

```
ml-hack/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Robust TSV parsing, ground truth, dynamic candidate pairs loading
│   ├── text_processing.py      # Text normalization, corporate suffixes, address abbreviations, digits
│   ├── feature_engineering.py  # 21 similarity, token, Jaccard, numeric, and missingness features
│   ├── matching_model.py       # Classifier wrappers (Random Forest, LightGBM, Gradient Boosting)
│   ├── threshold_tuning.py     # F0.5 metric calculation and validation threshold grid search
│   └── matching.py             # Batch candidate scoring, constraint validation, matching_results.tsv
├── scripts/
│   ├── __init__.py
│   ├── train_matching.py       # Full training pipeline + F0.5 threshold tuning + model saving
│   ├── predict_matching.py     # Test candidate scoring + constraint verification + submission output
│   └── generate_sample_candidates.py # Helper to generate sample candidates for testing
├── models/
│   ├── matching_model.joblib   # Trained classifier
│   ├── threshold_config.json   # Selected optimal F0.5 threshold and validation metrics
│   └── matching_model_meta.json# Model metadata and feature importances
├── output/
│   ├── candidate_pairs.tsv     # Input from Member 1 (candidate pairs to score)
│   └── matching_results.tsv    # Final Member 2 output scored on leaderboard
├── validate_submission.py      # Official challenge submission validator
├── requirements.txt            # Pinned dependencies
└── README.md                   # Instructions and documentation
```

---

## ⚙️ Environment Setup & Installation

### 1. Create and Activate Virtual Environment

```bash
# Using Python 3.10+
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Feature Engineering Architecture

For every candidate pair $(S_1, S_{cand})$, Member 2 extracts **21 specialized similarity and indicator features** designed to tackle entity resolution edge cases:

### Business Name Features
1. `name_edit_sim`: Normalized Levenshtein similarity on canonical business names (handles typos, spelling errors).
2. `name_root_edit_sim`: Normalized Levenshtein similarity on root business names with corporate suffixes removed (e.g., `Orelee's Barbershop LLC` vs `Orelee's Barbershop Inc` -> 100% root match).
3. `name_token_sort_sim`: Token sort similarity (handles word order permutations like `Barbershop Orelee` vs `Orelee Barbershop`).
4. `name_token_set_sim`: Token set similarity (handles subset words and extra words like `Vision Partners Corp` vs `Vision Partners`).
5. `name_jaccard_word`: Jaccard similarity on word token sets.
6. `name_jaccard_char_3gram`: Jaccard similarity on character 3-grams (resilient to character swaps and OCR noise).
7. `name_exact_match`: Binary indicator (1.0 if identical normalized strings).
8. `name_length_diff`: Normalized string length difference.

### Address & Numeric Features
9. `addr_edit_sim`: Normalized Levenshtein similarity on normalized addresses.
10. `addr_token_sort_sim`: Token sort similarity on address tokens (handles reordered components like `High Point, NC, 1795 Westchester Dr` vs `1795 Westchester Dr, High Point, NC`).
11. `addr_token_set_sim`: Token set similarity on address tokens.
12. `addr_jaccard_word`: Jaccard similarity on address word tokens.
13. `addr_jaccard_char_3gram`: Character 3-gram Jaccard similarity on addresses.
14. `addr_exact_match`: Binary indicator for identical normalized address.
15. `addr_numeric_exact`: 1.0 if both addresses contain digits and their digit sets match exactly; 0.0 if both have digits and differ (e.g. street number 1064 vs 1065); 0.5 if neither has digits.
16. `addr_numeric_jaccard`: Jaccard similarity on extracted street numbers and zip codes.
17. `addr_numeric_has_overlap`: 1.0 if any numeric component overlaps.

### Country & Missingness Features
18. `country_match`: 1.0 if countries match; 0.0 if they disagree; 0.5 if either is missing/unknown.
19. `name_missing`: Binary indicator if name is missing/empty.
20. `addr_missing`: Binary indicator if address is missing/empty.
21. `country_missing`: Binary indicator if country is missing/empty.

---

## 🚀 Running the Pipeline

### Step 1: Model Training & Threshold Tuning

Run the training pipeline from the project root:

```bash
# Default training (Random Forest, 50,000 candidate pairs, F0.5 threshold tuning)
python scripts/train_matching.py

# Optional: Train with LightGBM
python scripts/train_matching.py --model-type lightgbm --num-samples 50000

# Optional: If Member 1 provides train candidate pairs
python scripts/train_matching.py --train-candidates output/train_candidate_pairs.tsv
```

#### What happens during training:
1. Loads `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv`.
2. Generates/loads positive ground truth pairs + hard/soft negative candidate pairs.
3. Performs a **grouped train/validation split by Source 1 entity ID** (preventing data leakage).
4. Extracts feature matrices $X_{\text{train}}$ and $X_{\text{val}}$.
5. Trains the matching classifier with balanced class weighting.
6. Evaluates validation probabilities over a fine grid (0.10 to 0.95) to find the decision threshold $\tau^*$ that **maximizes $F_{0.5}$** (precision weighted twice as heavily as recall).
7. Saves the trained model to `models/matching_model.joblib` and threshold configuration to `models/threshold_config.json`.

---

### Step 2: Scoring Test Candidate Pairs & Generating Output

Once Member 1 provides `output/candidate_pairs.tsv`, run:

```bash
python scripts/predict_matching.py
```

Options:
- `--candidate-pairs output/candidate_pairs.tsv`: Path to candidate pairs from Member 1.
- `--threshold 0.70`: Optional manual threshold override (default: automatically loads tuned F0.5 threshold).
- `--batch-size 50000`: Batch size for streaming scoring without memory blowup.
- `--output output/matching_results.tsv`: Output file location.

#### Strict Constraint Validation:
The script automatically verifies:
- Every Source 1 entity from `test_source1.tsv` appears **exactly once**.
- `matched_entity_ids` contains **only valid `S2-` and `S3-` prefixed IDs** (no self-matches, no raw IDs).
- **No duplicate IDs** within any entity's match list.
- Entities with no match passing the threshold receive an empty string.
- **Every matched pair is strictly verified to exist in Member 1's `candidate_pairs.tsv`**.
- Automatically runs `validate_submission.py` to confirm exit code 0 (Pass).

---

## 🧪 Submission Validation

Run the official challenge validator at any time:

```bash
python validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir .
```

---

## 🛡️ Robustness & Error Handling

- **Flexible Column Detection**: `candidate_pairs.tsv` format is automatically inspected. Both comma-separated lists (`source1_entity_id\tcandidate_entity_ids`) and pairwise rows (`source1_entity_id\tcandidate_entity_id`) are supported.
- **Missing File Diagnostics**: Clear, actionable error messages explain what file is missing and whether it is expected from Member 1.
- **Memory Efficiency**: Entities are pre-filtered and cached as lightweight slots; scoring operates in batches of 50,000 to scale comfortably to millions of pairs.
- **Pure Python Fallbacks**: String algorithms dynamically use C-accelerated `rapidfuzz` when available with pure-Python fallbacks.
- **Zero External Dependencies**: Operates completely offline without any internet access, external databases, or geocoding APIs.
