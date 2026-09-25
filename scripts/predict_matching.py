#!/usr/bin/env python3
"""Run Prediction & Generate Final matching_results.tsv for Test Set.

Workflow:
1. Validates presence of test source files, candidate_pairs.tsv, and trained model.
2. Dynamically inspects column names of output/candidate_pairs.tsv.
3. Reads all candidate pairs and Source 1, 2, 3 test entities.
4. Computes rich similarity and component features in batches.
5. Scores candidate pairs using the trained MatchingClassifier and tuned F0.5 threshold.
6. Assembles output/matching_results.tsv ensuring:
   - Every Source 1 test entity appears exactly once.
   - matched_entity_ids contains only valid S2- and S3- prefixed IDs.
   - No duplicate IDs within any row.
   - If no candidate passes the threshold, matched_entity_ids is empty.
   - Every matched pair is strictly verified against candidate_pairs.tsv.
7. Validates format against validate_submission.py.

Usage:
    python scripts/predict_matching.py
    python scripts/predict_matching.py --candidate-pairs output/candidate_pairs.tsv --threshold 0.65
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    DELIM,
    load_candidate_pairs,
    load_source_tsv,
    resolve_file_path,
)
from src.matching import (
    format_and_verify_results,
    score_candidates_batch,
    write_matching_results,
)
from src.matching_model import MatchingClassifier


def main():
    parser = argparse.ArgumentParser(description="Predict matches on test candidate pairs and output matching_results.tsv")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv", help="Path to output/candidate_pairs.tsv from Member 1")
    parser.add_argument("--test-source1", default="test_source1.tsv", help="Path to test_source1.tsv")
    parser.add_argument("--test-source2", default="test_source2.tsv", help="Path to test_source2.tsv (optional if no S2 candidates)")
    parser.add_argument("--test-source3", default="test_source3.tsv", help="Path to test_source3.tsv")
    parser.add_argument("--model", default="models/matching_model.joblib", help="Path to trained model file")
    parser.add_argument("--threshold-config", default="models/threshold_config.json", help="Path to threshold config JSON")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold (default: uses tuned F0.5 threshold)")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for feature extraction and scoring")
    parser.add_argument("--output", default="output/matching_results.tsv", help="Path to output matching_results.tsv")
    parser.add_argument("--skip-validator", action="store_true", help="Skip running validate_submission.py")
    args = parser.parse_args()

    # Reconfigure stdout for UTF-8 display
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("MEMBER 2: TEST PREDICTION & MATCHING RESULTS GENERATION")
    print("=" * 70)

    # 1. Verify candidate pairs file existence
    cand_path = None
    try:
        cand_path = resolve_file_path(args.candidate_pairs, search_dirs=["output", "dataset/test", "."])
    except FileNotFoundError:
        print("\n" + "!" * 70)
        print(f"ERROR: Candidate pairs file not found: '{args.candidate_pairs}'")
        print("Candidate generation is the responsibility of MEMBER 1.")
        print("Once Member 1 completes blocking, ensure 'output/candidate_pairs.tsv'")
        print("is placed in the project root or output/ directory, then rerun this script.")
        print("!" * 70)
        sys.exit(1)

    # 2. Load Model & Threshold
    print("\n[Step 1/5] Loading trained matching model & decision threshold...")
    model_path = resolve_file_path(args.model, search_dirs=["models", "."])
    model, saved_threshold = MatchingClassifier.load(model_path)
    print(f"Loaded model ({model.model_type}) from: {model_path}")

    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using explicitly specified threshold: {threshold:.4f}")
    elif os.path.isfile(args.threshold_config):
        with open(args.threshold_config, "r", encoding="utf-8") as f:
            tcfg = json.load(f)
            threshold = float(tcfg.get("selected_threshold", saved_threshold or 0.5))
        print(f"Loaded optimal F0.5 threshold from {args.threshold_config}: {threshold:.4f}")
    else:
        threshold = saved_threshold or 0.5
        print(f"Using model saved threshold: {threshold:.4f}")

    # 3. Inspect and Load Candidate Pairs
    print(f"\n[Step 2/5] Inspecting and loading candidate pairs from: {cand_path}")
    flat_pairs, cand_map, detected_headers = load_candidate_pairs(cand_path)
    print(f"  Detected columns in candidate file: {detected_headers}")
    print(f"  Loaded {len(flat_pairs):,} total candidate pairs across {len(cand_map):,} Source 1 entities.")
    if len(flat_pairs) == 0:
        print("WARNING: Candidate pairs file contains 0 candidate pairs!")

    # 4. Load Test Source Records
    print("\n[Step 3/5] Loading test entities...")
    s1_path = resolve_file_path(args.test_source1, search_dirs=["dataset/test", "."])
    s1_entities = load_source_tsv(s1_path)
    print(f"  Test Source 1: {len(s1_entities):,} entities loaded from {s1_path}")
    all_s1_ids = set(s1_entities.keys())

    target_entities: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}

    # Test Source 2 (optional check)
    try:
        s2_path = resolve_file_path(args.test_source2, search_dirs=["dataset/test", "."], must_exist=False)
        if os.path.isfile(s2_path):
            s2_entities = load_source_tsv(s2_path)
            print(f"  Test Source 2: {len(s2_entities):,} entities loaded from {s2_path}")
            target_entities.update(s2_entities)
        else:
            print(f"  Notice: Test Source 2 not found at {s2_path}. Continuing...")
    except Exception as e:
        print(f"  Notice loading test_source2: {e}")

    # Test Source 3
    try:
        s3_path = resolve_file_path(args.test_source3, search_dirs=["dataset/test", "."])
        s3_entities = load_source_tsv(s3_path)
        print(f"  Test Source 3: {len(s3_entities):,} entities loaded from {s3_path}")
        target_entities.update(s3_entities)
    except Exception as e:
        print(f"  Warning loading test_source3: {e}")

    # 5. Score Candidates & Apply Threshold
    print(f"\n[Step 4/5] Scoring candidate pairs with threshold = {threshold:.4f}...")
    matches_by_s1, total_passed = score_candidates_batch(
        candidate_pairs=flat_pairs,
        source1_entities=s1_entities,
        target_entities=target_entities,
        model=model,
        threshold=threshold,
        batch_size=args.batch_size,
        verbose=True,
    )
    print(f"Scoring complete. {total_passed:,} candidate pairs passed threshold ({threshold:.4f}).")

    # 6. Format Results & Verify Constraints
    print("\n[Step 5/5] Formatting matching results and validating constraints...")
    results = format_and_verify_results(
        all_source1_ids=all_s1_ids,
        matches_by_s1=matches_by_s1,
        candidate_map=cand_map,
    )

    out_file = write_matching_results(results, args.output)
    non_empty = sum(1 for _, m in results if m.strip())
    empty_cnt = len(results) - non_empty
    print(f"Successfully generated: {out_file}")
    print(f"  Total Source 1 entities: {len(results):,}")
    print(f"  Entities with matches:   {non_empty:,}")
    print(f"  Entities without match:  {empty_cnt:,} (singletons/no candidate passed threshold)")

    # Run official validate_submission.py
    if not args.skip_validator:
        validator_script = resolve_file_path("validate_submission.py", search_dirs=[".", "utils"], must_exist=False)
        if os.path.isfile(validator_script):
            print("\nRunning official submission validator (validate_submission.py)...")
            cmd = [
                sys.executable,
                validator_script,
                "--matching", out_file,
                "--candidate", cand_path,
                "--test-dir", os.path.dirname(s1_path) or ".",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
            if res.returncode == 0:
                print(">>> OFFICIAL VALIDATOR PASSED (Exit Code 0) <<<")
            else:
                print(f">>> Official validator exited with code {res.returncode} <<<")

    print("\n" + "=" * 70)
    print(f"PREDICTION COMPLETED! Final results saved at: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
