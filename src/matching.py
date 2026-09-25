"""Matching orchestrator: scores candidate pairs, applies threshold, and produces matching_results.tsv.

Enforces all challenge constraints:
1. Every Source 1 entity appears exactly once.
2. matched_entity_ids contains only valid S2- and S3- prefixed IDs.
3. No duplicate IDs per entity.
4. Empty string if no candidates pass the threshold.
5. Every matched pair is verified against candidate_pairs.tsv.
6. Memory-efficient batch scoring for large candidate sets.
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.data_loader import DELIM
from src.feature_engineering import (
    ProcessedEntity,
    build_entity_cache,
    compute_features_matrix,
)
from src.matching_model import MatchingClassifier


def score_candidates_batch(
    candidate_pairs: List[Tuple[str, str]],
    source1_entities: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]],
    target_entities: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]],
    model: MatchingClassifier,
    threshold: float,
    batch_size: int = 50000,
    verbose: bool = True,
) -> Tuple[Dict[str, List[Tuple[str, float]]], int]:
    """Score candidate pairs in batches and collect pairs passing threshold.
    
    Returns:
        (matches_by_s1: {s1_id: [(cand_id, prob), ...]}, total_passed_count)
    """
    if verbose:
        print(f"Preparing entity caches for {len(candidate_pairs)} candidate pairs...")

    # Only cache entities that actually appear in candidate pairs to conserve memory
    needed_s1 = {pair[0] for pair in candidate_pairs}
    needed_target = {pair[1] for pair in candidate_pairs}

    sub_s1 = {eid: source1_entities[eid] for eid in needed_s1 if eid in source1_entities}
    sub_target = {eid: target_entities[eid] for eid in needed_target if eid in target_entities}

    s1_cache = build_entity_cache(sub_s1)
    target_cache = build_entity_cache(sub_target)

    if verbose:
        print(f"Entity cache built ({len(s1_cache)} S1, {len(target_cache)} S2/S3). Starting batch scoring...")

    matches_by_s1: Dict[str, List[Tuple[str, float]]] = {}
    total_passed = 0
    n_pairs = len(candidate_pairs)

    for start_idx in range(0, n_pairs, batch_size):
        end_idx = min(start_idx + batch_size, n_pairs)
        batch_pairs = candidate_pairs[start_idx:end_idx]

        # Extract features
        X_batch = compute_features_matrix(batch_pairs, s1_cache, target_cache)

        # Predict match probabilities
        probs = model.predict_proba(X_batch)

        # Filter by threshold
        passing_indices = np.where(probs >= threshold)[0]
        for idx in passing_indices:
            s1_id, cand_id = batch_pairs[idx]
            prob = float(probs[idx])
            if s1_id not in matches_by_s1:
                matches_by_s1[s1_id] = []
            matches_by_s1[s1_id].append((cand_id, prob))
            total_passed += 1

        if verbose and (end_idx % 200000 == 0 or end_idx == n_pairs):
            print(f"  Processed {end_idx:,}/{n_pairs:,} pairs... {total_passed:,} matches passed threshold ({threshold:.3f})")

    return matches_by_s1, total_passed


def format_and_verify_results(
    all_source1_ids: Set[str],
    matches_by_s1: Dict[str, List[Tuple[str, float]]],
    candidate_map: Optional[Dict[str, Set[str]]] = None,
) -> List[Tuple[str, str]]:
    """Format matching results and verify all challenge rules.
    
    Returns:
        List of (source1_entity_id, comma_separated_matched_entity_ids)
    """
    results: List[Tuple[str, str]] = []
    seen_s1: Set[str] = set()

    for s1_id in sorted(all_source1_ids):
        if s1_id in seen_s1:
            raise ValueError(f"Duplicate source1_entity_id found: {s1_id}")
        seen_s1.add(s1_id)

        raw_matches = matches_by_s1.get(s1_id, [])
        if not raw_matches:
            results.append((s1_id, ""))
            continue

        # Sort matches by probability descending, then candidate ID
        raw_matches.sort(key=lambda item: (-item[1], item[0]))

        # Deduplicate candidate IDs while preserving order
        deduped_cands: List[str] = []
        seen_cand_ids: Set[str] = set()
        allowed_candidates = candidate_map.get(s1_id, set()) if candidate_map is not None else None

        for cand_id, _ in raw_matches:
            if cand_id in seen_cand_ids:
                continue
            
            # Validation: Candidate must start with S2- or S3-
            if not cand_id.startswith(("S2-", "S3-")):
                raise ValueError(
                    f"Invalid match ID '{cand_id}' for '{s1_id}'. Must start with 'S2-' or 'S3-'."
                )
            
            # Validation: Self-match check
            if cand_id.startswith("S1-"):
                raise ValueError(f"Self-match error: Source 1 ID '{cand_id}' matched with '{s1_id}'.")

            # Validation: Must exist in candidate pairs
            if allowed_candidates is not None and cand_id not in allowed_candidates:
                raise ValueError(
                    f"Match ID '{cand_id}' for '{s1_id}' was NOT present in candidate_pairs.tsv."
                )

            seen_cand_ids.add(cand_id)
            deduped_cands.append(cand_id)

        results.append((s1_id, ",".join(deduped_cands)))

    # Ensure every single S1 entity is present
    missing_s1 = all_source1_ids - seen_s1
    if missing_s1:
        raise ValueError(f"Missing {len(missing_s1)} Source 1 entities in matching results!")

    return results


def write_matching_results(
    results: List[Tuple[str, str]],
    output_path: str = "output/matching_results.tsv",
) -> str:
    """Write verified matching results to TSV file.
    
    Format:
    source1_entity_id \\t matched_entity_ids
    """
    resolved_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(resolved_output), exist_ok=True)

    with open(resolved_output, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id{DELIM}matched_entity_ids\n")
        for s1_id, matched_str in results:
            f.write(f"{s1_id}{DELIM}{matched_str}\n")

    return resolved_output
