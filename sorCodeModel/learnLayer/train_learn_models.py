# sorCodeModel/learnLayer/train_learn_models.py
from __future__ import annotations

"""
Train learning-layer models
---------------------------

This module combines:

1) Re-ranker training
2) Quantity helper training
3) Orchestration via train_all_models()
"""

import csv
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error
import joblib
 
from sorCodeModel.pathsAndImports import (
    BM_FEEDBACK_CSV_PATH,
    BM_TEXT_CACHE_PATH,
    BM_PACK_CACHE_PATH,
    BM_FORM_CACHE_PATH,
    BM_LEARN_MODEL_DIR,
    BM_RERANKER_MODEL_PATH,
    BM_RERANKER_META_PATH,
    BM_QTY_SRC_MODEL_PATH,
    BM_QTY_MAG_MODEL_PATH,
    BM_QTY_META_PATH,
)

from sorCodeModel.learnLayer.learn_core import (
    build_features_for_candidate,
    features_to_vector,
    FEATURE_NAMES,
)
 
FEEDBACK   = BM_FEEDBACK_CSV_PATH
PACK_CACHE = BM_PACK_CACHE_PATH
FORM_CACHE = BM_FORM_CACHE_PATH
TEXT_CACHE = BM_TEXT_CACHE_PATH
 
MODEL_DIR = BM_LEARN_MODEL_DIR
 
RERANKER_MODEL_PATH = BM_RERANKER_MODEL_PATH
RERANKER_META_PATH  = BM_RERANKER_META_PATH
 
SRC_PATH      = BM_QTY_SRC_MODEL_PATH
MAG_PATH      = BM_QTY_MAG_MODEL_PATH
QTY_META_PATH = BM_QTY_META_PATH
 
SRC_LABELS = ["NONE", "LM", "EACH", "M2", "ELEV"]  


def _idx(label: str) -> int:
    try:
        return SRC_LABELS.index(label)
    except Exception:
        return 0


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _coerce_pred_map(obj) -> Dict[str, float]:
    """
    Coerce a JSON-decoded map of code->qty (or similar) into a clean
    { "123456": float(qty) } dict. Non-numeric values default to 1.0.
    """
    pm: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            code = str(k).strip()
            try:
                pm[code] = float(v)
            except Exception:
                pm[code] = 1.0
    return pm

# =============================================================================
# 1) Re-ranker training
# =============================================================================

def build_reranker_rows() -> Tuple[List[List[float]], List[int]]:
    """
    Build training rows for the re-ranker from feedback + caches.

    Returns:
      X: List of feature vectors
      y: List of labels (1 = in corrected set, 0 = not)
    """
    packs = _load_json(PACK_CACHE)
    forms = _load_json(FORM_CACHE)
    texts = _load_json(TEXT_CACHE)

    X: List[List[float]] = []
    y: List[int] = []

    if not os.path.exists(FEEDBACK):
        return X, y

    with open(FEEDBACK, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pack_hash = row.get("pack_hash") or ""
            text_hash = row.get("text_hash") or ""

            try:
                pred = _coerce_pred_map(json.loads(row.get("predicted_json") or "{}"))
            except Exception:
                pred = {}
            try:
                corr = _coerce_pred_map(json.loads(row.get("corrected_json") or "{}"))
            except Exception:
                corr = {}

            # If we have no predicted codes and no corrected codes, skip
            if not pred and not corr:
                continue

            pack = packs.get(pack_hash, {})
            form = forms.get(pack_hash, {})
            text_blob = texts.get(text_hash, "")

            # Candidate list = union(predicted codes, corrected codes)
            candidate_codes = sorted(set(list(pred.keys()) + list(corr.keys())))
            if not candidate_codes:
                continue

            # Basic pipeline signals: use predicted map as proxy for base score/rank
            ranks = {
                c: i + 1
                for i, c in enumerate(sorted(pred.keys(), key=lambda k: -pred[k]))
            }

            for c in candidate_codes:
                pipe_signals = {
                    "score": pred.get(c, 0.0),
                    "rank": float(ranks.get(c, len(candidate_codes) + 1)),
                    # Semantic/rule signals could be logged separately in the future; use zeros now
                    "sem_score": 0.0,
                    "rule_bonus": 0.0,
                }
                feats = build_features_for_candidate(
                    c,
                    pack=pack,
                    form=form,
                    text_blob=text_blob,
                    pipeline_signals=pipe_signals,
                )
                X.append(features_to_vector(feats, FEATURE_NAMES))
                y.append(1 if c in corr else 0)

    return X, y


def train_reranker_and_save() -> Dict[str, Any]:
    """
    Train the re-ranker model and save it to disk.

    Returns:
      meta: dict containing feature_names, auc, rows.
    """
    X, y = build_reranker_rows()
    if not X:
        raise RuntimeError("No training rows built from feedback. Add rows via pipeline and try again.")

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_np,
        y_np,
        test_size=0.2,
        random_state=42,
        stratify=y_np,
    )

    clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)
    clf.fit(X_tr, y_tr)

    # AUC for sanity
    try:
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    except Exception:
        auc = None

    _ensure_dir(MODEL_DIR)
    joblib.dump(clf, RERANKER_MODEL_PATH)

    meta = {
        "feature_names": FEATURE_NAMES,
        "auc": auc,
        "rows": int(len(X_np)),
    }
    with open(RERANKER_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# =============================================================================
# 2) Quantity helper training
# =============================================================================

def _infer_source_label(pack: Dict[str, Any], feats: Dict[str, float]) -> str:
    """
    Simple heuristic for inferring which source is likely responsible for the qty:
      - leads/lead hints → LM
      - tiles/slates hints → EACH or M2
      - coverings hints → M2
      - elevation mentions → ELEV
      - otherwise NONE

    This is only used to create noisy targets for the qty_source classifier.
    """
    meas = pack.get("meas", {}) or {}

    # Leads → LM
    if (meas.get("lead_lm") or 0) > 0 or feats.get("hint_lead_lm", 0) > 0:
        return "LM"

    # Tiles / slates → EACH (if explicit count)
    if feats.get("hint_tiles_each", 0) > 0:
        return "EACH"

    # Areas → M2 (tile_area or coverings)
    if (meas.get("tile_area") or 0) > 0 or feats.get("hint_coverings_m2", 0) > 0:
        return "M2"

    # Explicit elevation flags → ELEV
    if feats.get("flag_elevation_mentions", 0) > 0:
        return "ELEV"

    return "NONE"


def build_qty_rows() -> Tuple[List[List[float]], List[int], List[List[float]], List[float]]:
    """
    Build training rows for the quantity helper models.

    Returns:
      Xsrc, ysrc: source classifier rows/labels
      Xmag, ymag: magnitude regressor rows/targets
    """
    packs = _load_json(PACK_CACHE)
    forms = _load_json(FORM_CACHE)
    texts = _load_json(TEXT_CACHE)

    Xsrc: List[List[float]] = []
    ysrc: List[int] = []
    Xmag: List[List[float]] = []
    ymag: List[float] = []

    if not os.path.exists(FEEDBACK):
        return Xsrc, ysrc, Xmag, ymag

    with open(FEEDBACK, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pack_hash = row.get("pack_hash") or ""
            text_hash = row.get("text_hash") or ""
            try:
                corr = _coerce_pred_map(json.loads(row.get("corrected_json") or "{}"))
            except Exception:
                corr = {}
            if not corr:
                continue

            pack = packs.get(pack_hash, {})
            form = forms.get(pack_hash, {})
            text_blob = texts.get(text_hash, "")

            # For each positive code, build features and targets
            for code, qty in corr.items():
                feats = build_features_for_candidate(
                    code,
                    pack=pack,
                    form=form,
                    text_blob=text_blob,
                    pipeline_signals={
                        "score": 0.0,
                        "rank": 0.0,
                        "sem_score": 0.0,
                        "rule_bonus": 0.0,
                    },
                )
                vec = features_to_vector(feats, FEATURE_NAMES)

                # Source label
                src_label = _infer_source_label(pack, feats)
                Xsrc.append(vec)
                ysrc.append(_idx(src_label))

                # Magnitude target
                try:
                    ymag.append(float(qty))
                    Xmag.append(vec)
                except Exception:
                    # if qty can't be coerced, skip for magnitude
                    pass

    return Xsrc, ysrc, Xmag, ymag


def train_qty_models_and_save() -> Dict[str, Any]:
    """
    Train the qty_source classifier and qty_magnitude regressor (if enough data),
    and save them to disk.

    Returns:
      meta: dict containing feature_names, src_labels, src_f1, mag_mae, rows_source, rows_mag.
    """
    Xsrc, ysrc, Xmag, ymag = build_qty_rows()

    if not Xsrc:
        # No data at all for source; nothing to train
        raise RuntimeError("No quantity training rows built from feedback. Add rows via pipeline and try again.")

    _ensure_dir(MODEL_DIR)

    # --- Source classifier ---
    Xs = np.array(Xsrc, dtype=np.float32)
    ys = np.array(ysrc, dtype=np.int32)

    # stratify only if we have at least 2 classes
    stratify = ys if len(set(ys.tolist())) > 1 else None

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        Xs,
        ys,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    clf = LogisticRegression(max_iter=200, multi_class="auto")
    clf.fit(Xs_tr, ys_tr)
    src_f1 = f1_score(ys_te, clf.predict(Xs_te), average="macro")
    joblib.dump(clf, SRC_PATH)

    # --- Magnitude regressor (optional) ---
    mag_mae = None
    mag_rows = 0
    if len(Xmag) >= 20:
        Xm = np.array(Xmag, dtype=np.float32)
        ym = np.array(ymag, dtype=np.float32)
        Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(
            Xm,
            ym,
            test_size=0.2,
            random_state=42,
        )
        reg = Ridge(alpha=1.0)
        reg.fit(Xm_tr, ym_tr)
        mag_mae = float(mean_absolute_error(ym_te, reg.predict(Xm_te)))
        mag_rows = int(len(Xm))
        joblib.dump(reg, MAG_PATH)

    meta = {
        "feature_names": FEATURE_NAMES,
        "src_labels": SRC_LABELS,
        "src_f1": float(src_f1),
        "mag_mae": mag_mae,
        "rows_source": int(len(Xs)),
        "rows_mag": mag_rows,
    }
    with open(QTY_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# =============================================================================
# 3) Orchestration
# =============================================================================

def train_all_models() -> Dict[str, Any]:
    """
    Convenience entrypoint to train both the re-ranker and the quantity helpers.

    Returns:
      {
        "reranker": {...},
        "qty": {...}
      }
    """
    rerank_meta = train_reranker_and_save()
    qty_meta = train_qty_models_and_save()
    return {
        "reranker": rerank_meta,
        "qty": qty_meta,
    }


if __name__ == "__main__":
    info = train_all_models()
    print(f"Saved re-ranker to {RERANKER_MODEL_PATH}")
    print(f"Saved qty models to {SRC_PATH} (source) and {MAG_PATH} (magnitude, if trained)")
    print(json.dumps(info, indent=2))
