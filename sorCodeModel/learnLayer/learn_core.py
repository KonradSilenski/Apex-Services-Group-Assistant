# sorCodeModel/learnLayer/learn_core.py
from __future__ import annotations

"""
Learning core utilities
-----------------------

This module unifies five related components:

1) Quantity schema builder
   - Infers expected units of measure (UOM) and keyword sets for each SOR code.

2) Feature builder
   - Builds feature dictionaries & vectors for (job, candidate_code) pairs using:
       * Evidence Pack
       * Full form
       * Free text
       * Pipeline signals (semantic/rule scores, ranks, etc.)

3) Memory index
   - Lightweight similarity memory over operator feedback rows, using:
       * Optional SentenceTransformer embeddings of text
       * Evidence-pack similarity

4) Quantity advisor
   - Loads quantity helper models (if enabled) and produces per-code hints:
       {
         "<code>": {
           "prefer_source": "LM|EACH|M2|ELEV|NONE",
           "quantity_prior": float|None
         }
       }

5) Reranker
   - Optional second-stage scorer to adjust candidate ordering using learned
     features. If the model is missing or disabled, it is a clean no-op.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable, Tuple, Set
import csv
import hashlib
import json
import os
import re

import numpy as np

# Optional deps for memory + models
try:  # pragma: no cover - optional deps
    import torch
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except Exception:  # pragma: no cover
    _HAS_ST = False

try:  # pragma: no cover
    import joblib
    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    _HAS_JOBLIB = False

# Centralised paths (no .env here)
from sorCodeModel.pathsAndImports import (
    BM_LEARN_MODEL_DIR,
    BM_QTY_SRC_MODEL_PATH,
    BM_QTY_MAG_MODEL_PATH,
    BM_QTY_META_PATH,
    BM_RERANKER_MODEL_PATH,
    BM_RERANKER_META_PATH,
)

# Evidence similarity lives in evidence_core
from sorCodeModel.learnLayer.evidence_core import evidence_similarity


# =============================================================================
# 1) Quantity schema builder
# =============================================================================

CANONICAL_UOMS: Set[str] = {"EACH", "PAIR", "SET", "M", "M2", "M3", "HOUR", "DAY", "OTHER"}

UNIT_TOKENS: List[Tuple[str, str]] = [
    # area
    (r"\b(m2|sqm|sq\s*m|square\s*metre|square\s*meter)\b", "M2"),
    # length
    (r"\b(per\s*m|per\s*metre|per\s*meter|m|metre|meter|lm|linear\s*m)\b", "M"),
    # volume
    (r"\b(m3|cubic\s*metre|cubic\s*meter)\b", "M3"),
    # time
    (r"\b(hour|hr|hrs|day|per\s*hour|per\s*day)\b", "HOUR"),
    # itemised
    (r"\b(each|item|unit|pair|set|per\s*item|per\s*unit)\b", "EACH"),
]

HARDWARE_TOKENS: List[str] = [
    "bracket", "brackets", "offset", "offsets",
    "outlet", "outlets", "stop end", "stop ends", "stop-end", "stopend",
    "clip", "clips", "union", "unions", "hanger", "hangers",
]

RWP_TOKENS: List[str] = [
    "rwp", "downpipe", "down pipe", "downspout", "down spout", "dp",
]

GUTTER_TOKENS: List[str] = [
    "gutter", "guttering",
]

CATEGORY_FALLBACKS: List[Tuple[List[str], str]] = [
    (["lead", "flashing", "soaker", "apron"], "M"),
    (["felt", "membrane", "underlay"], "M2"),
    (["tile", "slate"], "EACH"),
    (["cladding", "soffit", "fascia"], "M"),
    (["inspection", "survey", "callout"], "EACH"),
]


def _infer_uom(desc: str, job_type: str, element: str, category: str) -> str:
    blob = " ".join([desc, job_type, element, category]).lower()

    # Direct unit tokens
    for pattern, uom in UNIT_TOKENS:
        if re.search(pattern, blob):
            return uom

    # Guesstimate by category / keywords
    for keywords, uom in CATEGORY_FALLBACKS:
        if any(k in blob for k in keywords):
            return uom

    # Gutters / RWPs default to length or item depending on wording
    if any(tok in blob for tok in GUTTER_TOKENS):
        return "M"
    if any(tok in blob for tok in RWP_TOKENS):
        return "EACH"

    return "EACH"


def build_quantity_schema_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        code: {
          'expected_uom': canonical UOM,
          'keywords': [list of keywords for linking],
          'category','element','job_type','description'
        }
      }
    """
    schema: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("Code") or "").strip()
            if not code:
                continue
            desc = (row.get("Medium Description") or "").strip()
            job_type = (row.get("Job Type") or "").strip()
            element = (row.get("Element") or "").strip()
            category = (row.get("Work Categories") or "").strip()

            uom = _infer_uom(desc, job_type, element, category)

            # build keyword set
            keywords = set()
            blob = " ".join([desc, job_type, element, category]).lower()
            for token in re.findall(r"[a-z]+", blob):
                keywords.add(token)

            # add synonyms/families to help matching
            synonyms = {
                "lead": ["lead", "flashing", "soaker", "apron"],
                "felt": ["felt", "underlay", "membrane"],
                "gutter": ["gutter", "guttering"],
                "rwp": ["rwp", "downpipe", "down", "downspout", "dp", "down", "pipe"],
                "tile": ["tile", "tiles", "slate", "slates"],
                "cladding": ["cladding", "shiplap", "board"],
                "soffit": ["soffit", "soffits"],
                "fascia": ["fascia", "fascias"],
                "insulation": ["insulation", "kingspan", "quilt"],
                "hardware": HARDWARE_TOKENS,
            }
            for fam_words in synonyms.values():
                if any(w in blob for w in fam_words):
                    keywords.update(fam_words)

            schema[code] = {
                "expected_uom": uom if uom in CANONICAL_UOMS else "OTHER",
                "keywords": list(keywords),
                "category": category.lower(),
                "element": element.lower(),
                "job_type": job_type.lower(),
                "description": desc,
            }
    return schema


# =============================================================================
# 2) Feature builder
# =============================================================================

def _get(d: Dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


def _bool(x) -> int:
    return 1 if bool(x) else 0


def _float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _intent_has(pack: Dict[str, Any], verb: str, material: str) -> int:
    ints = pack.get("intents", []) or []
    for vb, mat in ints:
        if vb == verb and mat == material:
            return 1
    return 0


def _family_present(pack: Dict[str, Any], fam: str, strength: str | None = None) -> int:
    fams = pack.get("families", {}) or {}
    entry = fams.get(fam, {}) or {}
    if not entry.get("present"):
        return 0
    if strength and entry.get("strength") != strength:
        return 0
    return 1


def _quant_hint(pack: Dict[str, Any], cat: str, unit: str) -> float:
    hints = pack.get("text_quant_hints", {}) or {}
    return _float(_get(hints, [cat, unit], 0.0))


# Keep this list stable. New features should be appended to avoid breaking old models.
FEATURE_NAMES: List[str] = [
    # Pipeline signals (to be provided by caller)
    "pipe_score",             # base score assigned by current pipeline
    "pipe_rank",              # 1.K rank (lower is better)
    "pipe_sem_score",         # semantic score if available
    "pipe_rule_bonus",        # aggregated rule bonus if available

    # Families (from Evidence Pack)
    "fam_lead_soft", "fam_lead_hard",
    "fam_tile_soft", "fam_tile_hard",
    "fam_gutter", "fam_downpipe",
    "fam_ridge", "fam_valley", "fam_verge",
    "fam_fascia", "fam_soffit",
    "fam_chimney",

    # Measurements (numeric)
    "lead_lm", "tile_area", "fascia_lm", "soffit_lm",
    "rwp_present", "gutter_flag",

    # Tile hints
    "tile_small_job", "tile_subtype_concrete", "tile_subtype_clay", "tile_subtype_slate",

    # Intents (examples; generic enough)
    "intent_renew_flashing", "intent_repoint_flashing", "intent_clean_gutter",
    "intent_refix_tile", "intent_renew_tile",

    # Text quantity hints (numeric)
    "hint_lead_lm", "hint_gutter_lm", "hint_ridge_lm", "hint_valley_lm", "hint_verge_lm",
    "hint_tiles_each", "hint_rwp_each", "hint_coverings_m2",

    # Generic flags
    "flag_scaffold", "flag_elevation_mentions",
]


def build_features_for_candidate(
    candidate_code: str,
    *,
    pack: Dict[str, Any],
    form: Dict[str, Any] | None = None,
    text_blob: str | None = None,
    pipeline_signals: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Returns a flat dict matching FEATURE_NAMES order."""
    pipe = pipeline_signals or {}
    fam = pack.get("families", {}) or {}
    meas = pack.get("meas", {}) or {}
    tile = pack.get("tile", {}) or {}
    flags = pack.get("flags", {}) or {}

    subtype = (tile.get("subtype_hint") or "").lower()

    feats = {
        # Pipeline
        "pipe_score": _float(pipe.get("score")),
        "pipe_rank": _float(pipe.get("rank")),
        "pipe_sem_score": _float(pipe.get("sem_score")),
        "pipe_rule_bonus": _float(pipe.get("rule_bonus")),

        # Families
        "fam_lead_soft":  _family_present(pack, "leadwork", "SOFT"),
        "fam_lead_hard":  _family_present(pack, "leadwork", "HARD"),
        "fam_tile_soft":  _family_present(pack, "tile", "SOFT"),
        "fam_tile_hard":  _family_present(pack, "tile", "HARD"),
        "fam_gutter":     _family_present(pack, "guttering"),
        "fam_downpipe":   _family_present(pack, "downpipe"),
        "fam_ridge":      _family_present(pack, "ridge"),
        "fam_valley":     _family_present(pack, "valley"),
        "fam_verge":      _family_present(pack, "verge"),
        "fam_fascia":     _family_present(pack, "fascia"),
        "fam_soffit":     _family_present(pack, "soffit"),
        "fam_chimney":    _family_present(pack, "chimney"),

        # Measurements
        "lead_lm":   _float(_get(meas, ["lead", "lm"], 0.0)),
        "tile_area": _float(_get(meas, ["tiles", "area_m2"], 0.0)),
        "fascia_lm": _float(_get(meas, ["fascia", "lm"], 0.0)),
        "soffit_lm": _float(_get(meas, ["soffit", "lm"], 0.0)),
        "rwp_present": _bool(_get(meas, ["rwp", "present"], False)),
        "gutter_flag": _bool(_get(meas, ["gutter", "flag"], False)),

        # Tile hints
        "tile_small_job":          _bool(tile.get("small_job")),
        "tile_subtype_concrete":   1 if subtype == "concrete" else 0,
        "tile_subtype_clay":       1 if subtype == "clay" else 0,
        "tile_subtype_slate":      1 if subtype == "slate" else 0,

        # Intents (generic examples)
        "intent_renew_flashing":   _intent_has(pack, "renew", "leadwork"),
        "intent_repoint_flashing": _intent_has(pack, "repoint", "leadwork"),
        "intent_clean_gutter":     _intent_has(pack, "clean", "gutter"),
        "intent_refix_tile":       _intent_has(pack, "refix", "tile"),
        "intent_renew_tile":       _intent_has(pack, "renew", "tile"),

        # Text quantity hints
        "hint_lead_lm":      _quant_hint(pack, "lead", "lm"),
        "hint_gutter_lm":    _quant_hint(pack, "gutter", "lm"),
        "hint_ridge_lm":     _quant_hint(pack, "ridge", "lm"),
        "hint_valley_lm":    _quant_hint(pack, "valley", "lm"),
        "hint_verge_lm":     _quant_hint(pack, "verge", "lm"),
        "hint_tiles_each":   _quant_hint(pack, "tiles", "each"),
        "hint_rwp_each":     _quant_hint(pack, "rwp", "each"),
        "hint_coverings_m2": _quant_hint(pack, "coverings", "m2"),

        # Generic flags
        "flag_scaffold":           1 if flags.get("scaffold_mentions") else 0,
        "flag_elevation_mentions": 1 if flags.get("elevation_mentions") else 0,
    }

    return feats


def features_to_vector(feats: Dict[str, float], order: List[str] | None = None) -> List[float]:
    names = order or FEATURE_NAMES
    return [float(feats.get(n, 0.0)) for n in names]


# =============================================================================
# 3) Memory index
# =============================================================================

@dataclass
class MemoryItem:
    key: str
    text_hash: str
    text_blob: str
    codes_qty: Dict[str, float]
    evidence_hash: str = ""
    evidence_pack: Optional[Dict] = None
    embedding: Optional[object] = None  # torch.tensor when available


class MemoryIndex:
    """
    Minimal in-RAM memory with optional embeddings using SentenceTransformer.
    Backed by an optional CSV feedback file + sidecar caches for text & packs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.items: List[MemoryItem] = []
        self.model = None
        self.device = "cuda" if _HAS_ST and torch.cuda.is_available() else "cpu"
        self.model_name = model_name

    def _ensure_model(self):
        if _HAS_ST and self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    # ---- Build / load ----

    def add_memory(
        self,
        text_blob: str,
        codes_qty: Dict[str, float],
        key: Optional[str] = None,
        evidence_pack: Optional[Dict] = None,
        evidence_hash: str = "",
    ) -> MemoryItem:
        self._ensure_model()
        text_hash = self._hash_text(text_blob)
        key = key or text_hash
        emb = None
        if _HAS_ST and self.model is not None and text_blob:
            emb = self.model.encode(text_blob, convert_to_tensor=True)
        item = MemoryItem(
            key=key,
            text_hash=text_hash,
            text_blob=text_blob or "",
            codes_qty=dict(codes_qty),
            evidence_hash=evidence_hash or "",
            evidence_pack=evidence_pack,
            embedding=emb,
        )
        self.items.append(item)
        return item

    def build_from_feedback_csv(self, feedback_csv_path: str) -> int:
        """
        Load corrected_json (+ optional text_hash/pack_hash) from operator feedback and index it.
        Text and evidence packs can be hydrated later via hydrate_*() calls.
        """
        if not os.path.exists(feedback_csv_path):
            return 0
        count = 0
        with open(feedback_csv_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                corrected_json = row.get("corrected_json") or "{}"
                text_hash = row.get("text_hash") or ""
                pack_hash = row.get("pack_hash") or ""
                try:
                    codes_qty = json.loads(corrected_json)
                    if isinstance(codes_qty, dict) and codes_qty:
                        # Normalise keys defensively (e.g. '"221001' -> '221001')
                        norm: Dict[str, float] = {}
                        for k, v in codes_qty.items():
                            nk = _normalize_code_key(k)
                            if not str(nk).strip():
                                continue
                            try:
                                fv = float(v)
                            except Exception:
                                continue
                            # merge collisions by keeping the larger qty
                            if nk in norm:
                                norm[nk] = max(norm[nk], fv)
                            else:
                                norm[nk] = fv
                        codes_qty = norm

                        self.items.append(
                            MemoryItem(
                                key=text_hash or pack_hash or str(len(self.items)),
                                text_hash=text_hash,
                                text_blob="",
                                codes_qty=codes_qty,
                                evidence_hash=pack_hash or "",
                                evidence_pack=None,
                                embedding=None,
                            )
                        )
                        count += 1
                except Exception:
                    continue
        return count

    def hydrate_embeddings(self, text_map: Dict[str, str]) -> int:
        """Attach text & compute embeddings for items that lack them."""
        self._ensure_model()
        done = 0
        for it in self.items:
            if (not it.text_blob) and it.text_hash and it.text_hash in text_map:
                it.text_blob = text_map[it.text_hash]
            if it.embedding is None and it.text_blob and _HAS_ST and self.model is not None:
                it.embedding = self.model.encode(it.text_blob, convert_to_tensor=True)
                done += 1
        return done

    def hydrate_packs(self, pack_map: Dict[str, Dict]) -> int:
        """Attach evidence packs by evidence_hash if missing."""
        done = 0
        for it in self.items:
            if it.evidence_pack is None and it.evidence_hash and it.evidence_hash in pack_map:
                it.evidence_pack = pack_map[it.evidence_hash]
                done += 1
        return done

    # ---- Query ----

    def query_similar(
        self,
        query_text: str,
        pack: Optional[Dict] = None,
        k: int = 3,
        w_text: float = 0.6,
        w_ev: float = 0.4,
        min_sim: float = 0.73,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Returns up to k similar items using a combined score:
          sim = w_text * cosine(text) + w_ev * evidence_similarity(pack)
        If embeddings or packs are missing, the corresponding term is skipped.
        """
        if not self.items:
            return []

        # compute text embedding if possible
        q_emb = None
        if _HAS_ST and query_text and self.model is None:
            self._ensure_model()
        if _HAS_ST and self.model is not None and query_text:
            q_emb = self.model.encode(query_text, convert_to_tensor=True)

        hits: List[Tuple[MemoryItem, float]] = []
        for it in self.items:
            s_text = 0.0
            s_ev = 0.0
            has_any = False

            if q_emb is not None and it.embedding is not None:
                try:
                    s_text = float(util.pytorch_cos_sim(q_emb, it.embedding).item())
                    has_any = True
                except Exception:
                    s_text = 0.0

            if pack is not None and it.evidence_pack is not None:
                try:
                    s_ev = float(evidence_similarity(pack, it.evidence_pack))
                    has_any = True
                except Exception:
                    s_ev = 0.0

            if not has_any:
                continue

            wt, we = w_text, w_ev
            if q_emb is None or it.embedding is None:
                wt, we = 0.0, 1.0
            if pack is None or it.evidence_pack is None:
                wt, we = 1.0, 0.0

            score = wt * s_text + we * s_ev
            if score >= min_sim:
                hits.append((it, score))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]


# ---- Memory helpers / singleton access ----

_MEMORY_INDEX_SINGLETON: Optional[MemoryIndex] = None


def _env_memory_enabled() -> bool:
    # Feature flag still allowed to come from environment; we just don't
    # auto-load any .env file here.
    return os.environ.get("SOR_ENABLE_MEMORY", "0") == "1"


def get_memory_index(
    feedback_csv_path: str,
    text_cache_path: str | None = None,
    pack_cache_path: str | None = None,
    min_rows: int = 25,
) -> Optional[MemoryIndex]:
    """
    Builds (or returns a cached) MemoryIndex, hydrating it from:

      - feedback_csv_path   (operator corrections / gold labels)
      - text_cache_path     (optional, keyed by text_hash)
      - pack_cache_path     (optional, keyed by pack_hash)

    Returns None if memory is disabled or there is insufficient data.
    """
    global _MEMORY_INDEX_SINGLETON

    if not _env_memory_enabled():
        return None

    if _MEMORY_INDEX_SINGLETON is not None:
        return _MEMORY_INDEX_SINGLETON

    idx = MemoryIndex()
    n = idx.build_from_feedback_csv(feedback_csv_path)
    if n < min_rows:
        # Not enough data yet; don't enable memory.
        _MEMORY_INDEX_SINGLETON = None
        return None

    # --- Hydrate embeddings (if cache is present) ---
    if text_cache_path and os.path.exists(text_cache_path):
        try:
            with open(text_cache_path, "r", encoding="utf-8") as f:
                text_map_raw = json.load(f)
            if isinstance(text_map_raw, dict):
                text_map = {str(k): str(v or "") for k, v in text_map_raw.items()}
                idx.hydrate_embeddings(text_map)
        except Exception:
            # Non-fatal; we still have a purely symbolic memory
            pass

    # --- Hydrate evidence packs (if cache is present) ---
    if pack_cache_path and os.path.exists(pack_cache_path):
        try:
            with open(pack_cache_path, "r", encoding="utf-8") as f:
                pack_map_raw = json.load(f)
            if isinstance(pack_map_raw, dict):
                idx.hydrate_packs(pack_map_raw)  # type: ignore[arg-type]
        except Exception:
            pass

    _MEMORY_INDEX_SINGLETON = idx
    return idx


def apply_memory_layer(
    codes: List[str],
    *,
    text_blob: str,
    evidence_pack: Optional[Dict[str, Any]],
    memory_index: Optional[MemoryIndex],
    top_k: int = 3,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Lightweight, read-only memory layer.

    For now this does NOT change the code list; it only:
      - queries the memory index for similar jobs
      - returns the same codes + a metadata dict containing logs and neighbor
        snapshots.

    This gives us visibility into what memory *would* do, without risking
    silent behaviour changes. Once we are happy with the signal, we can
    start adding "add / replace / qty tweak" logic on top.
    """
    logs: List[str] = []

    # If env says "off" or we couldn't load memory, bail out early.
    if not _env_memory_enabled() or memory_index is None:
        logs.append("[memory] disabled or unavailable; returning baseline codes unchanged.")
        meta = {
            "enabled": False,
            "neighbors": [],
        }
        return codes, {"logs": logs, **meta}

    text_blob = (text_blob or "").strip()
    if not text_blob:
        logs.append("[memory] empty text blob; skipping memory lookup.")
        meta = {
            "enabled": False,
            "neighbors": [],
        }
        return codes, {"logs": logs, **meta}

    # Query memory – we use both text and evidence_pack if available.
    try:
        neighbors = memory_index.query_similar(
            query_text=text_blob,
            pack=evidence_pack,
            k=top_k,
        )
    except Exception as e:
        logs.append(f"[memory] query failed ({e}); skipping.")
        meta = {
            "enabled": False,
            "neighbors": [],
        }
        return codes, {"logs": logs, **meta}

    neighbor_snapshots: List[Dict[str, Any]] = []
    for item, score in neighbors:
        neighbor_snapshots.append(
            {
                "key": item.key,
                "score": float(score),
                "codes_qty": item.codes_qty,
            }
        )

    if neighbor_snapshots:
        logs.append(
            "[memory] top neighbors: "
            + " ; ".join(
                f"{n['key']}@{n['score']:.3f}" for n in neighbor_snapshots
            )
        )
    else:
        logs.append("[memory] no neighbors returned from index.")

    meta = {
        "enabled": True,
        "neighbors": neighbor_snapshots,
    }
    # IMPORTANT: we return the original `codes` unchanged for now.
    return codes, {"logs": logs, **meta}


# =============================================================================
# 4) Quantity advisor
# =============================================================================

# Central model paths from pathsAndImports
SRC_PATH = BM_QTY_SRC_MODEL_PATH
MAG_PATH = BM_QTY_MAG_MODEL_PATH
QTY_META_PATH = BM_QTY_META_PATH

_DEFAULT_SRC_LABELS = ["NONE", "LM", "EACH", "M2", "ELEV"]  # must match training script order


class QtyAdvisor:
    """
    Loads quantity helper models (if enabled) and produces per-code hints:
      {"<code>": {"prefer_source": "LM|EACH|M2|ELEV|NONE", "quantity_prior": float|None}}

    Controlled by env flag: SOR_ENABLE_QTY_HINTS=1

    If the models are missing, joblib is unavailable, or anything fails,
    this class quietly returns empty hints and the pipeline continues.
    """

    def __init__(self):
        self.enabled = os.environ.get("SOR_ENABLE_QTY_HINTS", "0") == "1"
        self.ready = False
        self.src_model = None
        self.mag_model = None
        self.feature_names: List[str] = FEATURE_NAMES
        self.src_labels: List[str] = _DEFAULT_SRC_LABELS
        if self.enabled:
            self._maybe_load()

    def _maybe_load(self):
        if not _HAS_JOBLIB:
            self.ready = False
            return
        if not os.path.exists(SRC_PATH):
            self.ready = False
            return
        try:
            self.src_model = joblib.load(SRC_PATH)
            if os.path.exists(MAG_PATH):
                self.mag_model = joblib.load(MAG_PATH)
            if os.path.exists(QTY_META_PATH):
                with open(QTY_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.feature_names = meta.get("feature_names", self.feature_names)
                self.src_labels = meta.get("src_labels", self.src_labels)
            self.ready = True
        except Exception:
            self.ready = False

    def is_ready(self) -> bool:
        return self.enabled and self.ready

    def get_hints(
        self,
        candidates: List[str],
        *,
        pack: Dict[str, Any],
        form: Dict[str, Any] | None,
        text_blob: str | None,
        base_scores: Dict[str, float] | None = None,
        sem_scores: Dict[str, float] | None = None,
        rule_bonuses: Dict[str, float] | None = None,
    ) -> Dict[str, Dict[str, Optional[float] | str]]:
        """
        Returns a hints dict per candidate code.
        Example:
          {
            "231011": {"prefer_source": "LM", "quantity_prior": 6.0},
            "603903": {"prefer_source": "EACH", "quantity_prior": 2.0},
            .
          }
        """
        hints: Dict[str, Dict[str, Optional[float] | str]] = {}
        if not self.is_ready():
            return hints

        base_scores = base_scores or {}
        sem_scores = sem_scores or {}
        rule_bonuses = rule_bonuses or {}

        # ranks derived from base scores (descending)
        sorted_codes = sorted(candidates, key=lambda c: -float(base_scores.get(c, 0.0)))
        ranks = {c: i + 1 for i, c in enumerate(sorted_codes)}

        X = []
        order: List[str] = []
        for code in candidates:
            pipe_signals = {
                "score": float(base_scores.get(code, 0.0)),
                "rank": float(ranks.get(code, len(candidates) + 1)),
                "sem_score": float(sem_scores.get(code, 0.0)),
                "rule_bonus": float(rule_bonuses.get(code, 0.0)),
            }
            feats = build_features_for_candidate(
                code,
                pack=pack,
                form=form,
                text_blob=text_blob,
                pipeline_signals=pipe_signals,
            )
            X.append(features_to_vector(feats, self.feature_names))
            order.append(code)

        if not X:
            return hints

        X_np = np.array(X, dtype=float)

        # Source prediction
        try:
            src_probs = self.src_model.predict_proba(X_np)  # shape (N, C)
            src_idx = src_probs.argmax(axis=1)
            src_labels = [
                self.src_labels[i] if 0 <= i < len(self.src_labels) else "NONE"
                for i in src_idx
            ]
        except Exception:
            return hints

        # Magnitude prior (optional)
        mag_preds = None
        if self.mag_model is not None:
            try:
                mag_preds = self.mag_model.predict(X_np)  # shape (N,)
            except Exception:
                mag_preds = None

        for i, code in enumerate(order):
            label = src_labels[i]
            q = None
            if mag_preds is not None:
                try:
                    q = float(mag_preds[i])
                except Exception:
                    q = None
            hints[code] = {"prefer_source": label, "quantity_prior": q}

        return hints


# Convenience singleton (lazy load on first use)
_qty_advisor_singleton: QtyAdvisor | None = None


def get_qty_advisor() -> QtyAdvisor:
    global _qty_advisor_singleton
    if _qty_advisor_singleton is None:
        _qty_advisor_singleton = QtyAdvisor()
    return _qty_advisor_singleton


# =============================================================================
# 5) Reranker (optional second-stage scorer)
# =============================================================================

# Central model paths
RERANKER_MODEL_PATH = BM_RERANKER_MODEL_PATH
RERANKER_META_PATH = BM_RERANKER_META_PATH


class Reranker:
    """
    Optional second-stage reranker for candidate codes.

    Controlled by env flag: SOR_ENABLE_RERANKER=1

    If joblib or the .pkl/meta files are missing, or the model call fails,
    this class safely degenerates to a no-op:

      - returns the original candidate ordering
      - returns a score dict based on base_scores (or a simple fallback)
    """

    def __init__(self):
        self.enabled = os.environ.get("SOR_ENABLE_RERANKER", "0") == "1"
        self.ready = False
        self.model = None
        self.feature_names: List[str] = FEATURE_NAMES
        if self.enabled:
            self._maybe_load()

    def _maybe_load(self):
        if not _HAS_JOBLIB:
            self.ready = False
            return
        if not os.path.exists(RERANKER_MODEL_PATH):
            self.ready = False
            return
        try:
            self.model = joblib.load(RERANKER_MODEL_PATH)
            if os.path.exists(RERANKER_META_PATH):
                with open(RERANKER_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.feature_names = meta.get("feature_names", self.feature_names)
            self.ready = True
        except Exception:
            self.ready = False

    def is_ready(self) -> bool:
        return self.enabled and self.ready

    def rerank(
        self,
        candidates: List[str],
        *,
        pack: Dict[str, Any],
        form: Dict[str, Any] | None,
        text_blob: str | None,
        base_scores: Dict[str, float] | None = None,
        sem_scores: Dict[str, float] | None = None,
        rule_bonuses: Dict[str, float] | None = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Returns (new_order, score_dict).

        - If the model is unavailable or disabled, returns the original
          candidate list plus a simple score dict derived from base_scores.
        - If the model is available, uses it to compute scores and sort.
        """
        if not candidates:
            return [], {}

        base_scores = base_scores or {}
        sem_scores = sem_scores or {}
        rule_bonuses = rule_bonuses or {}

        # Quick no-op path if not ready
        if not self.is_ready():
            scores = {c: float(base_scores.get(c, 0.0)) for c in candidates}
            # If everything is zero, give a simple descending rank score
            if not any(scores.values()):
                for i, c in enumerate(candidates):
                    scores[c] = float(len(candidates) - i)
            return candidates, scores

        # ranks derived from base scores (descending)
        sorted_by_base = sorted(candidates, key=lambda c: -float(base_scores.get(c, 0.0)))
        ranks = {c: i + 1 for i, c in enumerate(sorted_by_base)}

        X: List[List[float]] = []
        order: List[str] = []
        for code in candidates:
            pipe_signals = {
                "score": float(base_scores.get(code, 0.0)),
                "rank": float(ranks.get(code, len(candidates) + 1)),
                "sem_score": float(sem_scores.get(code, 0.0)),
                "rule_bonus": float(rule_bonuses.get(code, 0.0)),
            }
            feats = build_features_for_candidate(
                code,
                pack=pack,
                form=form,
                text_blob=text_blob,
                pipeline_signals=pipe_signals,
            )
            X.append(features_to_vector(feats, self.feature_names))
            order.append(code)

        if not X:
            scores = {c: float(base_scores.get(c, 0.0)) for c in candidates}
            return candidates, scores

        X_np = np.array(X, dtype=float)

        # Infer scores from model with a robust fallback
        try:
            model = self.model
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_np)
                if probs.shape[1] > 1:
                    raw = probs[:, 1]
                else:
                    raw = probs[:, 0]
            elif hasattr(model, "decision_function"):
                raw = model.decision_function(X_np)
            else:
                raw = model.predict(X_np)
            raw = [float(s) for s in raw]
        except Exception:
            # If anything goes wrong, do not break the pipeline.
            scores = {c: float(base_scores.get(c, 0.0)) for c in candidates}
            return candidates, scores

        scores = {code: raw[i] for i, code in enumerate(order)}
        new_order = sorted(candidates, key=lambda c: -scores.get(c, 0.0))
        return new_order, scores


# Convenience singleton for reranker
_reranker_singleton: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = Reranker()
    return _reranker_singleton
def _normalize_code_key(code: str) -> str:
    """Canonicalise a SOR code key to digits-only to avoid duplicates like '"221001'."""
    s = str(code).strip()
    s = s.strip('"''"“”‘’"''')
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits or s


