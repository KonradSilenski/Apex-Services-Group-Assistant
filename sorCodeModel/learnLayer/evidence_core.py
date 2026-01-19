# sorCodeModel/evidence_core.py
from __future__ import annotations

"""
Evidence & Feedback Core
------------------------

This module unifies three related components:

1) Operator feedback helpers (formerly operator_feedback.py)
   - Formatting predicted code×qty pairs
   - Parsing operator corrections
   - Validating SOR codes against the SOR CSV
   - Appending rows to feedback.csv

2) Evidence pack builder (formerly evidence_pack.py)
   - Derives a compact 'evidence pack' from form data + free text:
       * families, measurements, intents, flags, tile hints
       * text-derived quantity hints
   - Provides a similarity function between two packs
   - Provides a stable hash for caching (pack_hash)

3) Evidence snapshot builder (formerly evidence_snapshot.py)
   - Creates a compact, model-friendly snapshot of:
       * families, measurements, intents, text hints, flags
       * candidate code ranks/scores
       * hashes + text preview

All public functions keep their original names to minimise downstream changes.
"""

import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from sorCodeModel.pathsAndImports import (
    BM_SOR_CSV_PATH,
    BM_FEEDBACK_CSV_PATH,
    BM_TEXT_CACHE_PATH,
    BM_PACK_CACHE_PATH,
    BM_FORM_CACHE_PATH,
)

# =============================================================================
# 1) Feedback helpers (formatting, parsing, persistence)
#    (from operator_feedback.py)
# =============================================================================

def _coerce_predicted_dict(
    quantities_obj: Any,
    fallback_codes: Optional[Iterable[str]] = None
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if isinstance(quantities_obj, dict):
        for k, v in quantities_obj.items():
            code = str(k).strip()
            if not code.isdigit():
                continue
            qty = None
            if isinstance(v, (int, float)):
                qty = float(v)
            elif isinstance(v, dict):
                # inspect common keys first
                for key in ("qty", "quantity", "value", "amount"):
                    if key in v and isinstance(v[key], (int, float)):
                        qty = float(v[key])
                        break
                if qty is None:
                    # fall back to any numeric value
                    for val in v.values():
                        if isinstance(val, (int, float)):
                            qty = float(val)
                            break
            if qty is None:
                qty = 1.0
            out[code] = qty

    elif isinstance(quantities_obj, (list, tuple)):
        for item in quantities_obj:
            code = None
            qty = None
            if isinstance(item, dict):
                code = str(item.get("code") or item.get("sor") or "").strip()
                qty = item.get("qty") or item.get("quantity") or item.get("value")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                code = str(item[0]).strip()
                qty = item[1]
            if code and code.isdigit():
                try:
                    out[code] = float(qty) if isinstance(qty, (int, float)) else float(str(qty))
                except Exception:
                    out[code] = 1.0

    if not out and fallback_codes:
        for c in fallback_codes:
            out[str(c)] = 1.0

    return out


def format_predicted_pairs(
    quantities_obj: Any,
    codes_list: Optional[Iterable[str]] = None
) -> str:
    """
    Format a predicted quantities object (dict, list, etc.) into a friendly
    string like: "201303×1, 201307×4"
    """
    qmap = _coerce_predicted_dict(quantities_obj, fallback_codes=codes_list)
    parts = []
    for code in sorted(qmap.keys()):
        qty = qmap[code]
        qty_str = str(int(qty)) if abs(qty - int(qty)) < 1e-9 else f"{qty}"
        parts.append(f"{code}×{qty_str}")
    return ", ".join(parts)


_PAIR_RE = re.compile(r"^\s*([0-9]{6})\s*[xX]\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def parse_code_qty_pairs(s: str) -> Dict[str, float]:
    """
    Parse an operator string like:
      "201303x1, 201307x4"
    into {"201303":1.0, "201307":4.0}.

    Special commands:
      "same" -> {}
      "skip" -> raises ValueError("SKIP")
      "help" -> raises ValueError("HELP")
    """
    if s is None:
        return {}
    s = s.strip()
    if not s or s.lower() == "same":
        return {}
    if s.lower() == "skip":
        raise ValueError("SKIP")
    if s.lower() == "help":
        raise ValueError("HELP")

    result: Dict[str, float] = {}
    for raw in s.split(","):
        m = _PAIR_RE.match(raw.strip())
        if not m:
            raise ValueError(f"Could not parse pair: '{raw.strip()}'. Use format 201303x1, 201307x4")
        code, qty = m.group(1), float(m.group(2))
        result[code] = qty
    return result


def validate_codes_exist(codes: Iterable[str], sor_csv_path: str) -> Tuple[bool, list]:
    """
    Check that all given SOR codes exist in the SOR CSV.
    Returns (ok, unknown_codes_list).
    """
    existing = set()
    try:
        with open(sor_csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            code_field = "Code"
            if code_field not in reader.fieldnames:
                code_field = reader.fieldnames[0]
            for row in reader:
                c = str(row.get(code_field, "")).strip()
                if c:
                    existing.add(c)
    except Exception:
        # if we cannot open the CSV, don't block operator feedback
        return True, []

    unknown = [c for c in codes if c not in existing]
    return (len(unknown) == 0), unknown


def _ensure_feedback_csv(path: str) -> None:
    """
    Ensure feedback.csv exists with the correct header.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "visit_id",
                "job_id",
                "text_hash",
                "pack_hash",
                "predicted_json",
                "corrected_json",
                "notes",
            ])


def append_feedback_row(
    feedback_csv_path: str,
    visit_id: Optional[str],
    job_id: Optional[str],
    text_hash: Optional[str],
    pack_hash: Optional[str],
    predicted_dict: Dict[str, float],
    corrected_dict: Dict[str, float],
    notes: str = "",
) -> None:
    """
    Append a new row to the feedback CSV with predicted vs corrected codes.
    """
    _ensure_feedback_csv(feedback_csv_path)
    with open(feedback_csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            visit_id or "",
            job_id or "",
            text_hash or "",
            pack_hash or "",
            json.dumps(predicted_dict, ensure_ascii=False),
            json.dumps(corrected_dict, ensure_ascii=False),
            notes,
        ])


HELP_TEXT = """
Enter corrected codes & quantities as:  201303x1, 201307x4, 231011x6, 603903x2
Commands:
  same  -> accept model predictions as-is
  skip  -> skip saving feedback for this run
  help  -> show this help
"""


def prompt_for_corrections(
    predicted_quantities: Any,
    sor_csv_path: str,
    visit_id: Optional[str] = None,
    job_id: Optional[str] = None,
    text_hash: Optional[str] = None,
    pack_hash: Optional[str] = None,
    feedback_csv_path: str = BM_FEEDBACK_CSV_PATH,
    predicted_codes_list: Optional[Iterable[str]] = None,
) -> None:
    """
    Interactive console prompt. Safe to call from pipeline_demo after predictions.
    Records pack_hash so the memory layer can hydrate evidence packs later.
    """
    predicted_map = _coerce_predicted_dict(predicted_quantities, fallback_codes=predicted_codes_list)
    pred_str = format_predicted_pairs(predicted_quantities, codes_list=predicted_codes_list)

    print("\n— Operator Feedback —")
    print("Predicted (code×qty):")
    print(f"  {pred_str or '(no quantities)'}")

    while True:
        try:
            user = input("\nEnter corrections (or 'same', 'skip', 'help'): ").strip()
            parsed = parse_code_qty_pairs(user)
            if parsed == {}:
                corrected = predicted_map
            else:
                corrected = parsed

            ok, unknown = validate_codes_exist(corrected.keys(), sor_csv_path)
            if not ok:
                print(f"⚠️  Unknown SOR codes: {', '.join(unknown)}")
                resp = input("Re-enter? (y to re-enter / any other key to continue anyway): ").strip().lower()
                if resp == "y":
                    continue

            echo = ", ".join(
                f"{c}×{int(v) if abs(v - int(v)) < 1e-9 else v}"
                for c, v in corrected.items()
            )
            print(f"\nI heard: {echo}")
            yn = input("Save feedback? (y/n): ").strip().lower()
            if yn != "y":
                print("Not saved. You can re-enter or type 'skip' to exit.")
                continue

            append_feedback_row(
                feedback_csv_path=feedback_csv_path,
                visit_id=str(visit_id) if visit_id is not None else "",
                job_id=str(job_id) if job_id is not None else "",
                text_hash=text_hash or "",
                pack_hash=pack_hash or "",
                predicted_dict=predicted_map,
                corrected_dict=corrected,
                notes="",
            )
            print(f"✅ Saved to {feedback_csv_path}")
            return

        except ValueError as e:
            msg = str(e)
            if msg == "SKIP":
                print("⏭  Skipping feedback save.")
                return
            if msg == "HELP":
                print(HELP_TEXT)
                continue
            print(f"❌ {msg}")
            print("Type 'help' for examples.")
        except KeyboardInterrupt:
            print("\n⏭  Feedback cancelled by user.")
            return


# =============================================================================
# 2) Evidence pack (intent extraction, tile hints, quantity hints)
#    (from evidence_pack.py)
# =============================================================================

# Basic NLP helpers

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def _canon(s: str) -> str:
    return (s or "").strip().lower()


def _parse_float(x: Any) -> float:
    """
    Robust float coercion; accepts strings like '4', '4.5', ' 6 m ', 'approx 2.5m'
    Returns 0.0 if nothing numeric is present.
    """
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = (x or "").strip().lower().replace(",", "")
        if not s:
            return 0.0
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0


# Domain dictionaries (lite)

from typing import Set as _Set  # keep type alias clean in this section

PRIORITY_VERBS: _Set[str] = {
    "renew", "replace", "repoint", "rebuild", "remove", "rake", "rebed", "refix",
    "repair", "clean", "clear", "unblock", "jet", "wash",
}

VERB_ALIASES = {
    "clean": {
        "clean", "cleaned", "cleaning",
        "clear", "cleared", "clearing",
        "unblock", "unblocked", "unblocking",
        "jet", "jetted", "jetting",
        "wash", "washed", "washing",
    },
    "renew": {"renew", "renewed", "renewing", "replace", "replaced", "replacing"},
    "refix": {"refix", "refixed", "refixing", "repair", "repaired", "repairing"},
    "repoint": {"repoint", "repointed", "repointing"},
    "rebed": {"rebed", "rebedded", "rebedding"},
    "rebuild": {"rebuild", "rebuilt", "rebuilding"},
}

MATERIAL_ALIASES = {
    # lead / roof parts
    "flashing": {"flashing", "flashings", "abutment", "abutments", "step", "cover", "apron"},
    "apron": {"apron", "aprons"},
    "soaker": {"soaker", "soakers"},
    "valley": {"valley", "valleys"},
    "hip": {"hip", "hips"},
    "ridge": {"ridge", "ridges"},
    "verge": {"verge", "verges"},

    # rainwater goods
    "gutter": {"gutter", "gutters", "guttering"},
    "downpipe": {"downpipe", "downpipes", "rwp", "rainwater", "rain"},

    # tiles
    "tile": {"tile", "tiles"},
    "slate": {"slate", "slates"},
    "plain": {"plain"},
    "concrete": {"concrete"},
    "clay": {"clay"},
}

TILE_SUBTYPE_TOKENS = {
    "plain": {"plain"},
    "concrete": {"concrete"},
    "clay": {"clay"},
    "slate": {"slate", "slates"},
}


# Intent extraction

def _sentence_chunks(text: str) -> List[str]:
    if not text:
        return []
    return re.split(r"[.;\n]+", text)


def _normalize_verb(w: str) -> Optional[str]:
    wl = (w or "").lower()
    for base, forms in VERB_ALIASES.items():
        if wl in forms:
            return base
    return wl if wl in PRIORITY_VERBS else None


def _nearest_material(words: List[str], idx: int) -> Optional[str]:
    best = None
    best_dist = 9999
    flat_aliases = []
    for key, aliases in MATERIAL_ALIASES.items():
        for a in aliases:
            flat_aliases.append((key, a))
    for j, w in enumerate(words):
        for key, a in flat_aliases:
            if w == a:
                d = abs(j - idx)
                if d < best_dist:
                    best_dist = d
                    best = key
    return best


def _intent_pairs_from_text(text: str) -> List[Tuple[str, str]]:
    pairs = []
    for s in _sentence_chunks(text or ""):
        words = _tok(s)
        for i, w in enumerate(words):
            vb = _normalize_verb(w)
            if not vb:
                continue
            mat = _nearest_material(words, i)
            if mat:
                pairs.append((vb, mat))
    # de-dup preserve order
    seen = set()
    res = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            res.append(p)
    return res


# Tile hints

def _infer_tile_subtype(text_blob: str) -> Optional[str]:
    s = (text_blob or "").lower()
    for sub, toks in TILE_SUBTYPE_TOKENS.items():
        if any(t in s for t in toks):
            return sub
    return None


def _extract_tile_count_from_text(text_blob: str) -> Optional[int]:
    m = re.search(
        r"(replace|renew|refix|repair)\s+(?:approximately\s+)?(\d{1,3})\s+(tile|tiles|slate|slates)",
        (text_blob or "").lower(),
    )
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    return None


# Cross-field quantity hints (free-text)

RE_LM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|metre|meter|lm)\b", re.I)
RE_M2 = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m2|m²|sqm|square\s*met(?:er|re)s?)\b", re.I)
RE_EACH_GENERIC = re.compile(
    r"(\d+(?:\.\d+)?)\s*(tile|tiles|slate|slates|ridge|verge|valley|rwp|downpipe|downpipes|outlet|hopper)s?\b",
    re.I,
)

LM_CLASSIFIERS = [
    ("lead", {"lead", "flashing", "apron", "soaker", "abutment", "step", "cover"}),
    ("gutter", {"gutter", "guttering"}),
    ("ridge", {"ridge", "hip"}),
    ("verge", {"verge"}),
    ("valley", {"valley"}),
    ("fascia", {"fascia"}),
    ("soffit", {"soffit"}),
]

EACH_CLASSIFIERS = [
    ("tiles", {"tile", "tiles", "slate", "slates"}),
    ("ridge", {"ridge"}),
    ("verge", {"verge"}),
    ("valley", {"valley"}),
    ("rwp", {"rwp", "downpipe", "downpipes"}),
]


def _nearest_label_by_tokens(words: List[str], hit_index: int, classifier) -> Optional[str]:
    best_label = None
    best_d = 9999
    for label, toks in classifier:
        for i, w in enumerate(words):
            if w in toks:
                d = abs(i - hit_index)
                if d < best_d:
                    best_d = d
                    best_label = label
    return best_label


def _extract_text_quantity_hints(all_text_fields: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Scan all free-text fields for numbers + units and map them to semantic categories.
    Returns dict like:
      {
        "lead":   {"lm": 4.0},
        "gutter": {"lm": 6.0},
        "tiles":  {"each": 6.0},
        "coverings": {"m2": 12.0},
        ...
      }
    """
    hints: Dict[str, Dict[str, float]] = {}
    for field_name, text in (all_text_fields or {}).items():
        s = text or ""
        words = _tok(s)

        # Linear metres
        for m in RE_LM.finditer(s):
            val = float(m.group(1))
            idx = s[:m.start()].count(" ")  # rough token index
            label = _nearest_label_by_tokens(words, idx, LM_CLASSIFIERS) or "lead"
            hints.setdefault(label, {})
            prev = hints[label].get("lm", 0.0)
            hints[label]["lm"] = max(prev, val)

        # Square metres
        for m in RE_M2.finditer(s):
            val = float(m.group(1))
            label = "coverings"
            hints.setdefault(label, {})
            prev = hints[label].get("m2", 0.0)
            hints[label]["m2"] = max(prev, val)

        # EACH counts
        for m in RE_EACH_GENERIC.finditer(s):
            val = float(m.group(1))
            token = m.group(2).lower()
            idx = s[:m.start()].count(" ")
            label = _nearest_label_by_tokens(words, idx, EACH_CLASSIFIERS)
            if token in {"tile", "tiles", "slate", "slates"}:
                label = label or "tiles"
            elif token in {"rwp", "downpipe", "downpipes"}:
                label = label or "rwp"
            elif token in {"ridge"}:
                label = label or "ridge"
            elif token in {"verge"}:
                label = label or "verge"
            elif token in {"valley"}:
                label = label or "valley"
            else:
                label = label or "items"

            hints.setdefault(label, {})
            prev = hints[label].get("each", 0.0)
            hints[label]["each"] = max(prev, val)

    return hints


def build_evidence_pack(
    instance: Any,
    form_data: Dict[str, str],
    free_text_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Derives a compact, stable 'evidence pack' combining:
      - family presence/strength
      - measurements presence/values
      - tile subtype/count/small-job hints
      - intents (verb, material) across all free-text fields
      - elevation/scaffold/clean intents
      - text-derived quantity hints (LM/EACH/m²) across all free-text fields
      - fields_filled: numeric measurement fields with normalised values
    """
    # Aggregate a global free-text blob
    fields = list(free_text_map.keys())
    global_text = " ".join([free_text_map.get(f, "") or "" for f in fields]).strip()
    global_tokens = set(_tok(global_text))

    pack: Dict[str, Any] = {
        "families": {},
        "meas": {},
        "tile": {},
        "intents": [],
        "flags": {},
        "fields_present": [],
        "fields_filled": {},
        "text_quant_hints": {},
    }

    # --- Measurements pulled from structured fields ---

    # Tile measurement fields
    tile_meas_keys = [
        "slate_tile_measurement",
        "concrete_tile_measurement",
        "clay_tile_measurement",
        "sml_slate_tile_measurement",
        "sml_concrete_tile_measurement",
        "sml_clay_tile_measurement",
        "lg_slate_tile_measurement",
        "lg_concrete_tile_measurement",
        "lg_clay_tile_measurement",
    ]
    tile_area = 0.0
    for k in tile_meas_keys:
        if k in form_data and (form_data.get(k) or "").strip():
            v = _parse_float(form_data.get(k))
            if v:
                tile_area += v
                pack["fields_present"].append(k)
                pack["fields_filled"][k] = {"type": "AREA_M2", "value": v, "raw": form_data.get(k)}

    # Leadwork measurements
    lead_keys = [
        "lead_flashings_measurement",
        "leadwork_repoint_measurement",
        "leadwork_renew_measurement",
    ]
    lead_lm = 0.0
    for k in lead_keys:
        if k in form_data and (form_data.get(k) or "").strip():
            v = _parse_float(form_data.get(k))
            if v:
                lead_lm += v
                pack["fields_present"].append(k)
                pack["fields_filled"][k] = {"type": "LM", "value": v, "raw": form_data.get(k)}

    # Fascia / Soffit (LM if given)
    fascia_keys = ["pvc_fascia_measurement", "timber_fascia_measurement"]
    fascia_lm = 0.0
    for k in fascia_keys:
        if k in form_data and (form_data.get(k) or "").strip():
            v = _parse_float(form_data.get(k))
            if v:
                fascia_lm += v
                pack["fields_present"].append(k)
                pack["fields_filled"][k] = {"type": "LM", "value": v, "raw": form_data.get(k)}

    soffit_keys = ["pvc_soffit_measurement", "timber_soffit_measurement"]
    soffit_lm = 0.0
    for k in soffit_keys:
        if k in form_data and (form_data.get(k) or "").strip():
            v = _parse_float(form_data.get(k))
            if v:
                soffit_lm += v
                pack["fields_present"].append(k)
                pack["fields_filled"][k] = {"type": "LM", "value": v, "raw": form_data.get(k)}

    # Guttering / RWP presence (flags or implicit)
    rwp_fields = [
        "rwp_replacement",
        "rwp_refix",
        "cast_iron_rwp_replacement",
        "cast_iron_rwp_refix",
        "pvc_rwp_replacement",
        "pvc_rwp_refix",
    ]
    rwp_present = any(((form_data.get(k) or "").strip()) for k in rwp_fields if k in form_data)

    gutter_fields = [
        "guttering_replacement",
        "guttering_refit",
        "cast_iron_guttering_replacement",
        "cast_iron_guttering_refit",
        "pvc_guttering_replacement",
        "pvc_guttering_refit",
    ]
    gutter_present = any(((form_data.get(k) or "").strip()) for k in gutter_fields if k in form_data)

    # Families (SOFT via text; HARD via measurements)
    fams = {
        "leadwork": {"present": False, "strength": "NONE"},
        "tile": {"present": False, "strength": "NONE"},
        "guttering": {"present": False, "strength": "NONE"},
        "downpipe": {"present": False, "strength": "NONE"},
        "ridge": {"present": False, "strength": "NONE"},
        "chimney": {"present": False, "strength": "NONE"},
        "fascia": {"present": False, "strength": "NONE"},
        "soffit": {"present": False, "strength": "NONE"},
        "verge": {"present": False, "strength": "NONE"},
        "valley": {"present": False, "strength": "NONE"},
    }

    # soft text cues
    if {"lead", "flashing", "apron", "soaker", "abutment", "step", "cover"} & global_tokens:
        fams["leadwork"]["present"] = True
        fams["leadwork"]["strength"] = "SOFT"
    if {"tile", "slate", "concrete", "clay", "plain"} & global_tokens:
        fams["tile"]["present"] = True
        fams["tile"]["strength"] = "SOFT"
    if {"gutter", "guttering"} & global_tokens:
        fams["guttering"]["present"] = True
        fams["guttering"]["strength"] = "SOFT"
    if {"downpipe", "rwp"} & global_tokens:
        fams["downpipe"]["present"] = True
        fams["downpipe"]["strength"] = "SOFT"
    if "chimney" in global_tokens:
        fams["chimney"]["present"] = True
        fams["chimney"]["strength"] = "SOFT"
    if "ridge" in global_tokens or "hip" in global_tokens:
        fams["ridge"]["present"] = True
        fams["ridge"]["strength"] = "SOFT"
    if "verge" in global_tokens:
        fams["verge"]["present"] = True
        fams["verge"]["strength"] = "SOFT"
    if "valley" in global_tokens:
        fams["valley"]["present"] = True
        fams["valley"]["strength"] = "SOFT"
    if "fascia" in global_tokens:
        fams["fascia"]["present"] = True
        fams["fascia"]["strength"] = "SOFT"
    if "soffit" in global_tokens:
        fams["soffit"]["present"] = True
        fams["soffit"]["strength"] = "SOFT"

    # measurement upgrades to HARD
    if lead_lm > 0:
        fams["leadwork"]["present"] = True
        fams["leadwork"]["strength"] = "HARD"
    if tile_area > 0:
        fams["tile"]["present"] = True
        fams["tile"]["strength"] = "HARD"
    if rwp_present:
        fams["downpipe"]["present"] = True
        fams["downpipe"]["strength"] = "HARD"
    if gutter_present:
        fams["guttering"]["present"] = True
        fams["guttering"]["strength"] = "SOFT"  # keep soft unless numeric

    pack["families"] = fams
    pack["meas"] = {
        "lead_lm": lead_lm,
        "tile_area": tile_area,
        "fascia_lm": fascia_lm,
        "soffit_lm": soffit_lm,
        "gutter_flag": bool(gutter_present),
        "rwp_present": bool(rwp_present),
    }

    # Tile extras
    tile_subtype = _infer_tile_subtype(global_text)
    tile_count = _extract_tile_count_from_text(global_text)
    small_tile_job = False
    if tile_count is not None and tile_count <= 10:
        small_tile_job = True
    elif tile_area > 0 and tile_area <= 3.0:
        small_tile_job = True

    pack["tile"] = {
        "subtype_hint": tile_subtype or "",
        "count_hint": tile_count if tile_count is not None else "",
        "small_job": small_tile_job,
    }

    # Intents
    intents = set()
    for field, text in free_text_map.items():
        for vb, mat in _intent_pairs_from_text(text or ""):
            intents.add((vb, mat))
    pack["intents"] = sorted(list(intents))

    # Flags: elevation, scaffold, gutter-clean, valley vs verge
    tokens = global_tokens
    clean_gutter = (
        ("clean" in tokens or "clear" in tokens or "unblock" in tokens)
        and ("gutter" in tokens or "guttering" in tokens)
    )
    elevation_mentions = any(t in tokens for t in {"elevation", "elevations", "elev"})
    scaffold_mentions = any(t in tokens for t in {"scaffold", "scaffolding"})
    valley_present = "valley" in tokens or "valleys" in tokens
    verge_present = "verge" in tokens or "verges" in tokens

    pack["flags"] = {
        "clean_gutter_intent": clean_gutter,
        "elevation_mentions": elevation_mentions,
        "scaffold_mentions": scaffold_mentions,
        "valley_present": valley_present,
        "verge_present": verge_present,
    }

    # Text-derived quantity hints across all free-text fields
    pack["text_quant_hints"] = _extract_text_quantity_hints(free_text_map)

    return pack


# =============================================================================
# 3) Evidence similarity & hashing
# =============================================================================

def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _close_num(a: float, b: float, scale: float = 1.0) -> float:
    # returns [0..1], 1 if identical, falling with relative diff
    a = float(a or 0.0)
    b = float(b or 0.0)
    if a == b:
        return 1.0
    denom = max(scale, abs(a) + abs(b), 1.0)
    diff = abs(a - b) / denom
    return max(0.0, 1.0 - diff)


def _hints_similarity(
    ha: Dict[str, Dict[str, float]],
    hb: Dict[str, Dict[str, float]],
) -> float:
    """
    Compare text-derived quantity hints.
    Averaged across overlapping categories and units.
    """
    if not ha and not hb:
        return 1.0
    if not ha or not hb:
        return 0.0
    cats = set(ha.keys()) | set(hb.keys())
    if not cats:
        return 1.0
    scores = []
    for c in cats:
        ua = ha.get(c, {})
        ub = hb.get(c, {})
        units = set(ua.keys()) | set(ub.keys())
        if not units:
            continue
        unit_scores = []
        for u in units:
            unit_scores.append(_close_num(ua.get(u, 0.0), ub.get(u, 0.0), scale=10.0))
        scores.append(sum(unit_scores) / len(unit_scores))
    return sum(scores) / len(scores) if scores else 1.0


def evidence_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    Compute a similarity score [0..1] between two evidence packs.
    """
    if not a or not b:
        return 0.0

    fams_a = a.get("families", {})
    fams_b = b.get("families", {})
    fam_set_a = {k for k, v in fams_a.items() if v.get("present")}
    fam_set_b = {k for k, v in fams_b.items() if v.get("present")}
    sim_fam = _jaccard(fam_set_a, fam_set_b)

    intents_a = set(tuple(x) for x in a.get("intents", []))
    intents_b = set(tuple(x) for x in b.get("intents", []))
    sim_int = _jaccard(intents_a, intents_b)

    meas_a = a.get("meas", {})
    meas_b = b.get("meas", {})
    sim_lead = _close_num(meas_a.get("lead_lm", 0), meas_b.get("lead_lm", 0), scale=10.0)
    sim_tile = _close_num(meas_a.get("tile_area", 0), meas_b.get("tile_area", 0), scale=20.0)
    sim_fascia = _close_num(meas_a.get("fascia_lm", 0), meas_b.get("fascia_lm", 0), scale=20.0)
    sim_soffit = _close_num(meas_a.get("soffit_lm", 0), meas_b.get("soffit_lm", 0), scale=20.0)
    sim_rwp = 1.0 if bool(meas_a.get("rwp_present")) == bool(meas_b.get("rwp_present")) else 0.0
    sim_gut = 1.0 if bool(meas_a.get("gutter_flag")) == bool(meas_b.get("gutter_flag")) else 0.0

    tile_a = a.get("tile", {})
    tile_b = b.get("tile", {})
    sim_subtype = 1.0 if (tile_a.get("subtype_hint") == tile_b.get("subtype_hint")) else 0.0
    sim_small = 1.0 if bool(tile_a.get("small_job")) == bool(tile_b.get("small_job")) else 0.0

    flags_a = a.get("flags", {})
    flags_b = b.get("flags", {})
    sim_clean = 1.0 if bool(flags_a.get("clean_gutter_intent")) == bool(flags_b.get("clean_gutter_intent")) else 0.0
    sim_elev = 1.0 if bool(flags_a.get("elevation_mentions")) == bool(flags_b.get("elevation_mentions")) else 0.0
    sim_scaff = 1.0 if bool(flags_a.get("scaffold_mentions")) == bool(flags_b.get("scaffold_mentions")) else 0.0

    # text quantity hints
    sim_hints = _hints_similarity(a.get("text_quant_hints", {}), b.get("text_quant_hints", {}))

    # contradictions: valley vs verge
    penalty = 0.0
    if bool(flags_a.get("valley_present")) != bool(flags_b.get("valley_present")):
        penalty += 0.12
    if bool(flags_a.get("verge_present")) != bool(flags_b.get("verge_present")):
        penalty += 0.12

    # weighted blend
    w_fam, w_int, w_meas, w_tile, w_flags, w_hints = 0.28, 0.20, 0.22, 0.10, 0.10, 0.10
    sim_meas = (sim_lead + sim_tile + sim_fascia + sim_soffit + sim_rwp + sim_gut) / 6.0
    sim_tile_block = 0.6 * sim_subtype + 0.4 * sim_small
    sim_flags = (sim_clean + sim_elev + sim_scaff) / 3.0

    score = (
        w_fam * sim_fam +
        w_int * sim_int +
        w_meas * sim_meas +
        w_tile * sim_tile_block +
        w_flags * sim_flags +
        w_hints * sim_hints
    )
    score = max(0.0, score - penalty)
    return min(1.0, score)


def hash_pack(pack: Dict[str, Any]) -> str:
    """
    Stable hash of an evidence pack, used as pack_hash in caches.
    """
    try:
        payload = json.dumps(pack, sort_keys=True, ensure_ascii=False)
    except Exception:
        payload = str(pack)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# 4) Evidence snapshot (logging for training/diagnostics)
#    (from evidence_snapshot.py)
# =============================================================================

def _float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def build_evidence_snapshot(
    *,
    pack: Dict[str, Any],
    form: Dict[str, Any] | None,
    text_blob: str | None,
    predicted_map: Dict[str, float] | None,
    text_hash: str,
    pack_hash: str,
) -> Dict[str, Any]:
    """
    Build a compact, model-friendly "evidence snapshot" to log with each run, so
    we can later train/diagnose without re-running the whole pipeline.

    Captures (examples):
      - families + strengths
      - measurements
      - intents
      - text quantity hints
      - fields_filled
      - flags
      - candidate ranks/scores
      - hashes (text_hash, pack_hash)
      - short text_preview
    """
    pack = pack or {}
    form = form or {}
    predicted_map = predicted_map or {}

    # families with strength
    fams = {}
    for k, v in (pack.get("families") or {}).items():
        fams[k] = {
            "present": bool(v.get("present")),
            "strength": v.get("strength") or "NONE",
        }

    # measurements
    meas = pack.get("meas") or {}
    meas_out = {
        "lead_lm": _float(meas.get("lead_lm")),
        "tile_area": _float(meas.get("tile_area")),
        "fascia_lm": _float(meas.get("fascia_lm")),
        "soffit_lm": _float(meas.get("soffit_lm")),
        "rwp_present": bool(meas.get("rwp_present")),
        "gutter_flag": bool(meas.get("gutter_flag")),
    }

    # intents
    intents = pack.get("intents") or []

    # text quantity hints
    text_hints = pack.get("text_quant_hints") or {}

    # fields_filled (normalized numeric measurement fields with raw values)
    fields_filled = pack.get("fields_filled") or {}

    # flags
    flags = pack.get("flags") or {}

    # candidate ranks from predicted_map
    sorted_codes = sorted(predicted_map.keys(), key=lambda c: -_float(predicted_map.get(c, 0.0)))
    candidate_ranks: List[Dict[str, Any]] = []
    for i, code in enumerate(sorted_codes, start=1):
        candidate_ranks.append({
            "code": str(code),
            "rank": i,
            "score": _float(predicted_map.get(code, 0.0)),
        })

    return {
        "families": fams,
        "measurements": meas_out,
        "intents": intents,
        "text_quant_hints": text_hints,
        "fields_filled": fields_filled,
        "flags": flags,
        "candidate_ranks": candidate_ranks,
        "hashes": {"text_hash": text_hash, "pack_hash": pack_hash},
        "text_preview": (text_blob or "")[:240],
    }
