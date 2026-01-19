
# ---------------------------------------------------------------------------
# Helpers: code normalization + SOR description lookup (for material gating)
# ---------------------------------------------------------------------------
from __future__ import annotations
import re as _re
from functools import lru_cache as _lru_cache
# sorCodeModel/training_ops.py

# comment
import csv
import json
import os
import math
from typing import Dict, Optional, Any, List, Tuple

# NOTE:
# We no longer load .env files here. All path configuration is centralised in
# sorCodeModel.pathsAndImports. apiModule.py is the only place allowed to deal
# with .env directly.

from sorCodeModel.pathsAndImports import (
    BM_SOR_CSV_PATH,
    BM_FEEDBACK_CSV_PATH,
    BM_TEXT_CACHE_PATH,
    BM_PACK_CACHE_PATH,
    BM_FORM_CACHE_PATH,
)

from sorCodeModel.mainModelLayer.formProcessing import DetailsResponse, flatten_survey_data
from sorCodeModel.mainModelLayer.quantity_resolver import resolve_quantities
from sorCodeModel.mainModelLayer.theCondensedFunction import (
    get_sor_codes_from_instance,
    get_sor_codes_and_quantities_from_instance,
)
from sorCodeModel.learnLayer.evidence_core import (
    build_evidence_pack,
    hash_pack,
    build_evidence_snapshot,
)
from sorCodeModel.learnLayer.learn_core import (
    MemoryIndex,
    get_memory_index,
    apply_memory_layer,
)

def _normalise_code_key(code: Any) -> str:
    """Normalize SOR code keys so we never end up with variants like '"221001'."""
    if code is None:
        return ""
    s = str(code).strip()
    if not s:
        return ""
    # strip surrounding quotes/backticks
    s = s.strip('"\'`')
    # collapse internal whitespace
    s = "".join(s.split())
    return s

@_lru_cache(maxsize=8)
def _load_sor_desc_map(sor_csv_path: str) -> Dict[str, str]:
    """Best-effort map: code -> lowercased concatenated description fields."""
    m: Dict[str, str] = {}
    try:
        import csv as _csv
        with open(sor_csv_path, "r", encoding="utf-8", errors="ignore") as _f:
            r = _csv.DictReader(_f)
            for row in r:
                c = _normalise_code_key(row.get("Code") or row.get("code") or row.get("SOR") or row.get("sor") or "")
                if not c:
                    continue
                desc = " ".join([
                    str(row.get("Short Description") or ""),
                    str(row.get("Medium Description") or ""),
                    str(row.get("Element") or ""),
                ]).strip().lower()
                if desc:
                    m[c] = desc
    except Exception:
        return {}
    return m

def _sor_desc_has(sor_desc_map: Dict[str, str], code: str, needle: str) -> bool:
    d = sor_desc_map.get(code, "")
    return bool(d) and needle.lower() in d



# Canonical inspection code used by the job matcher.
INSPECTION_CODE = "221001"


def _is_inspection_only_visit(instance: DetailsResponse) -> bool:
    """Best-effort check for 'Inspection Only' visits.

    We keep this intentionally tolerant because upstream values can vary
    (e.g. "Inspection Only", "inspection only", "Inspection").
    """
    vt = (getattr(instance, "visit_type", None) or "").strip().lower()
    if not vt:
        return False
    return ("inspection" in vt) and ("only" in vt)

# ---------------------------------------------------------------------------
# Centralised default paths (no .env lookup here)
# ---------------------------------------------------------------------------

SOR_CSV_PATH_DEFAULT      = BM_SOR_CSV_PATH
FEEDBACK_CSV_PATH_DEFAULT = BM_FEEDBACK_CSV_PATH
TEXT_CACHE_PATH_DEFAULT   = BM_TEXT_CACHE_PATH
PACK_CACHE_PATH_DEFAULT   = BM_PACK_CACHE_PATH
FORM_CACHE_PATH_DEFAULT   = BM_FORM_CACHE_PATH


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _get_instance(visit_id: int) -> tuple[dict, DetailsResponse, list]:
    """
    Thin wrapper around your extractor. Keeps training_ops independent.
    """
    try:
        from apiModuleCode.apiModule import extract_instance_from_visit
    except Exception as e:
        raise RuntimeError(
            "The function 'extract_instance_from_visit' was not found. "
            "Please ensure 'apiModuleCode/apiModule.py' is available and callable."
        ) from e

    extra_details, instance, images = extract_instance_from_visit(
        visit_id, save_dir="apiModuleCode/", local_id="75e7764f-ed46-4749-996a-c57eb6c95d37"
    )
    return extra_details, instance, images


def _collect_free_text(instance: DetailsResponse) -> dict:
    return {
        "1.2_Work_Description": instance.work_desc or "",
        "6.3_Leadwork_Comment": getattr(instance, "leadwork_comment", "") or "",
        "7.3_Chimney_Comment": getattr(instance, "chimney_comment", "") or "",
        "7.5_Chimney_Flaunch_Comment": getattr(instance, "chimney_comment", "") or "",
        "11.1_Other_Works_Completed": getattr(instance, "other_works_completed", "") or "",
        "11.2_Other_Works_Needed": getattr(instance, "other_works_needed", "") or "",
        "13.2_Issues_Comments": getattr(instance, "issues_comments", "") or "",
        # Ensure we include 13.1 for the future-quantity pass
        "13.1_Issues_Present": getattr(instance, "issues_present", "") or "",
    }


def _hash_free_text(instance: DetailsResponse) -> str:
    import hashlib

    ft = _collect_free_text(instance)
    chunks = []
    for k in sorted(ft.keys()):
        chunks.append(f"{k}={ft[k]}")
    blob = "\n".join(chunks)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def parse_code_qty_pairs(s: str) -> Dict[str, float]:
    """
    Parses "201303x1, 201307x4, 231011x6" -> {"201303": 1.0, "201307": 4.0, "231011": 6.0}
    """
    out: Dict[str, float] = {}
    for part in (s or "").split(","):
        t = part.strip()
        if not t:
            continue
        if "x" in t:
            code, qty = t.split("x", 1)
        elif "X" in t:
            code, qty = t.split("X", 1)
        else:
            code, qty = t, "1"
        code = code.strip()
        try:
            out[code] = float(qty.strip())
        except Exception:
            out[code] = 1.0
    return out


def _ensure_feedback_header(path: str, fieldnames: list[str]) -> None:
    """If file doesn't exist, create with header. If header is missing new fields,
    we rewrite it with the union of previous and new fieldnames."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        existing_fieldnames = rdr.fieldnames

    if not existing_fieldnames:
        # Empty or corrupt header, rewrite fresh.
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return

    missing = [f for f in fieldnames if f not in existing_fieldnames]
    if not missing:
        return

    # Need to rewrite with the union of old + new fieldnames.
    new_fieldnames = list(existing_fieldnames) + missing
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _append_feedback_row_with_evidence(
    *,
    feedback_csv_path: str,
    visit_id: str,
    job_id: str,
    text_hash: str,
    pack_hash: str,
    predicted_dict: Dict[str, float],
    corrected_dict: Dict[str, float],
    evidence_json: str,
    notes: str = "",
) -> None:
    """
    Appends a new row to the feedback CSV with the given values. Ensures
    header has correct fields before writing.
    """
    fieldnames = [
        "timestamp",
        "visit_id",
        "job_id",
        "text_hash",
        "pack_hash",
        "predicted_json",
        "corrected_json",
        "evidence_json",
        "notes",
    ]
    _ensure_feedback_header(feedback_csv_path, fieldnames)

    import datetime as _dt
    ts = _dt.datetime.utcnow().isoformat()

    row = {
        "timestamp": ts,
        "visit_id": visit_id,
        "job_id": job_id,
        "text_hash": text_hash,
        "pack_hash": pack_hash,
        "predicted_json": json.dumps(predicted_dict, ensure_ascii=False),
        "corrected_json": json.dumps(corrected_dict, ensure_ascii=False),
        "evidence_json": evidence_json,
        "notes": notes or "",
    }

    with open(feedback_csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)


def _load_json_file(path: str) -> Dict[str, Any]:
    """Safe JSON loader; returns {} if file missing or invalid."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Feedback update API
# ---------------------------------------------------------------------------

def update_feedback_for_visit(
    visit_id: int | str,
    corrected_codes_str: str,
    feedback_csv_path: str = FEEDBACK_CSV_PATH_DEFAULT
) -> bool:
    """
    Idempotently updates the 'corrected_json' column for a given visit_id in the
    feedback CSV. Does not alter other rows. Returns True if a row was updated.
    """
    visit_id_str = str(visit_id).strip()
    corrected_map = parse_code_qty_pairs(corrected_codes_str)

    if not os.path.exists(feedback_csv_path):
        return False

    with open(feedback_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        fieldnames = rdr.fieldnames or [
            "timestamp", "visit_id", "job_id", "text_hash", "pack_hash",
            "predicted_json", "corrected_json", "evidence_json", "notes"
        ]

    idx_to_update: Optional[int] = None
    for i, r in enumerate(rows):
        if (r.get("visit_id") or "") == visit_id_str:
            idx_to_update = i  # last match wins

    if idx_to_update is None:
        return False

    rows[idx_to_update]["corrected_json"] = json.dumps(corrected_map, ensure_ascii=False)

    tmp_path = feedback_csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp_path, feedback_csv_path)
    return True


# ---------------------------------------------------------------------------
# Memory layer application
# ---------------------------------------------------------------------------

def _extract_qty_like(val: Any, fallback: float = 1.0) -> float:
    """Normalise qty from either a raw number or {'qty': .} style dict."""
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return fallback
    if isinstance(val, dict):
        for key in ("qty", "quantity", "value"):
            if key in val:
                try:
                    return float(val[key])
                except Exception:
                    return fallback
    try:
        return float(val)
    except Exception:
        return fallback


def _apply_memory_layer(
    *,
    codes: List[str],
    predicted_map: Dict[str, float],
    instance: DetailsResponse,
    pack: Dict[str, Any],
    form_data: Dict[str, Any],
    free_text_map: Dict[str, str],
    text_blob: str,
    sor_csv_path: str,
    feedback_csv_path: str,
) -> None:
    """
    Use MemoryIndex to adjust codes + quantities based on past corrected jobs.

    Policy (per Joanne's spec):
      - 0.70–0.85: allow adding new codes.
      - 0.85–0.95: add codes, and treat overlaps as "reinforced" (no qty change yet).
      - >0.95: as above, plus allow bumping quantities up towards memory qty.
      - For any newly added codes, run resolve_quantities so quantities reflect
        this job's text, not blindly copied from memory.

    Mutates `codes` and `predicted_map` in place.
    """
    # Build memory from feedback CSV
    if not os.path.exists(feedback_csv_path):
        return

    mem = MemoryIndex()
    loaded = mem.build_from_feedback_csv(feedback_csv_path)
    if loaded == 0:
        return

    text_cache = _load_json_file(TEXT_CACHE_PATH_DEFAULT)
    pack_cache = _load_json_file(PACK_CACHE_PATH_DEFAULT)
    mem.hydrate_embeddings(text_cache)
    mem.hydrate_packs(pack_cache)

    hits = mem.query_similar(
        query_text=text_blob,
        pack=pack,
        k=5,
        w_text=0.6,
        w_ev=0.4,
        min_sim=0.7,
    )
    if not hits:
        return

    print(f"[MEMORY] {len(hits)} similar past jobs found (min_sim=0.70).")

    # Nearest-neighbour similarity (top hit). Used for qty-advisor gating.
    try:
        nearest_sim = float(max(sim for (_item, sim) in hits))
    except Exception:
        nearest_sim = 0.0

    # If we have an effectively identical match, treat memory as authoritative for codes+qty.
    # This prevents qty rules/defaults or weak advisors from overwriting a known-correct correction.
    try:
        top_item, top_sim = hits[0]
        if float(top_sim) >= 0.999:
            mem_codes = getattr(top_item, "codes_qty", None) or {}
            authoritative: Dict[str, float] = {}
            for k, v in mem_codes.items():
                ck = _normalise_code_key(k)
                if not ck:
                    continue
                authoritative[ck] = float(_extract_qty_like(v, fallback=1.0))
            # Ensure inspection code is present for inspection visits
            try:
                if "inspection" in str(getattr(instance, "visit_type", "")).lower():
                    authoritative.setdefault(INSPECTION_CODE, 1.0)
            except Exception:
                pass
            # Replace current prediction
            predicted_map.clear()
            predicted_map.update(authoritative)
            codes.clear()
            codes.extend(list(authoritative.keys()))
            print(f"[MEMORY] Exact match (sim={top_sim:.3f}) ⇒ using memory codes as authoritative ({len(authoritative)} codes).")
            return
    except Exception:
        pass

    # --- Collect candidates from memory hits ---
    original_codes = set(str(c) for c in codes)
    # --- Material signals (used to avoid PVC memory codes overriding cast iron jobs) ---
    t_low = (text_blob or "").lower()
    prefer_cast_iron = ("cast iron" in t_low or "cast-iron" in t_low or "castiron" in t_low) and ("pvc" not in t_low and "upvc" not in t_low and "plastic" not in t_low)
    sor_desc_map = _load_sor_desc_map(sor_csv_path)

    # Reranker/top-candidate protection list (best-effort; available in evidence_pack)
    reranker_top5: set = set()
    try:
        cr = (pack or {}).get("candidate_ranks") or []
        if cr and isinstance(cr, list):
            if all(isinstance(x, dict) and "rank" in x for x in cr):
                cr_sorted = sorted(cr, key=lambda d: int(d.get("rank", 999999)))
            else:
                cr_sorted = sorted(cr, key=lambda d: float(d.get("score", 0.0)), reverse=True)
            for d in cr_sorted[:5]:
                c = _normalise_code_key(d.get("code"))
                if c:
                    reranker_top5.add(c)
    except Exception:
        reranker_top5 = set()


    # code -> memory qty fallback (from the best similarity hit that contains the code)
    added_codes: Dict[str, float] = {}
    best_sim_for_code: Dict[str, float] = {}
    strong_qty_overrides: Dict[str, float] = {}    # for >0.95 matches

    for item, sim in hits:
        mem_codes = item.codes_qty or {}
        print(f"[MEMORY] Hit key={item.key} sim={sim:.3f} with {len(mem_codes)} codes.")
        for code_raw, mem_qty_val in mem_codes.items():
            code = _normalise_code_key(code_raw)
            if not code:
                continue
            mem_qty = _extract_qty_like(mem_qty_val, fallback=1.0)
            # Material gate: if job is explicitly cast iron, do not import PVC/uPVC/plastic-only codes from memory.
            if prefer_cast_iron and (_sor_desc_has(sor_desc_map, code, "pvc") or _sor_desc_has(sor_desc_map, code, "upvc") or _sor_desc_has(sor_desc_map, code, "plastic")) and not _sor_desc_has(sor_desc_map, code, "cast iron"):
                continue

            if sim < 0.70:
                continue

            # 0.70–0.85: allow adding codes only
            if 0.70 <= sim < 0.85:
                if code not in predicted_map:
                    prev_sim = best_sim_for_code.get(code, -1.0)
                    if sim >= prev_sim:
                        best_sim_for_code[code] = float(sim)
                        added_codes[code] = float(mem_qty)
                    print(f"[MEMORY] (0.70-0.85) consider adding code {code} (mem_qty={mem_qty}).")
                continue

            # 0.85–0.95: add codes + reinforce overlaps
            if 0.85 <= sim < 0.95:
                if code not in predicted_map:
                    prev_sim = best_sim_for_code.get(code, -1.0)
                    if sim >= prev_sim:
                        best_sim_for_code[code] = float(sim)
                        added_codes[code] = float(mem_qty)
                    print(f"[MEMORY] (0.85-0.95) consider adding code {code} (mem_qty={mem_qty}).")
                else:
                    print(f"[MEMORY] (0.85-0.95) reinforced existing code {code}.")
                continue

            # >0.95: treat as very strong match; add codes and allow bumping qty
            if sim >= 0.95:
                if code not in predicted_map:
                    prev_sim = best_sim_for_code.get(code, -1.0)
                    if sim >= prev_sim:
                        best_sim_for_code[code] = float(sim)
                        added_codes[code] = float(mem_qty)
                    print(f"[MEMORY] (>0.95) consider adding code {code} (mem_qty={mem_qty}).")
                else:
                    # mark as strong qty override candidate
                    prev = strong_qty_overrides.get(code, 0.0)
                    strong_qty_overrides[code] = max(prev, mem_qty)
                    print(f"[MEMORY] (>0.95) strong qty signal for code {code} (mem_qty={mem_qty}).")

    # If we have no candidates, nothing to do
    if not added_codes and not strong_qty_overrides:
        return

    # --- Quantity chain for any newly added codes ---
    # Precedence (updated):
    #   1) quantity_resolver evidence (measurements/intent)
    #   2) learnLayer qty advisor (if available) — rounded UP
    #   3) memory fallback qty
    # NOTE: For added codes we do NOT apply NE/OVER defaults ahead of memory; resolver already does postprocess.
    if added_codes:
        picked = list(added_codes.keys())

        qty_priors: Dict[str, float] = {}
        # 1) quantity_resolver (evidence-based)
        resolved_map: Dict[str, Dict[str, Any]] = {}
        try:
            from sorCodeModel.mainModelLayer.quantity_resolver import resolve_quantities
            resolved_map = resolve_quantities(
                instance=instance,
                picked_codes=picked,
                form_data=form_data,
                free_text_map=free_text_map,
                sor_csv_path=sor_csv_path,
                pack=pack,
            ) or {}
        except Exception:
            resolved_map = {}

        # 2) optional qty advisor
        advisor_fn = None
        try:
            from sorCodeModel.learnLayer.learn_core import get_qty_advisor
            advisor_fn = get_qty_advisor(feedback_csv_path=feedback_csv_path)
        except Exception:
            advisor_fn = None

        for code in picked:
            # Evidence qty
            res = resolved_map.get(code) or {}
            ev_qty = _extract_qty_like(res.get("qty"), fallback=1.0)
            qty = ev_qty
            src = str(res.get("source") or "")

            # If resolver couldn't find anything (or returned default 1), try advisor, else fallback to memory qty.
            if qty <= 1.0 and advisor_fn is not None:
                try:
                    adv = advisor_fn(code=code, instance=instance, pack=pack, text=text_blob)
                    adv_qty = _extract_qty_like(adv, fallback=1.0)
                    if adv_qty > 1.0:
                        qty = float(math.ceil(adv_qty))
                        qty_priors[code] = qty
                except Exception:
                    pass

            if qty <= 1.0:
                qty = float(added_codes.get(code, 1.0))

            predicted_map[code] = float(qty)
            if code not in codes:
                codes.append(code)
            if code in qty_priors:
                print(f"[MEMORY] Added code {code} with qty={qty} (advisor={qty_priors.get(code)}; mem_fallback={added_codes.get(code)}; resolver={src}).")
            else:
                print(f"[MEMORY] Added code {code} with qty={qty} (advisor=NONE; mem_fallback={added_codes.get(code)}; resolver={src or 'NONE'}).")

    # --- Apply strong qty bumps for >0.95 matches on existing codes ---
    for code, mem_qty in strong_qty_overrides.items():
        if code not in predicted_map:
            continue
        existing = _extract_qty_like(predicted_map.get(code), fallback=1.0)
        new_qty = max(existing, mem_qty)
        if new_qty != existing:
            predicted_map[code] = new_qty
            print(f"[MEMORY] Bumped qty for code {code} from {existing} -> {new_qty} based on strong memory.")
        else:
            print(f"[MEMORY] Qty for code {code} already >= memory qty ({existing} >= {mem_qty}), no change.")


# ---------------------------------------------------------------------------

    # --- Suppress unsupported ORIGINAL codes (score-based, safer than text-evidence pruning) ---
    # Keep inspection always. Keep if reranker top-5 or if present in top-5 similar memory jobs.
    mem_top5_codes: set = set()
    try:
        for it, _sim in hits[:5]:
            for c in (getattr(it, "codes_qty", None) or {}).keys():
                cc = _normalise_code_key(c)
                if cc:
                    mem_top5_codes.add(cc)
    except Exception:
        mem_top5_codes = set()

    def _supported(code: str) -> bool:
        if code == INSPECTION_CODE:
            return True
        if code in reranker_top5:
            return True
        if code in mem_top5_codes:
            return True
        # Material-aware protection: if current job is cast iron, keep cast-iron described codes.
        if prefer_cast_iron and _sor_desc_has(sor_desc_map, code, "cast iron"):
            return True
        return False

    removed = 0
    for oc in list(original_codes):
        ocn = _normalise_code_key(oc)
        if not ocn:
            continue
        if ocn not in predicted_map:
            continue
        if not _supported(ocn):
            predicted_map.pop(ocn, None)
            if ocn in codes:
                try:
                    codes.remove(ocn)
                except ValueError:
                    pass
            removed += 1
    if removed:
        print(f"[MEMORY] Suppressed {removed} original codes not supported by reranker/memory (kept inspection/material-protected).")

# Main run + log
# ---------------------------------------------------------------------------

def run_and_log_without_prompt(
    visit_id: int | str,
    sor_csv_path: str = SOR_CSV_PATH_DEFAULT,
    feedback_csv_path: str = FEEDBACK_CSV_PATH_DEFAULT
) -> dict:
    """
    Runs the model, logs outputs, and returns predictions.
    Ensures future quantities are returned as a dict {code: qty}.
    Ensures codes/future_codes are List[str] to match DB/service expectations.
    """
    visit_id_int = int(visit_id)

    # 1) Extract instance
    extra_details, instance, images = _get_instance(visit_id_int)
    print("EXTRA: ", extra_details)
    print("INSTANCE: ", instance)
    print("IMAGES: ", images)

    # 2) Run the model
    raw = get_sor_codes_from_instance(
        instance,
        sor_csv_path=sor_csv_path,
        images=images,
        extra_details=extra_details,
    )
    enriched = get_sor_codes_and_quantities_from_instance(
        instance,
        sor_csv_path=sor_csv_path,
        images=images,
        extra_details=extra_details,
    )

    # --- IMPORTANT: normalize codes to strings early ---
    codes_raw = enriched.get("codes", []) or raw.get("codes", []) or []
    future_codes_raw = enriched.get("future_codes", []) or []

    codes = [str(c) for c in codes_raw]
    future_codes = [str(c) for c in future_codes_raw]

    quantities = enriched.get("quantities", {}) or {}

    # Build evidence & caches
    try:
        form_data = flatten_survey_data(instance)
    except Exception:
        form_data = {}
    print("Form Process:  ", form_data)

    free_text_map = _collect_free_text(instance)
    text_blob = "\n".join(free_text_map.values()).strip()
    text_hash = _hash_free_text(instance)

    pack = build_evidence_pack(instance, form_data, free_text_map)
    pack_hash = hash_pack(pack)

    _update_json_cache(TEXT_CACHE_PATH_DEFAULT, text_hash, text_blob)
    _update_json_cache(PACK_CACHE_PATH_DEFAULT, pack_hash, pack)
    _update_json_cache(FORM_CACHE_PATH_DEFAULT, pack_hash, form_data)

    # --- Completed predictions ---
    predicted_map: Dict[str, float] = {}
    if isinstance(quantities, dict):
        for k, v in quantities.items():
            predicted_map[str(k)] = _extract_qty_like(v, fallback=1.0)

    print("PREDICTED (completed) BEFORE MEMORY:")
    for code, q in predicted_map.items():
        print(f"{code} x {q}")

    # Memory layer
    if os.environ.get("SOR_ENABLE_MEMORY", "0") == "1":
        print("[MEMORY] Applying memory layer.")
        _apply_memory_layer(
            codes=codes,  # now List[str]
            predicted_map=predicted_map,  # Dict[str, float]
            instance=instance,
            pack=pack,
            form_data=form_data,
            free_text_map=free_text_map,
            text_blob=text_blob,
            sor_csv_path=sor_csv_path,
            feedback_csv_path=feedback_csv_path,
        )
    else:
        print("[MEMORY] Skipping memory layer.")

    print("PREDICTED (completed) AFTER MEMORY:")
    for code, q in predicted_map.items():
        print(f"{code} x {q}")

    # Track quantities for any codes moved from completed -> future
    moved_to_future_qty: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Priority 1: Inspection-only visits
    # Even if memory adds codes, treat them as FUTURE codes for inspection-only,
    # except the inspection code itself.
    # ------------------------------------------------------------------
    if _is_inspection_only_visit(instance):
        moved = [c for c in codes if str(c) != INSPECTION_CODE]
        # Snapshot the quantities BEFORE we strip them out of completed
        moved_to_future_qty = {str(c): float(_extract_qty_like(predicted_map.get(str(c)), fallback=1.0)) for c in moved}
        if moved:
            print(f"[INSPECTION-ONLY] Moving {len(moved)} completed codes to future.")

        # completed should only contain inspection code (if present)
        codes = [c for c in codes if str(c) == INSPECTION_CODE]
        predicted_map = {INSPECTION_CODE: float(_extract_qty_like(predicted_map.get(INSPECTION_CODE), fallback=1.0))} if INSPECTION_CODE in predicted_map else {}

        # merge into future codes (dedupe while preserving order)
        seen = set()
        merged_future: List[str] = []
        for c in (future_codes + moved):
            cs = str(c)
            if cs == INSPECTION_CODE:
                continue
            if cs in seen:
                continue
            seen.add(cs)
            merged_future.append(cs)
        future_codes = merged_future

    # --- Future quantities ---
    future_free_text_map = {
        k: v for k, v in free_text_map.items()
        if k in (
            "11.2_Other_Works_Needed",
            "13.1_Issues_Present",
            "13.2_Issues_Comments",
        )
    }

    try:
        future_quantities_raw = resolve_quantities(
            instance=instance,
            picked_codes=future_codes,  # now List[str]
            form_data=form_data,
            free_text_map=future_free_text_map,
            sor_csv_path=sor_csv_path,
        )
    except Exception:
        future_quantities_raw = {str(c): {"qty": 1.0} for c in future_codes}

    # Normalize future quantities to {code: float}
    future_quantities: Dict[str, float] = {}

    if isinstance(future_quantities_raw, dict):
        for k, v in future_quantities_raw.items():
            future_quantities[str(k)] = _extract_qty_like(v, fallback=1.0)

    elif isinstance(future_quantities_raw, list):
        for item in future_quantities_raw:
            if isinstance(item, dict):
                code = str(item.get("code"))
                qty = _extract_qty_like(item, fallback=1.0)
            else:
                code = str(item)
                qty = 1.0
            future_quantities[code] = float(qty)

    # If we moved completed codes to future (inspection-only), ensure qtys go with them.
    if moved_to_future_qty:
        for c, q in moved_to_future_qty.items():
            future_quantities[str(c)] = float(q)
    print("PREDICTED (future):")
    for code, qty in future_quantities.items():
        print(f"{code} x {qty}")

    # Evidence snapshot
    evidence = build_evidence_snapshot(
        pack=pack,
        form=form_data,
        text_blob=text_blob,
        predicted_map=predicted_map,
        text_hash=text_hash,
        pack_hash=pack_hash,
    )
    evidence_json = json.dumps(evidence, ensure_ascii=False)

    # --- Combined predictions (completed + future) ---
    combined_predicted: Dict[str, float] = dict(predicted_map)
    for code, qty in future_quantities.items():
        if code not in combined_predicted:
            combined_predicted[code] = float(qty)

    job_id = (extra_details or {}).get("job_id") or (extra_details or {}).get("jobId") or ""

    _append_feedback_row_with_evidence(
        feedback_csv_path=feedback_csv_path,
        visit_id=str(visit_id_int),
        job_id=str(job_id) if job_id else "",
        text_hash=text_hash,
        pack_hash=pack_hash,
        predicted_dict=combined_predicted,
        corrected_dict={},
        evidence_json=evidence_json,
        notes=""
    )

    return {
        "visit_id": visit_id_int,
        # completed
        "codes": codes,  # List[str]
        "quantities": predicted_map,  # Dict[str, float]
        "completed_codes": codes,  # List[str]
        "completed_quantities": predicted_map,  # Dict[str, float]
        # future (DICT guaranteed)
        "future_codes": future_codes,  # List[str]
        "future_quantities": future_quantities,  # Dict[str, float]
        "text_hash": text_hash,
        "pack_hash": pack_hash,
    }




def _update_json_cache(path: str, key: str, value) -> None:
    """Tiny append/update cache for text/blob pack/form snapshots keyed by hash."""
    data: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[str(key)] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
#abcd

def run_with_memory_enabled_once(
    visit_id: int | str,
    sor_csv_path: str = SOR_CSV_PATH_DEFAULT,
    feedback_csv_path: str = FEEDBACK_CSV_PATH_DEFAULT,
    env_flag: str = "SOR_ENABLE_MEMORY",
) -> dict:
    """
    (3) Memory-on one-shot:
        Temporarily enables memory-assisted prediction by setting ENV flags for
        memory, re-ranker, and quantity hints, performs the same logging as (2),
        then restores the previous values.

        Returns the same dict summary as (2).
    """
    # We now toggle a small bundle of feature flags together for this call:
    #   - env_flag (default: SOR_ENABLE_MEMORY)
    #   - SOR_ENABLE_RERANKER
    #   - SOR_ENABLE_QTY_HINTS
    flags = {env_flag, "SOR_ENABLE_RERANKER", "SOR_ENABLE_QTY_HINTS"}

    prev_values = {name: os.environ.get(name, None) for name in flags}
    for name in flags:
        os.environ[name] = "1"

    try:
        result = run_and_log_without_prompt(
            visit_id=visit_id,
            sor_csv_path=sor_csv_path,
            feedback_csv_path=feedback_csv_path,
        )
    finally:
        # Restore previous values exactly (including absence)
        for name, old_value in prev_values.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value

    return result