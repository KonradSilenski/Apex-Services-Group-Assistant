import re
import math
from typing import List, Dict, Any, Optional, Tuple

# External types/utilities (do not modify / keep import stable)
from sorCodeModel.mainModelLayer.formProcessing import DetailsResponse
# quantity_schema_builder was removed; schema builder now lives in learn_core
from sorCodeModel.learnLayer.learn_core import build_quantity_schema_from_csv


# ---------------------------
# Basic helpers
# ---------------------------

def _parse_float(text: Any) -> Optional[float]:
    """Extract a float from a string like '6', '6.0', '6,5'. Returns None if no number."""
    if text is None:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(text))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except Exception:
        return None


def _coalesce(*vals: Any) -> Optional[str]:
    """Return the first non-empty string value, else None."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
        else:
            try:
                s = str(v).strip()
                if s:
                    return s
            except Exception:
                continue
    return None


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return None


# ---------------------------
# Configuration
# ---------------------------

# Free-text fields we scan for textual quantities / clues
FREE_TEXT_FIELDS_FOR_QTY: List[str] = [
    "1.2_Work_Description",
    "6.3_Leadwork_Comment",
    "7.3_Chimney_Comment",
    "7.5_Chimney_Flaunch_Comment",
    "11.1_Other_Works_Completed",
    "11.2_Other_Works_Needed",
    "13.1_Issues_Present",
    "13.2_Issues_Comments",
]

# Default UOM by family/material hint.
# (Used only when schema doesn't provide a clear UOM.)
DEFAULT_UOM_FOR_FAMILY: Dict[str, str] = {
    "tile": "EACH",
    "gutter": "EACH",      # default is per elevation unless LM found
    "downpipe": "EACH",
    "fascia": "M",
    "soffit": "M",
    "ridge": "M",
    "verge": "M",
    "valley": "M",
    "leadwork": "M",
    "chimney": "EACH",
    "scaffold": "EACH",
    "inspection": "EACH",
    "airbrick": "EACH",
    "special": "EACH",
    "flat_roof": "M2",
    "sheet_roof": "M2",
    "green_roof": "M2",
    "insulation": "M2",
    "structural_timber": "EACH",
    "cladding": "M2",
    "solar_panels": "EACH",
    "timber_treatment": "EACH",
    "asbestos": "EACH",
}


# ---------------------------
# Family inference (schema text → family)
# ---------------------------

# These are deliberately broad so we "account for everything possible".
# They are used only to decide WHICH extraction strategy to try.
FAMILY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    # Rainwater goods
    "gutter": ("gutter", "gutters", "guttering", "rainwater goods", "rain water goods", "rwg", "rainwater system"),
    "downpipe": ("downpipe", "downpipes", "down pipe", "down pipes", "rwp", "rainwater pipe", "rain water pipe"),

    # Roof edges
    "fascia": ("fascia", "fascias", "barge board", "bargeboard", "barge boards", "bargeboards"),
    "soffit": ("soffit", "soffits"),

    # Roof lines
    "ridge": ("ridge", "hip", "hips"),
    "verge": ("verge", "verges"),
    "valley": ("valley", "valleys"),

    # Chimneys / lead
    "chimney": ("chimney", "flaunch", "flaunching", "stack", "pots"),
    "leadwork": ("leadwork", "lead work", "lead", "flashing", "flashings", "apron", "soaker", "abutment", "step cover", "step flashing"),

    # Coverings
    "tile": ("tile", "tiles", "slate", "slates", "plain tile", "concrete tile", "clay tile"),
    "flat_roof": ("flat roof", "felt", "asphalt", "asphalt roofing", "flat coverings"),
    "sheet_roof": ("sheet roofing", "sheet roof", "sheeting", "corrugated", "profiled sheet"),
    "green_roof": ("green roof", "sedum"),

    # Access
    "scaffold": ("scaffold", "scaffolding"),

    # Other
    "inspection": ("inspect", "inspection", "attendance", "survey"),
    "airbrick": ("airbrick", "air brick", "air vent", "airvent", "air-vent", "vent"),
    "insulation": ("insulation", "insulate", "loft insulation"),
    "structural_timber": ("joist", "joists", "timber", "rafter", "rafters", "stud", "partition", "hanger", "plate", "collar", "strut"),
    "cladding": ("cladding", "external cladding", "shiplap", "weatherboard", "weatherboarding", "feather edge"),
    "solar_panels": ("solar", "solar panel", "solar panels", "pv", "photovoltaic"),
    "timber_treatment": ("timber treatment", "woodworm", "preservative"),
    "asbestos": ("asbestos", "asbestos removal"),
    "special": ("special", "call out", "call-out", "drone", "adjustment to invoice", "bird guard", "bird guards", "vcs"),
}


def _code_family_from_schema(code: str, schema: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Infer family from schema blob (description + facets)."""
    info = schema.get(code, {}) or {}
    blob = " ".join([
        info.get("description", "") or "",
        info.get("job_type", "") or "",
        info.get("element", "") or "",
        info.get("category", "") or "",
        info.get("sub_category", "") or "",
    ]).lower()

    # Priority ordering: keep gutter/downpipe split stable
    if any(w in blob for w in FAMILY_KEYWORDS["downpipe"]):
        return "downpipe"
    if any(w in blob for w in FAMILY_KEYWORDS["gutter"]):
        return "gutter"

    # then other families
    for fam, kws in FAMILY_KEYWORDS.items():
        if fam in ("gutter", "downpipe"):
            continue
        if any(w in blob for w in kws):
            return fam

    return None


# ---------------------------
# Text extraction
# ---------------------------

_UNIT_M_RE = r"(?:m|lm|metre|meter|metres|meters|lin\.?\s*m(?:etres?)?)"
_UNIT_M2_RE = r"(?:m2|m²|sqm|sq\.?\s*m|square\s*(?:metre|meter)s?)"


def _join_text(free_text_map: Dict[str, str]) -> str:
    return " . ".join([(free_text_map.get(f) or "") for f in FREE_TEXT_FIELDS_FOR_QTY]).strip()


def _extract_counts(text: str, words: Tuple[str, ...]) -> List[Tuple[float, str]]:
    """Extract counts like '2 downpipes' or 'downpipes 2'."""
    out: List[Tuple[float, str]] = []
    if not text:
        return out
    w_alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))

    # number before word
    for m in re.finditer(rf"\b(\d{{1,3}})\s*(?:x\s*)?(?:{w_alt})\b", text, flags=re.I):
        q = _parse_float(m.group(1))
        if q:
            out.append((q, m.group(0).strip()))

    # word before number
    for m in re.finditer(rf"\b(?:{w_alt})\b[^\d]{{0,10}}(\d{{1,3}})\b", text, flags=re.I):
        q = _parse_float(m.group(1))
        if q:
            out.append((q, m.group(0).strip()))

    return out


def _extract_lengths_m(text: str, words: Tuple[str, ...]) -> List[Tuple[float, str]]:
    """Extract lengths in metres tied to family words."""
    out: List[Tuple[float, str]] = []
    if not text:
        return out
    w_alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))

    # word then number + unit
    for m in re.finditer(
        rf"\b(?:{w_alt})\b[^\d]{{0,12}}(\d{{1,4}}(?:[.,]\d{{1,2}})?)\s*({_UNIT_M_RE})\b",
        text,
        flags=re.I,
    ):
        q = _parse_float(m.group(1))
        if q:
            out.append((q, m.group(0).strip()))

    # number + unit then word
    for m in re.finditer(
        rf"\b(\d{{1,4}}(?:[.,]\d{{1,2}})?)\s*({_UNIT_M_RE})\b[^a-zA-Z]{{0,12}}\b(?:{w_alt})\b",
        text,
        flags=re.I,
    ):
        q = _parse_float(m.group(1))
        if q:
            out.append((q, m.group(0).strip()))

    return out


def _extract_areas_m2(text: str, words: Tuple[str, ...]) -> List[Tuple[float, str]]:
    """Extract areas in m2 tied to family words."""
    out: List[Tuple[float, str]] = []
    if not text:
        return out
    w_alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))

    for m in re.finditer(
        rf"\b(?:{w_alt})\b[^\d]{{0,16}}(\d{{1,5}}(?:[.,]\d{{1,2}})?)\s*({_UNIT_M2_RE})\b",
        text,
        flags=re.I,
    ):
        q = _parse_float(m.group(1))
        if q:
            out.append((q, m.group(0).strip()))

    for m in re.finditer(
        rf"\b(\d{{1,5}}(?:[.,]\d{{1,2}})?)\s*({_UNIT_M2_RE})\b[^a-zA-Z]{{0,16}}\b(?:{w_alt})\b",
        text,
        flags=re.I,
    ):
        q = _parse_float(m.group(1))
        if q:
            out.append((q, m.group(0).strip()))

    return out


def _extract_text_quantities(free_text_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Pull quantities from narrative text; returns list of {mat, quantity, uom, source, evidence}."""
    out: List[Dict[str, Any]] = []
    text = _join_text(free_text_map)

    # Existing explicit tile patterns (kept because they work well)
    for m in re.finditer(
        r"(?:replace(?:d)?|renew|refix|repair|lift(?:ed)?|remove(?:d)?|took\s+up)[^\.]{0,40}?\b(\d{1,3})\s+(?:tile|tiles|slate|slates)\b",
        text,
        flags=re.I,
    ):
        q = _parse_float(m.group(1))
        if q:
            out.append({"mat": "tile", "quantity": q, "uom": "EACH", "source": "TEXT", "evidence": m.group(0).strip()})

    for m in re.finditer(r"\b(\d{1,3})\s+(?:tile|tiles|slate|slates)\b", text, flags=re.I):
        q = _parse_float(m.group(1))
        if q:
            out.append({"mat": "tile", "quantity": q, "uom": "EACH", "source": "TEXT", "evidence": m.group(0).strip()})

    # Generic family extraction
    for fam, kws in FAMILY_KEYWORDS.items():
        if fam in ("inspection", "special"):
            continue

        # Counts (EACH-like)
        if fam in ("downpipe", "chimney", "scaffold", "airbrick", "solar_panels", "timber_treatment", "asbestos", "structural_timber"):
            for q, ev in _extract_counts(text, kws):
                out.append({"mat": fam, "quantity": q, "uom": "EACH", "source": "TEXT", "evidence": ev})

        # Lengths (M-like)
        if fam in ("gutter", "fascia", "soffit", "ridge", "verge", "valley", "leadwork"):
            for q, ev in _extract_lengths_m(text, kws):
                out.append({"mat": fam, "quantity": q, "uom": "M", "source": "TEXT", "evidence": ev})

        # Areas (M2-like)
        if fam in ("flat_roof", "sheet_roof", "green_roof", "insulation", "cladding"):
            for q, ev in _extract_areas_m2(text, kws):
                out.append({"mat": fam, "quantity": q, "uom": "M2", "source": "TEXT", "evidence": ev})

            # Also allow LM if someone writes "10m of felt" etc.
            for q, ev in _extract_lengths_m(text, kws):
                out.append({"mat": fam, "quantity": q, "uom": "M", "source": "TEXT", "evidence": ev})

    return out


# ---------------------------
# Structured extraction (from the form)
# ---------------------------

def _get_first_present(form_data: Dict[str, Any], instance: DetailsResponse, keys: List[str]) -> Optional[str]:
    """Check form_data first, then instance fields for the first present value among keys."""
    for k in keys:
        v = _coalesce(form_data.get(k), getattr(instance, k, None))
        if v:
            return v
    return None


def _scan_form_by_key_tokens(
    form_data: Dict[str, Any],
    required_any: Tuple[str, ...],
    required_all: Tuple[str, ...] = (),
    preferred_any: Tuple[str, ...] = (),
    deny_any: Tuple[str, ...] = (),
) -> Optional[Tuple[float, str]]:
    """Scan form_data keys for numeric values safely.

    - required_any: at least one token must be present
    - required_all: all tokens must be present
    - preferred_any: used only to pick the best match among candidates
    - deny_any: if any token present, skip the key

    Returns (qty, evidence_key) or None.
    """
    req_any = tuple(_norm_key(t) for t in required_any)
    req_all = tuple(_norm_key(t) for t in required_all)
    pref_any = tuple(_norm_key(t) for t in preferred_any)
    deny = tuple(_norm_key(t) for t in deny_any)

    cands: List[Tuple[int, float, str]] = []  # (score, qty, key)
    for k, v in (form_data or {}).items():
        k_norm = _norm_key(str(k))

        if deny and any(d in k_norm for d in deny):
            continue

        if req_any and not any(t in k_norm for t in req_any):
            continue

        if req_all and not all(t in k_norm for t in req_all):
            continue

        q = _parse_float(v)
        if q is None:
            continue

        score = 0
        if pref_any and any(t in k_norm for t in pref_any):
            score += 2
        if "measurement" in k_norm or "measure" in k_norm:
            score += 2
        if "num" in k_norm or "count" in k_norm or "quantity" in k_norm or "no" in k_norm or "number" in k_norm:
            score += 1

        # Prefer larger qty slightly only if score tie
        cands.append((score, float(q), str(k)))

    if not cands:
        return None

    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = cands[0]
    return best[1], best[2]


def _structured_quantities(instance: DetailsResponse, form_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract quantities directly from structured fields (measurements/counts)."""
    out: List[Dict[str, Any]] = []

    # Existing explicit extraction (kept)
    tile_meas = _get_first_present(form_data, instance, [
        "4.7_Flat_Measurement",
        "4.4_Roof_Measurement",
        "4.10_Other_Measurement",
        "roof_measurement",
    ])
    q_tile = _parse_float(tile_meas)
    if q_tile:
        out.append({"mat": "tile", "quantity": q_tile, "uom": "EACH", "source": "MEASUREMENT", "evidence": "roof_measurement"})

    lead_meas = _get_first_present(form_data, instance, ["6.2_Leadwork_Measurement", "leadwork_measurement"])
    q_lead = _parse_float(lead_meas)
    if q_lead:
        out.append({"mat": "leadwork", "quantity": q_lead, "uom": "M", "source": "MEASUREMENT", "evidence": "lead_measurement"})

    ridge_meas = _get_first_present(form_data, instance, ["5.4_Ridge_Measurement", "ridge_measurement"])
    q_ridge = _parse_float(ridge_meas)
    if q_ridge:
        out.append({"mat": "ridge", "quantity": q_ridge, "uom": "M", "source": "MEASUREMENT", "evidence": "ridge_measurement"})

    gut_meas = _get_first_present(form_data, instance, [
        "9.3_Guttering_Replace_Measurement",
        "9.5_Guttering_Refix_Measurement",
        "9.6_Guttering_Clean",
        "guttering_clean",
    ])
    q_gut_lm = _parse_float(gut_meas)
    if q_gut_lm:
        out.append({"mat": "gutter", "quantity": q_gut_lm, "uom": "M", "source": "MEASUREMENT", "evidence": "gutter_measurement"})

    gut_elevs = _get_first_present(form_data, instance, ["9.7_Guttering_Num_Elevations"])
    q_gut_elevs = _parse_float(gut_elevs)
    if q_gut_elevs:
        out.append({"mat": "gutter_elevs", "quantity": q_gut_elevs, "uom": "EACH", "source": "MEASUREMENT", "evidence": "gutter_elevations"})

    # Generic structured scans (safe / additive)
    generic_rules = [
        ("fascia", ("fascia",), (), ("measurement", "measure", "length"), ("comment", "notes"), "M"),
        ("soffit", ("soffit",), (), ("measurement", "measure", "length"), ("comment", "notes"), "M"),
        ("verge", ("verge",), (), ("measurement", "measure", "length"), ("comment", "notes"), "M"),
        ("valley", ("valley",), (), ("measurement", "measure", "length"), ("comment", "notes"), "M"),
        ("downpipe", ("downpipe", "rwp", "rainwater"), (), ("num", "count", "quantity", "no", "number"), ("comment", "notes"), "EACH"),
        ("chimney", ("chimney", "stack"), (), ("num", "count", "quantity", "no", "number"), ("comment", "notes"), "EACH"),
        ("scaffold", ("scaffold", "scaffolding"), (), ("num", "count", "quantity", "no", "number", "elevation"), ("comment", "notes"), "EACH"),
        ("airbrick", ("airbrick", "vent"), (), ("num", "count", "quantity", "no", "number"), ("comment", "notes"), "EACH"),
        ("insulation", ("insulation",), (), ("measurement", "measure", "area", "m2", "sqm"), ("comment", "notes"), "M2"),
        ("flat_roof", ("flat", "felt", "asphalt"), ("roof",), ("measurement", "measure", "area", "m2", "sqm"), ("comment", "notes"), "M2"),
        ("sheet_roof", ("sheet", "corrugated"), ("roof",), ("measurement", "measure", "area", "m2", "sqm"), ("comment", "notes"), "M2"),
        ("green_roof", ("green", "sedum"), ("roof",), ("measurement", "measure", "area", "m2", "sqm"), ("comment", "notes"), "M2"),
        ("cladding", ("cladding", "weatherboard", "shiplap"), (), ("measurement", "measure", "area", "m2", "sqm"), ("comment", "notes"), "M2"),
        ("solar_panels", ("solar", "pv"), (), ("num", "count", "quantity", "no", "number"), ("comment", "notes"), "EACH"),
        ("asbestos", ("asbestos",), (), ("num", "count", "quantity", "no", "number"), ("comment", "notes"), "EACH"),
    ]

    for fam, req_any, req_all, pref_any, deny_any, uom in generic_rules:
        found = _scan_form_by_key_tokens(
            form_data=form_data,
            required_any=req_any,
            required_all=req_all,
            preferred_any=pref_any,
            deny_any=deny_any,
        )
        if not found:
            continue
        q, ev_key = found
        if q and q > 0:
            out.append({
                "mat": fam,
                "quantity": float(q),
                "uom": uom,
                "source": "MEASUREMENT",
                "evidence": f"form:{ev_key}",
            })

    return out


# ---------------------------
# Inference helpers
# ---------------------------

def _infer_elevations_from_text(text: str) -> int:
    """Crude inference of elevations from text via presence of front/rear/gable."""
    t = (text or "").lower()
    flags = 0
    if "front" in t:
        flags += 1
    if ("rear" in t) or ("back" in t):
        flags += 1
    if "gable" in t:
        flags += 1  # 'both gables' still counts as 1
    return max(flags, 0)


def _infer_scaffold_from_text(text: str) -> Optional[int]:
    """Infer scaffold count (EACH) from phrases when no explicit number is present."""
    t = (text or "").lower()
    if "scaffold" not in t and "scaffolding" not in t:
        return None

    # Explicit 'front & rear' etc.
    sides = 0
    if "front" in t:
        sides += 1
    if "rear" in t or "back" in t:
        sides += 1
    if "gable" in t:
        sides += 1

    if "full" in t or "all elevations" in t or "all sides" in t or "around" in t:
        # typical max is 3 (front, rear, gable)
        return 3

    if sides >= 1:
        return min(max(sides, 1), 3)

    # Mentions scaffold but no details
    return 1


def _infer_each_from_sides(
    text: str,
    family_keywords: Tuple[str, ...],
    *,
    max_sides: int = 3,
) -> Optional[int]:
    """Infer a conservative EACH quantity from "sides" language when no explicit number is present.

    This is intentionally cautious: it only triggers when the family is explicitly mentioned
    (via keywords), but the text does *not* contain a clear numeric count for that family.

    Examples it can capture:
      - "replace downpipes to front and rear" -> 2
      - "rwp to all elevations" -> 3

    Notes:
      - We cap at max_sides (default 3: front, rear, gable).
      - If the family is mentioned but no side cues exist, we default to 1.
    """
    t = (text or "").lower()
    if not t:
        return None

    # Must mention the family explicitly
    if not any(k.lower() in t for k in family_keywords):
        return None

    # If a clear number already exists nearby the family keywords, do nothing.
    # (We don't want to override the count extractor.)
    kw_alt = "|".join(re.escape(k) for k in sorted(family_keywords, key=len, reverse=True))
    if re.search(rf"\b(\d{{1,3}})\s*(?:x\s*)?(?:{kw_alt})\b", t, flags=re.I):
        return None
    if re.search(rf"\b(?:{kw_alt})\b[^\d]{{0,10}}(\d{{1,3}})\b", t, flags=re.I):
        return None

    # Side cues
    sides = 0
    if "front" in t:
        sides += 1
    if "rear" in t or "back" in t:
        sides += 1
    if "gable" in t:
        sides += 1

    if "full" in t or "all elevations" in t or "all sides" in t or "around" in t:
        return max_sides

    if sides >= 1:
        return min(max(sides, 1), max_sides)

    return 1


# ---------------------------
# Special rules (code-specific)
# ---------------------------

# For these codes, if a tile EACH quantity is present in text, use it as the code's quantity.
SPECIAL_QTY_TIED_TO_TILE_COUNT: Dict[str, str] = {
    "201150": "EACH",  # Felt/underlay patch around tiles; qty follows tile count from text
}

# detect "NE X NO" and "OVER X NO" in code description
_NE_NO_RE = re.compile(r"\bNE\s*(\d+)\s*NO\b", re.IGNORECASE)
_OVER_NO_RE = re.compile(r"\bOVER\s*(\d+)\s*NO\b", re.IGNORECASE)


def _ne_default_postprocess(code_str: str, res: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """If description contains 'NE X NO' and qty is missing/1, set qty = max(1, X-1)."""
    info = schema.get(code_str, {}) or {}
    desc = info.get("description", "") or ""
    m = _NE_NO_RE.search(desc)
    if not m:
        return res

    qty = res.get("qty", None)
    try:
        qty_val = float(qty) if qty is not None else None
    except Exception:
        qty_val = None

    if qty_val is not None and qty_val != 1.0:
        return res

    try:
        x = int(m.group(1))
    except Exception:
        return res

    new_qty = float(max(1, x - 1))
    new_ev = (res.get("evidence", "") or "").strip()
    if new_ev:
        new_ev += " ; "
    new_ev += f"NE {x} NO in description ⇒ default X-1 applied → qty={int(new_qty)}"

    res = dict(res)
    res["qty"] = new_qty
    res["uom"] = res.get("uom") or "EACH"
    res["source"] = "DEFAULT_NE_X_MINUS_1"
    res["confidence"] = min(float(res.get("confidence", 0.60)), 0.60)
    res["evidence"] = new_ev
    return res


def _over_default_postprocess(code_str: str, res: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """If description contains 'OVER X NO' and qty is missing/1, set qty = X+1."""
    info = schema.get(code_str, {}) or {}
    desc = info.get("description", "") or ""
    m = _OVER_NO_RE.search(desc)
    if not m:
        return res

    qty = res.get("qty", None)
    try:
        qty_val = float(qty) if qty is not None else None
    except Exception:
        qty_val = None

    if qty_val is not None and qty_val != 1.0:
        return res

    try:
        x = int(m.group(1))
    except Exception:
        return res

    new_qty = float(max(1, x + 1))
    new_ev = (res.get("evidence", "") or "").strip()
    if new_ev:
        new_ev += " ; "
    new_ev += f"OVER {x} NO in description ⇒ default X+1 applied → qty={int(new_qty)}"

    res = dict(res)
    res["qty"] = new_qty
    res["uom"] = res.get("uom") or "EACH"
    res["source"] = "DEFAULT_OVER_X_PLUS_1"
    res["confidence"] = min(float(res.get("confidence", 0.60)), 0.60)
    res["evidence"] = new_ev
    return res


def _qty_for_603903(
    free_text_map: Dict[str, str],
    mat_best: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Special rule for 603903 (Gutter clean): max(ceil(LM/8), side-count, 1)."""
    lm_q = None
    gut = mat_best.get("gutter") or mat_best.get("gutter_elevs")
    if gut and isinstance(gut.get("quantity"), (int, float)):
        lm_q = float(gut["quantity"])
    A = math.ceil(lm_q / 8.0) if (lm_q is not None and lm_q > 0) else 0

    work_text = (free_text_map.get("1.2_Work_Description") or "").lower()
    has_rear = ("rear" in work_text) or ("back" in work_text)
    has_gable = ("gable" in work_text)
    has_front = ("front" in work_text)

    B = (1 if has_rear else 0) + (1 if has_gable else 0) + (1 if has_front else 0)
    if B > 3:
        B = 3

    final_qty = max(A, B, 1)

    parts = []
    if lm_q is not None:
        parts.append(f"LM={lm_q:g} ⇒ ceil(LM/8)={A}")
    else:
        parts.append("No LM found ⇒ A=0")

    side_bits = []
    if has_front:
        side_bits.append("front")
    if has_rear:
        side_bits.append("rear/back")
    if has_gable:
        side_bits.append("gable")
    parts.append(f"sides={{ {', '.join(side_bits) if side_bits else 'none'} }} ⇒ B={B}")

    picked = "A" if A >= max(B, 1) else "B"
    evidence = f"{' ; '.join(parts)} ⇒ picked {picked} → qty={final_qty}"
    conf = 0.80 if lm_q is not None else 0.65

    return {
        "qty": float(final_qty),
        "uom": "EACH",
        "source": "SPECIAL_603903_MAX(LM/8, SIDES)",
        "confidence": conf,
        "evidence": evidence,
    }


# ---------------------------
# Main entrypoint
# ---------------------------

def resolve_quantities(
    instance: DetailsResponse,
    picked_codes: List[str],
    form_data: Dict[str, Any],
    free_text_map: Dict[str, str],
    sor_csv_path: str,
) -> Dict[str, Dict[str, Any]]:
    """Resolve quantities for the picked SOR codes.

    Strategy:
      1) Structured measurements/counts from the form (best confidence)
      2) Text-derived hints tied to family keywords
      3) Family defaults and safe heuristics (elevations, scaffold sides)
      4) Code-specific special cases (e.g. 603903)

    Returns:
      { code: {qty,uom,source,confidence,evidence} }
    """
    schema = build_quantity_schema_from_csv(sor_csv_path)

    # Build candidate pools
    struct = _structured_quantities(instance, form_data)
    textqs = _extract_text_quantities(free_text_map)

    # Choose best per "mat" (family/material) - prefer MEASUREMENT over TEXT; if tie, prefer larger qty
    mat_best: Dict[str, Dict[str, Any]] = {}
    priority = {"MEASUREMENT": 2, "TEXT": 1}
    for q in struct + textqs:
        mat = q.get("mat")
        if not mat:
            continue
        prev = mat_best.get(mat)
        if prev is None:
            mat_best[mat] = q
            continue
        if priority.get(q.get("source", ""), 0) > priority.get(prev.get("source", ""), 0):
            mat_best[mat] = q
        elif priority.get(q.get("source", ""), 0) == priority.get(prev.get("source", ""), 0):
            if float(q.get("quantity") or 0) > float(prev.get("quantity") or 0):
                mat_best[mat] = q

    joined_text = _join_text(free_text_map)
    elevations_inferred = _infer_elevations_from_text(joined_text)
    if elevations_inferred <= 0:
        elevations_inferred = 1

    scaffold_inferred = _infer_scaffold_from_text(joined_text)

    # Conservative inference for SOME EACH-type families when they are mentioned
    # but no explicit number is present. This is additive and will never override
    # explicit structured/text counts.
    downpipe_inferred = _infer_each_from_sides(joined_text, FAMILY_KEYWORDS["downpipe"], max_sides=3)
    chimney_inferred = _infer_each_from_sides(joined_text, FAMILY_KEYWORDS["chimney"], max_sides=3)

    results: Dict[str, Dict[str, Any]] = {}

    for code in picked_codes:
        code_str = str(code)

        # Special case: 603903 (Gutter clean)
        if code_str == "603903":
            res = _qty_for_603903(free_text_map, mat_best)
            res = _ne_default_postprocess(code_str, res, schema)
            res = _over_default_postprocess(code_str, res, schema)
            results[code_str] = res
            continue

        # Tile-tied special cases
        if code_str in SPECIAL_QTY_TIED_TO_TILE_COUNT:
            tile_best = mat_best.get("tile")
            if tile_best and isinstance(tile_best.get("quantity"), (int, float)) and float(tile_best["quantity"]) > 0:
                res = {
                    "qty": float(tile_best["quantity"]),
                    "uom": SPECIAL_QTY_TIED_TO_TILE_COUNT[code_str],
                    "source": "TEXT",
                    "confidence": 0.65,
                    "evidence": tile_best.get("evidence", "tile count from narrative"),
                }
                res = _ne_default_postprocess(code_str, res, schema)
                res = _over_default_postprocess(code_str, res, schema)
                results[code_str] = res
                continue

        family = _code_family_from_schema(code_str, schema)

        # Determine best UOM: prefer schema expected_uom, else defaults
        expected_uom = (schema.get(code_str, {}) or {}).get("expected_uom")
        default_uom = DEFAULT_UOM_FOR_FAMILY.get(family or "", "EACH")
        uom = expected_uom or default_uom

        # Pick best evidence for this family
        best = None
        if family:
            if family == "gutter":
                best = mat_best.get("gutter") or mat_best.get("gutter_elevs")
            else:
                best = mat_best.get(family)

            # If schema thinks leadwork but structured stored under leadwork (OK)
            if best is None and family == "leadwork":
                best = mat_best.get("leadwork")

        if best:
            # Normal case: direct evidence found
            res = {
                "qty": float(best["quantity"]),
                "uom": best.get("uom") or uom,
                "source": best.get("source", "TEXT"),
                "confidence": 0.80 if best.get("source") == "MEASUREMENT" else 0.60,
                "evidence": best.get("evidence", ""),
            }
            res = _ne_default_postprocess(code_str, res, schema)
            res = _over_default_postprocess(code_str, res, schema)
            results[code_str] = res
            continue

        # Safe family-specific fallbacks
        if family == "gutter":
            res = {
                "qty": float(elevations_inferred),
                "uom": "EACH",
                "source": "ELEVATIONS_DEFAULT",
                "confidence": 0.55,
                "evidence": f"{elevations_inferred} elevations inferred from text presence (front/rear/gable).",
            }
            res = _ne_default_postprocess(code_str, res, schema)
            res = _over_default_postprocess(code_str, res, schema)
            results[code_str] = res
            continue

        if family == "scaffold" and scaffold_inferred is not None:
            res = {
                "qty": float(scaffold_inferred),
                "uom": "EACH",
                "source": "SCAFFOLD_SIDES_INFERRED",
                "confidence": 0.55,
                "evidence": f"Scaffold mentioned; inferred {scaffold_inferred} from text cues (front/rear/gable/full).",
            }
            res = _ne_default_postprocess(code_str, res, schema)
            res = _over_default_postprocess(code_str, res, schema)
            results[code_str] = res
            continue

        if family == "downpipe" and downpipe_inferred is not None:
            res = {
                "qty": float(downpipe_inferred),
                "uom": "EACH",
                "source": "SIDES_INFERRED",
                "confidence": 0.50,
                "evidence": f"Downpipe/RWP mentioned; inferred {downpipe_inferred} from text cues (front/rear/gable/full).",
            }
            res = _ne_default_postprocess(code_str, res, schema)
            res = _over_default_postprocess(code_str, res, schema)
            results[code_str] = res
            continue

        if family == "chimney" and chimney_inferred is not None:
            res = {
                "qty": float(chimney_inferred),
                "uom": "EACH",
                "source": "SIDES_INFERRED",
                "confidence": 0.50,
                "evidence": f"Chimney/stack mentioned; inferred {chimney_inferred} from text cues (front/rear/gable/full).",
            }
            res = _ne_default_postprocess(code_str, res, schema)
            res = _over_default_postprocess(code_str, res, schema)
            results[code_str] = res
            continue

        # Final conservative default
        res = {
            "qty": 1.0,
            "uom": uom,
            "source": "FIXED_DEFAULT",
            "confidence": 0.45,
            "evidence": "No direct evidence found; applied conservative default.",
        }
        res = _ne_default_postprocess(code_str, res, schema)
        res = _over_default_postprocess(code_str, res, schema)
        results[code_str] = res

    return results