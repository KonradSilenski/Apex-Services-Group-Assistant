# sorCodeModel/mainModelLayer/jobmatcher_model_v2.py
"""
Jobmatcher v2 — Steps 1–8 with structured-family inference and fixed facet lookup.

Key points:
- Facet reader now pulls from JobCodeMatcher.job_code_data (your actual loader).
- Family chosen by STRUCTURED COLUMNS in PRIORITY:
    1) Job Type
    2) Work Category
    3) Element
    4) Work Sub-Category
  (Stops at the first match.)
- Tile caps are 0 by default; only open when tiles are actually referenced or roof section has structured data.
- Sentence-aware recall, lightweight + stronger scoring, minimal inline rules, guarded recruitment.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import re
import difflib
from sorCodeModel.learnLayer.learn_core import get_reranker
from sorCodeModel.learnLayer.evidence_core import build_evidence_pack

# External matcher for SOR codes
from sorCodeModel.mainModelLayer.jobmatcher import JobCodeMatcher

# =============================
# Tunables / Public constants
# =============================

# Default field weights for the "completed" (main) pass. (user-edited earlier)
DEFAULT_FIELD_WEIGHTS_COMPLETED: Dict[str, float] = {
    "1.2_Work_Description": 1.60,
    "6.3_Leadwork_Comment": 1.20,
    "7.3_Chimney_Comment": 1.20,
    "7.5_Chimney_Flaunch_Comment": 1.20,
    "11.1_Other_Works_Completed": 1.60,
}

# Default field weights for the "future" pass. (user-edited earlier)
DEFAULT_FIELD_WEIGHTS_FUTURE: Dict[str, float] = {
    "11.2_Other_Works_Needed": 1.40,
    "13.1_Issues_Present":    1.20,
    "13.2_Issues_Comments":   1.20,
}

NOISE_TOKENS = {"", "n/a", "na", "none", "no", "-"}

# Sentence-aware recall controls
MAX_CHUNKS_PER_FIELD = 12
TOPK_PER_CHUNK = 25
CAP_PER_FIELD_AFTER_RECALL = 40

# Chunk weighting
CHUNK_LEN_SOFT_CAP = 0.5
CHUNK_LEN_PER_20TOK = 1/20
WD_POS_BONUS_FIRST = 0.10

# Fuzzy cutoff
FUZZ_RATIO = 0.84

# Step 4 scoring weights
W_INTENT   = 0.12
W_MAT      = 0.10
W_LOCAL    = 0.05
W_SMALL    = 0.08
W_CONFLICT = 0.12
W_OVERHAUL = 0.10
BOOST_MIN  = 0.75
BOOST_MAX  = 1.35

# Step 6 stronger blending
MULT_MIN = 0.70
MULT_MAX = 1.40
A_MULTI_FIELD   = 0.03
A_CHUNK_DENS    = 0.02
A_FAMILY_CROWD  = 0.02
A_LONG_CHUNK    = 0.02
A_JT_NUDGE      = 0.03

OVERHAUL_RX = re.compile(r"\b(overhaul|entire elevation|full elevation|complete roof|strip and re-?cover)\b", re.I)

# Roof covering material vocabulary and normalisation
ROOF_MATERIAL_TOKENS = {
    "concrete": {"concrete", "marley modern", "marley mendip", "double roman", "bold roll",
                 "pantile", "pantiles", "renown", "redland 49", "wessex", "fenland"},
    "slate": {"slate", "slates"},
    "clay": {"clay"},
    "felt": {"felt", "bitumen", "torch on", "torch-on"},
    "grp": {"grp", "fibreglass"},
    "single-ply": {"single ply", "single-ply", "singleply", "epdm", "tpo", "pvc"},
    "both": {"concrete and clay", "clay and concrete"},
    "other": {"other"},
}

SUBAREA_TOKENS = {
    "valley": {"valley", "valleys"},
    "verge": {"verge", "verges"},
    "ridge": {"ridge", "hip", "hips"},
    "chimney": {"chimney", "flaunch", "flaunching"},
    "eaves": {"eaves"},
    "abutment": {"abutment", "abutments"},
}

# =============================
# Simple NLP helpers
# =============================

_WORD_RE = re.compile(r"[a-z0-9]+")

def _tok(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())

def _canon(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split()).strip()

def _is_noise(s: str) -> bool:
    return _canon(s).lower() in NOISE_TOKENS

def _sim_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

def _fuzzy_hit(text: str, terms: Set[str], threshold: float = FUZZ_RATIO) -> bool:
    s = (text or "").lower()
    words = _tok(s)
    for t in terms:
        t_l = t.lower().strip()
        if " " in t_l and t_l in s:
            return True
    for w in words:
        w_singular = w[:-1] if w.endswith("s") else w
        for t in terms:
            t_l = t.lower().strip()
            if " " in t_l:
                continue
            if w == t_l or w_singular == t_l:
                return True
            if _sim_ratio(w, t_l) >= threshold:
                return True
    return False

_SENT_SPLIT_RE = re.compile(r"[.;:\n•]+")

def _sentence_chunks(text: str) -> List[str]:
    return [x.strip() for x in _SENT_SPLIT_RE.split(text or "") if x.strip()]

def _clause_split(s: str) -> List[str]:
    parts = re.split(r",|\band\b|\bwith\b", s, flags=re.I)
    out = []
    for p in parts:
        p = " ".join((p or "").split()).strip(" ,;-")
        if p:
            out.append(p)
    return out

def _chunk_text(field: str, text: str, max_chunks: int = MAX_CHUNKS_PER_FIELD) -> List[str]:
    chunks: List[str] = []
    for sent in _sentence_chunks(text):
        if len(_tok(sent)) < 3:
            continue
        if len(_tok(sent)) > 24:
            parts = _clause_split(sent)
            for c in parts:
                if len(_tok(c)) >= 3:
                    chunks.append(c)
        else:
            chunks.append(sent)
        if len(chunks) >= max_chunks:
            break
    return chunks[:max_chunks]

def _chunk_weight(field: str, chunk_index: int, chunk_text: str) -> float:
    tokens = _tok(chunk_text)
    length_bonus = min(CHUNK_LEN_SOFT_CAP, len(tokens) * CHUNK_LEN_PER_20TOK)
    pos_bonus = WD_POS_BONUS_FIRST if (field == "1.2_Work_Description" and chunk_index in (0, 1)) else 0.0
    return 1.0 + length_bonus + pos_bonus

# =============================
# Intents / materials (SOFT)
# =============================

PRIORITY_VERBS: Set[str] = {
    "renew", "replace", "refix", "repair", "repoint", "rebed", "rebuild", "remove",
    "clean", "clear", "unblock", "jet", "wash", "adjust", "dress", "redress", "readjust"
}

VERB_ALIASES = {
    "clean": {"clean","cleaned","cleaning","clear","cleared","clearing","unblock",
              "unblocked","unblocking","jet","jetted","jetting","wash","washed","washing"},
    "renew": {"renew","renewed","renewing","replace","replaced","replacing"},
    "refix": {"refix","refixed","refixing","repair","repaired","repairing","dress",
              "dressed","redress","readjust","adjust","adjusted"},
    "repoint": {"repoint","repointed","repointing"},
    "rebed": {"rebed","rebedded","rebedding"},
    "rebuild": {"rebuild","rebuilt","rebuilding"},
}

MATERIAL_ALIASES = {
    "tile": {"tile","tiles","plain","concrete","clay","slate","slates"},
    "slate": {"slate","slates"},
    "plain": {"plain"},
    "concrete": {"concrete"},
    "clay": {"clay"},
    "ridge": {"ridge","ridges","hip","hips"},
    "verge": {"verge","verges"},
    "valley": {"valley","valleys"},
    "flashing": {"flashing","flashings","abutment","abutments","step","cover","apron","soaker","soakers"},
    "apron": {"apron","aprons"},
    "soaker": {"soaker","soakers"},
    "gutter": {"gutter","gutters","guttering"},
    "downpipe": {"downpipe","downpipes","rwp","rainwater","rain"},
    "fascia": {"fascia","fascias"},
    "soffit": {"soffit","soffits"},
    "chimney": {"chimney","chimneys"},
    "airbrick": {"airbrick","air brick","air-brick","airvent","air vent","air-vent","vent"},

}

TILE_SUBTYPE_TOKENS = {
    "plain": {"plain"},
    "concrete": {
        "concrete","double roman","marley double roman","bold roll","pantile","pantiles",
        "marley mendip","marley modern","redland renown","redland 49","wessex","fenland"
    },
    "clay": {"clay"},
    "slate": {"slate","slates"},
}

def _normalize_verb(w: str) -> Optional[str]:
    wl = (w or "").lower()
    for base, forms in VERB_ALIASES.items():
        if wl in forms:
            return base
    return wl if wl in PRIORITY_VERBS else None

def _nearest_material(words: List[str], idx: int) -> Optional[str]:
    best = None; best_d = 9999
    flat = [(key, t) for key, toks in MATERIAL_ALIASES.items() for t in toks]
    for j, w in enumerate(words):
        for key, t in flat:
            if w == t:
                d = abs(j - idx)
                if d < best_d:
                    best_d = d; best = key
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
    out, seen = [], set()
    for p in pairs:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _infer_tile_subtype(text_blob: str) -> Optional[str]:
    s = (text_blob or "").lower()
    for sub, toks in TILE_SUBTYPE_TOKENS.items():
        if any(t in s for t in toks):
            return sub
    return None

def _extract_tile_count_from_text(text_blob: str) -> Optional[int]:
    m = re.search(r"(replace|renew|refix|repair)\s+(?:approximately\s+)?(\d{1,3})\s+(tile|tiles|slate|slates)\b", (text_blob or "").lower())
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    return None

def _extract_roof_materials_from_text(text: str) -> Set[str]:
    s = (text or "").lower()
    found = set()
    for mat, toks in ROOF_MATERIAL_TOKENS.items():
        for t in toks:
            if t in s:
                found.add(mat)
                break
    return found

# =============================
# Field weights (completed vs future)
# =============================

def _select_field_weights(fields: List[str]) -> Dict[str, float]:
    has_future = any(f in DEFAULT_FIELD_WEIGHTS_FUTURE for f in fields)
    weights: Dict[str, float] = {}
    for f in fields:
        if has_future and f in DEFAULT_FIELD_WEIGHTS_FUTURE:
            weights[f] = DEFAULT_FIELD_WEIGHTS_FUTURE[f]
        elif f in DEFAULT_FIELD_WEIGHTS_COMPLETED:
            weights[f] = DEFAULT_FIELD_WEIGHTS_COMPLETED[f]
        else:
            weights[f] = 1.0
    return weights

# =============================
# Facets (pull from job_code_data)
# =============================

def _code_facets(code: str, matcher: JobCodeMatcher) -> Dict[str, str]:
    """
    Pull facets from JobCodeMatcher.job_code_data.
    Expected keys: description, job_type, element, category[, sub_category]
    """
    rec = {}
    try:
        rec = getattr(matcher, "job_code_data", {}).get(code, {}) or {}
    except Exception:
        rec = {}

    short_desc = str(rec.get("description", "") or "")
    job_type   = str(rec.get("job_type", "") or "")
    element    = str(rec.get("element", "") or "")
    work_cat   = str(rec.get("category", "") or "")
    work_sub   = str(rec.get("sub_category", "") or "")  # may not exist

    facet_text = " | ".join(p for p in (short_desc, job_type, work_cat, work_sub, element) if p)

    return {
        "short_desc": short_desc,
        "job_type": job_type,
        "work_category": work_cat,
        "work_subcategory": work_sub,
        "element": element,
        "facet_text": facet_text or short_desc,
    }

# =============================
# Family inference (structured priority + fallback regex)
# =============================

FAMILY_KEYWORDS = {
    "guttering": {"gutter","guttering","rainwater goods","rain water goods","rwg","rwp system","rainwater system"},
    "downpipe":  {"downpipe","down pipe","rwp","rainwater pipe"},
    "fascia":    {"fascia","fascias","barge board","barge boards","bargeboard","bargeboards"},
    "soffit":    {"soffit","soffits"},
    "ridge":     {"ridge","hip","hips"},
    "verge":     {"verge","verges"},
    "valley":    {"valley","valleys"},
    "chimney":   {"chimney","flaunch","flaunching"},
    "leadwork":  {"leadwork","lead work","lead roof coverings","flashings","flashing","apron","soaker","abutment","step cover"},
    "tile":      {"tile","tiles","plain","slate","concrete","clay"},
    "scaffold":  {"scaffold","scaffolding"},
    "inspection":{"inspect","attendance","survey"},
    "airbrick":  {"airbrick","air brick","air bricks","air-vent","air vent","airvent","airbricks and vents"},
    # NEW FAMILIES (used by structured facet inference)
    "special":   {"special","call out","call-out","drone","vcs","adjustment to invoice","bird guard","bird guards"},
    "flat_roof": {"flat roof","felt roofing","felt roof","asphalt","asphalt roofing","flat coverings"},
    "sheet_roof": {"sheet roofing","sheet roof","sheeting","corrugated","profiled sheet"},
    "green_roof": {"green roof","green rooing","sedum","sedums"},
    "insulation": {"loft insulation","insulation","insulate"},
    "structural_timber": {
        "partitions","joist","joists","timber floor","partition","stud wall",
        "plate","rafter","rafters","collar","strut","hanger"
    },
    "cladding":  {"external cladding","cladding","shiplap","weatherboarding","weatherboard","feather edge"},
    "solar_panels": {"solar panel","solar panels","solar pv","pv","photovoltaic"},
    "timber_treatment": {"timber treatment","woodworm","preservative"},
    "asbestos":  {"asbestos","asbestos removal"},
}

def _normalize(s: str) -> str:
    return " ".join((s or "").lower().split())

def _family_from_structured_facets(job_type: str, work_cat: str, element: str, work_sub: str) -> Tuple[str, str]:
    fields = [
        ("job_type",      _normalize(job_type)),
        ("work_category", _normalize(work_cat)),
        ("element",       _normalize(element)),
        ("work_sub",      _normalize(work_sub)),
    ]
    for field_name, val in fields:
        if not val:
            continue
        for fam, vocab in FAMILY_KEYWORDS.items():
            for kw in vocab:
                if kw in val:
                    return fam, field_name
    return "misc", ""

def _infer_family_from_facets(facets: Dict[str, str]) -> str:
    fam, _ = _family_from_structured_facets(
        facets.get("job_type",""),
        facets.get("work_category",""),
        facets.get("element",""),
        facets.get("work_subcategory",""),
    )
    if fam != "misc":
        return fam

    # Fallback to facet-text regex
    s = (facets.get("facet_text") or "").lower()
    if re.search(r"\b(gutter|guttering|rain ?water (?:goods|system|gutter|pipe)|rwg|rwgs)\b", s): return "guttering"
    if re.search(r"\b(downpipe|down pipe|rwp|rain ?water pipe)\b", s): return "downpipe"
    if re.search(r"\b(fascia|fascias|barge ?board|bargeboards)\b", s): return "fascia"
    if re.search(r"\b(soffit|soffits)\b", s): return "soffit"
    if re.search(r"\b(ridge|hip|hips)\b", s): return "ridge"
    if re.search(r"\b(verge|verges)\b", s): return "verge"
    if re.search(r"\b(valley|valleys)\b", s): return "valley"
    if re.search(r"\b(chimney|flaunch)\b", s): return "chimney"
    if re.search(r"\b(flash|flashing|apron|soaker|abutment|step|cover|lead)\b", s): return "leadwork"
    if re.search(r"\b(scaffold|scaffolding)\b", s): return "scaffold"
    if re.search(r"\b(inspect|attendance|survey)\b", s): return "inspection"
    if re.search(r"\b(tile|tiles|slate|plain|concrete|clay)\b", s): return "tile"
    if re.search(r"\b(air ?brick|air-?vent|air ?vent|airbricks? and vents?)\b", s): return "airbrick"
    
    # NEW families from facet text
    if re.search(r"\b(felt|asphalt|flat roof|flat roofing|flat-roof)\b", s): return "flat_roof"
    if re.search(r"\b(insulation|insulate|loft insulation)\b", s): return "insulation"
    if re.search(r"\b(asbestos)\b", s): return "asbestos"
    if re.search(r"\b(green roof|green rooing|sedum|sedums)\b", s): return "green_roof"
    if re.search(r"\b(sheet roofing|sheet roof|sheeting|corrugated (?:iron|steel|plastic)|profiled sheet)\b", s): return "sheet_roof"
    if re.search(r"\b(joist|joists|rafters?|collar|strut|plate|partition|stud wall|timber floor)\b", s): return "structural_timber"
    if re.search(r"\b(cladding|shiplap|weatherboarding|feather edge)\b", s): return "cladding"
    if re.search(r"\b(pv|solar panel|photovoltaic)\b", s): return "solar_panels"
    if re.search(r"\b(woodworm|timber treatment|preservative guarantee)\b", s): return "timber_treatment"
    if re.search(r"\b(drone|call ?out|adjustment to invoice|vcs)\b", s): return "special"
    return "misc"

def _infer_action_from_text(s: str) -> Optional[str]:
    ACTION_PATTERNS = [
        ("CLEAN",   re.compile(r"\b(clean|clear|jet|wash|unblock)\b", re.I)),
        ("REFIX",   re.compile(r"\b(refix|repair|dress|redress|readjust|adjust|rebed)\b", re.I)),
        ("RENEW",   re.compile(r"\b(replace|renew|install|installed|installing|fit|fitted|fitting|supply\s*&\s*fit|supply\s+and\s+fit)\b", re.I)),
        ("REPOINT", re.compile(r"\b(repoint)\b", re.I)),
        ("REBUILD", re.compile(r"\b(rebuild)\b", re.I)),
    ]
    for label, rx in ACTION_PATTERNS:
        if rx.search(s or ""):
            return label
    return None

def _normalize_action_group(action: Optional[str]) -> str:
    a = (action or "").upper()
    if a in {"CLEAN"}: return "CLEAN"
    if a in {"REFIX"}: return "REFIX"
    if a in {"RENEW"}: return "RENEW"
    if a in {"REPOINT"}: return "REPOINT"
    if a in {"REBUILD"}: return "REBUILD"
    return "OTHER"

def _code_facets_to_family_action(code: str, matcher: JobCodeMatcher) -> Tuple[str, str, Dict[str,str]]:
    facets = _code_facets(code, matcher)
    action = _infer_action_from_text(facets["short_desc"]) or _infer_action_from_text(facets["facet_text"]) or "OTHER"
    fam = _infer_family_from_facets(facets)
    group = _normalize_action_group(action)
    return fam, group, facets

# --- Helper: robustly resolve a canonical code id ---
def _resolve_canonical_code(
    matcher: "JobCodeMatcher",
    preferred_id: str,
    fam_hint: str,
    action_hint: str,
    ranked_pool: list[tuple[str, float]],
) -> tuple[str | None, str]:
    """
    Returns (code_id_or_None, reason).
    Tries:
      (1) exact-ish id (strip/zeropad/int)
      (2) regex over SOR text for 'gutter.*(clean|flush)' (family/action checked)
      (3) best from current pool with family==fam_hint & action==action_hint
    """
    data = getattr(matcher, "job_code_data", {}) or {}

    def exists(cid: str) -> bool:
        return cid in data and bool(data[cid])

    # 1) exact-ish
    candidates = [preferred_id, str(preferred_id).strip()]
    try:
        as_int = str(int(float(preferred_id)))
        candidates += [as_int, as_int.zfill(len(preferred_id))]
    except Exception:
        pass
    for cid in candidates:
        if cid and exists(cid):
            return cid, f"exact:{cid}"

    # 2) regex search over SOR record text
    rx = re.compile(r"\bgutter(?:ing)?\b.*\b(clean|flush)\b|\b(clean|flush)\b.*\bgutter(?:ing)?\b", re.I)
    for cid, rec in data.items():
        blob = " ".join([
            rec.get("description") or "",
            rec.get("job_type") or "",
            rec.get("category") or "",
            rec.get("element") or "",
            rec.get("short_desc") or "",
        ])
        if rx.search(blob):
            facets = _code_facets(cid, matcher)
            fam = _infer_family_from_facets(facets)
            act = _normalize_action_group(
                _infer_action_from_text(facets.get("short_desc","")) or
                _infer_action_from_text(facets.get("facet_text",""))
            )
            if fam == fam_hint and (act == action_hint or act in {"CLEAN", "OTHER"}):
                return cid, f"regex:{cid}"

    # 3) fallback: best from ranked pool that matches family/action
    best = None; best_score = -1.0
    for cid, score in ranked_pool:
        facets = _code_facets(cid, matcher)
        fam = _infer_family_from_facets(facets)
        act = _normalize_action_group(
            _infer_action_from_text(facets.get("short_desc","")) or
            _infer_action_from_text(facets.get("facet_text",""))
        )
        if fam == fam_hint and (act == action_hint or action_hint == "ANY" or act == "CLEAN"):
            if score > best_score:
                best = cid; best_score = score
    if best:
        return best, f"pool:{best}"

    return None, "not_found"

# =============================
# Step 4 lightweight scoring
# =============================

def _score_step4_lightweight(
    matcher: JobCodeMatcher,
    global_pool: Dict[str, Dict],
    cleaned_texts: Dict[str, str],
    intents: List[Tuple[str, str]],
    subtype_hint: Optional[str],
    tile_count: Optional[int],
) -> Tuple[List[Tuple[str, float]], List[str]]:
    logs: List[str] = []
    full_text = " ".join(cleaned_texts.values())
    intent_actions = {vb.upper() for vb, _ in intents}
    roof_mats_text = _extract_roof_materials_from_text(full_text)

    re_scored: List[Tuple[str, float]] = []

    for code, info in global_pool.items():
        facets = _code_facets(code, matcher)
        facet_text = facets["facet_text"]
        short_desc = facets["short_desc"]
        code_action = _infer_action_from_text(short_desc) or _infer_action_from_text(facet_text)

        boost = 1.0
        notes = []

        if code_action and intent_actions:
            if code_action in intent_actions:
                boost += W_INTENT; notes.append(f"+intent({code_action})")
            else:
                conflict = (
                    ("CLEAN" in intent_actions and code_action in {"RENEW","REBUILD"}) or
                    ("RENEW" in intent_actions and code_action in {"CLEAN"}) or
                    ("REFIX" in intent_actions and code_action in {"RENEW","REBUILD"})
                )
                if conflict:
                    boost -= W_CONFLICT; notes.append(f"-intent_conflict({code_action})")

        sample = info.get("sample_chunk") or ""
        if sample:
            sample_intents = _intent_pairs_from_text(sample)
            sample_actions = {vb.upper() for vb, _ in sample_intents}
            if code_action and code_action in sample_actions:
                boost += W_LOCAL; notes.append("+local_action")

        if tile_count is not None and tile_count <= 10:
            if re.search(r"\b(tile|slate|plain)\b", short_desc, re.I):
                boost += W_SMALL; notes.append("+small_tile_hint")
            if OVERHAUL_RX.search(short_desc) or OVERHAUL_RX.search(facet_text):
                boost -= W_OVERHAUL; notes.append("-overhaul_smalljob")

        if roof_mats_text:
            facet_lower = (facet_text or "").lower()
            matched_any = False
            for m in roof_mats_text:
                if any(tok in facet_lower for tok in ROOF_MATERIAL_TOKENS.get(m, [])):
                    boost += 0.08; notes.append(f"+roof_mat({m})")
                    matched_any = True
            if not matched_any:
                for m, toks in ROOF_MATERIAL_TOKENS.items():
                    if m in roof_mats_text:
                        continue
                    if any(tok in facet_lower for tok in toks if len(tok) > 3):
                        boost -= 0.06; notes.append(f"-roof_mat_conflict({m})")
                        break

        boost = max(BOOST_MIN, min(BOOST_MAX, boost))
        new_score = float(info["score"]) * boost
        re_scored.append((code, new_score))

        if notes:
            logs.append(f"[score4] {code}: x{round(boost,3)} {' '.join(notes)} | base={round(info['score'],3)} -> {round(new_score,3)}")

    re_scored.sort(key=lambda x: (-x[1], -global_pool[x[0]]["score"], x[0]))
    return re_scored, logs

# =============================
# Step 6 stronger blending
# =============================

def _score_step6_stronger(
    matcher: JobCodeMatcher,
    ranked4: List[Tuple[str, float]],
    global_pool: Dict[str, Dict],
    per_field_pool: Dict[str, Dict[str, Dict]],
    cleaned_texts: Dict[str, str],
) -> Tuple[List[Tuple[str, float]], List[str]]:
    logs: List[str] = []
    full_text = " ".join(cleaned_texts.values()).lower()
    tokens = set(_tok(full_text))
    airbrick_cover_hint = ("vent cover" in full_text) or bool(re.search(r"(air ?brick|air-?vent|air ?vent).{0,12}cover", full_text))

    scope_large = bool(re.search(r"\b(whole|entire|full|all elevations|strip and re-?cover)\b", full_text))
    scope_small = bool(re.search(r"\b(couple|few|isolated|local|small|single)\b", full_text))

    def subarea_in_text(label: str) -> bool:
        return any(t in tokens for t in SUBAREA_TOKENS.get(label, set()))

    out: List[Tuple[str, float]] = []

    def code_field_hits(c: str) -> int:
        return sum(1 for f, fmap in per_field_pool.items() if c in fmap)

    def wd_chunk_density(c: str) -> int:
        return int(global_pool.get(c, {}).get("hits", 1))

    for rank_idx, (code, base5) in enumerate(ranked4):
        info = global_pool.get(code, {})
        facets = _code_facets(code, matcher)
        facet_text = (facets["facet_text"] or "")
        short_desc = (facets["short_desc"] or "")
        facet_lower = facet_text.lower()

        m = 1.0
        m_notes = []

        sample = info.get("sample_chunk") or ""
        fam_dense = 2 <= sum(1 for w in ("gutter","downpipe","ridge","valley","verge","chimney","flashing","lead","tile","slate") if w in sample.lower())
        if fam_dense:
            m += 0.04; m_notes.append("+fam_density")

        if scope_large and (OVERHAUL_RX.search(short_desc) or OVERHAUL_RX.search(facet_text)):
            m += 0.06; m_notes.append("+scope_large")
        if scope_small:
            if OVERHAUL_RX.search(short_desc) or OVERHAUL_RX.search(facet_text):
                m -= 0.08; m_notes.append("-scope_overhaul_small")
            if re.search(r"\b(tile|slate|plain)\b", short_desc, re.I):
                m += 0.06; m_notes.append("+scope_small_repair")

        for label in ("valley","verge","ridge","chimney","eaves","abutment"):
            if subarea_in_text(label) and label in facet_lower:
                m += 0.06; m_notes.append(f"+subarea({label})")

        m = max(MULT_MIN, min(MULT_MAX, m))

        add = 0.0
        a_notes = []

        # Prefer rebed/refix airbrick codes when the job text says "vent cover"
        if airbrick_cover_hint:
            if ("air brick" in facet_lower or "air-vent" in facet_lower or "air vent" in facet_lower or "vent" in facet_lower):
                # Strongly favour the rebed/refix line (115009) over install lines
                if re.search(r"\b(rebed|refix|loose)\b", facet_lower) or (code.strip() == "115009"):
                    add += 0.60; a_notes.append("+airbrick_cover_hint")
                # Mildly de-prefer explicit install/new lines in this context
                if re.search(r"\b(install|renew|new)\b", facet_lower):
                    add -= 0.15; a_notes.append("-install_vs_cover")

        if code_field_hits(code) >= 2:
            add += A_MULTI_FIELD; a_notes.append("+multi_field")
        if wd_chunk_density(code) >= 2:
            add += A_CHUNK_DENS; a_notes.append("+multi_chunk")
        if re.search(r"\bgeneral\b", facet_lower):
            add -= 0.02; a_notes.append("-generic")
        if len(_tok(sample)) > 40:
            add -= A_LONG_CHUNK; a_notes.append("-long_chunk")

        jt_word = None
        for w in ("chimney","gutter","downpipe","ridge","valley","verge","fascia","soffit","flashing","tile","slate","lead"):
            if w in full_text and w in facet_lower:
                jt_word = w; break
        if jt_word:
            add += A_JT_NUDGE; a_notes.append(f"+jt({jt_word})")

        new_score = base5 * m + add
        out.append((code, new_score))
        if m_notes or a_notes:
            logs.append(f"[score6] {code}: x{round(m,3)} {' '.join(m_notes)} {' '.join(a_notes)} | base5={round(base5,3)} -> {round(new_score,3)}")

    out.sort(key=lambda x: (-x[1], x[0]))
    return out, logs


# =============================
# Step 6.5: learned re-ranker helper
# =============================

def _apply_learned_reranker(
    ranked6: List[Tuple[str, float]],
    global_pool: Dict[str, Dict],
    form_texts: Dict[str, str],
    cleaned_texts: Dict[str, str],
    logs: List[str],
) -> List[Tuple[str, float]]:
    """
    Optional learned re-ranker applied after Step 6.

    - Uses get_reranker() from learn_core.
    - Builds a lightweight evidence pack from form_texts + cleaned_texts.
    - Returns a re-ordered list of (code, score) while preserving the numeric scores.
    - If the reranker is not enabled/ready or any error occurs, it returns ranked6 unchanged.
    """
    try:
        reranker = get_reranker()
    except NameError:
        # Imports not present or not wired yet
        logs.append("[reranker] get_reranker not available; skipping learned re-rank.")
        return ranked6
    except Exception as e:
        logs.append(f"[reranker] Could not initialise reranker: {e}")
        return ranked6

    try:
        if not reranker.is_ready():
            return ranked6

        # Build free-text view and blob
        free_text_map = {k: v for k, v in cleaned_texts.items() if isinstance(v, str) and v.strip()}
        text_blob = " ".join(free_text_map.values()).strip()

        try:
            pack = build_evidence_pack(
                instance=None,
                form_data=form_texts,
                free_text_map=free_text_map,
            )
        except NameError:
            logs.append("[reranker] build_evidence_pack not available; skipping learned re-rank.")
            return ranked6

        base_scores_map = {code: score for code, score in ranked6}
        ranked_codes_only = [code for code, _ in ranked6]

        new_order = reranker.rerank_codes(
            ranked_codes_only,
            pack=pack,
            form=form_texts,
            text_blob=text_blob,
            base_scores=base_scores_map,
        )

        score_map = {c: s for c, s in ranked6}
        re_scored = [(c, score_map.get(c, 0.0)) for c in new_order]
        logs.append("[reranker] Applied learned re-ranker to Step 6 candidates.")
        return re_scored

    except Exception as e:
        logs.append(f"[reranker] Disabled at runtime due to error: {e}")
        return ranked6


# =============================
# Step 5: caps, diversity, duplicates
# =============================

# --- ADDED: robust fuzzy keyword matcher ---
_word_sep_re = re.compile(r"[\s\-\_\/]+")

def _norm_kw(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-\/]", "", s)  # drop punctuation except word/space/hyphen/slash
    s = _word_sep_re.sub(" ", s).strip()
    return s

def _generate_windows(text: str, max_len: int = 2) -> List[str]:
    toks = _word_sep_re.split(text.lower())
    toks = [t for t in toks if t]  # drop empties
    wins: List[str] = []
    for n in range(1, max_len + 1):
        for i in range(0, max(0, len(toks) - n + 1)):
            wins.append(" ".join(toks[i:i+n]))
    return wins

def _fuzzy_hit_kw(text_blob: str, keywords: set, threshold: float = 0.82) -> bool:
    """
    Typo-tolerant fuzzy hit:
    - normalises hyphens/spaces,
    - matches 1–2 word windows,
    - uses existing _sim_ratio for robustness without new deps.
    """
    if not text_blob or not keywords:
        return False

    # Fast path: direct containment on normalised forms
    norm_text = _norm_kw(text_blob)
    norm_kws = {_norm_kw(k) for k in keywords}

    for kw in norm_kws:
        if kw and kw in norm_text:
            return True

    # Fuzzy path: compare token windows up to 2 words
    wins = _generate_windows(norm_text, max_len=2)
    for w in wins:
        for kw in norm_kws:
            if not kw:
                continue
            # exact first
            if w == kw:
                return True
            # similarity compare (uses your existing _sim_ratio)
            if _sim_ratio(w, kw) >= threshold:
                return True
    return False


def _variable_family_caps_from_text(cleaned: Dict[str, str], tile_count: Optional[int]) -> Dict[str, int]:
    """
    Variable caps keyed off text presence + structured hints.
    """
    text_blob = " ".join(cleaned.values()).lower()

    def present_field(key: str) -> bool:
        v = cleaned.get(key, "")
        return not _is_noise(v)

    # --- CHANGED: typo-tolerant fuzzy detection with tuned thresholds ---
    has_tile     = _fuzzy_hit_kw(text_blob, {"tile","tiles","slate","plain","concrete","clay"}, threshold=0.84)
    has_gutter   = _fuzzy_hit_kw(text_blob, {"gutter","gutters","guttering"}, threshold=0.82)
    has_down     = _fuzzy_hit_kw(text_blob, {"downpipe","downpipes","rwp","rain water pipe","rainwater pipe"}, threshold=0.84)
    has_ridge    = _fuzzy_hit_kw(text_blob, {"ridge","hip","hips"}, threshold=0.84)
    has_verge    = _fuzzy_hit_kw(text_blob, {"verge","verges"}, threshold=0.84)
    has_valley   = _fuzzy_hit_kw(text_blob, {"valley","valleys"}, threshold=0.84)
    has_fascia   = _fuzzy_hit_kw(text_blob, {"fascia","fascias","facia","barge","bargeboard","bargeboards"}, threshold=0.78)
    has_soffit   = _fuzzy_hit_kw(text_blob, {"soffit","soffits"}, threshold=0.82)
    has_scaff    = _fuzzy_hit_kw(text_blob, {"scaffold","scaffolding"}, threshold=0.84)
    has_airbrick = _fuzzy_hit_kw(text_blob, {"airbrick","air brick","air-vent","air vent","airvent"}, threshold=0.80)
    # NEW: signals for the new families (all via free text, so all fields are still "looked at")
    has_flat     = _fuzzy_hit_kw(text_blob, {"flat roof","felt roof","felt roofing","asphalt roof","balcony roof"}, threshold=0.80)
    has_sheet    = _fuzzy_hit_kw(text_blob, {"sheet roof","sheeting","corrugated","profiled sheet"}, threshold=0.80)
    has_cladding = _fuzzy_hit_kw(text_blob, {"cladding","shiplap","weatherboard","weatherboarding","feather edge"}, threshold=0.80)
    has_struct   = _fuzzy_hit_kw(text_blob, {"joist","joists","timber floor","partition","stud wall","rafter","rafters","collar","strut","plate","hanger"}, threshold=0.80)
    has_insul    = _fuzzy_hit_kw(text_blob, {"insulation","insulate","loft insulation"}, threshold=0.80)
    has_green    = _fuzzy_hit_kw(text_blob, {"green roof","sedum","sedums"}, threshold=0.80)
    has_asb      = _fuzzy_hit_kw(text_blob, {"asbestos"}, threshold=0.90)
    has_solar    = _fuzzy_hit_kw(text_blob, {"pv","solar panel","photovoltaic"}, threshold=0.80)
    has_treat    = _fuzzy_hit_kw(text_blob, {"woodworm","timber treatment","preservative"}, threshold=0.80)
    has_special  = _fuzzy_hit_kw(text_blob, {"drone","call out","call-out","adjustment to invoice","special","bird guard","bird guards"}, threshold=0.80)

    caps: Dict[str, int] = {}

    # Leadwork / Chimney based on comments present
    caps["leadwork"] = 2 if present_field("6.3_Leadwork_Comment") else 0
    caps["chimney"] = 4 if (present_field("7.3_Chimney_Comment") or present_field("7.5_Chimney_Flaunch_Comment")) else 0

    # Tile family: only allow if *actual* tile evidence exists
    tile_cap = 0
    if has_tile:
        tile_cap = 2
        if tile_count is not None and tile_count <= 10:
            tile_cap = 1

    # structured roof hint opens tiles slightly
    structured_roof = any(not _is_noise(cleaned.get(k, "")) for k in (
        "4.1_Roof_Type","4.2_Coverings_Type","4.3_Tile_Size","4.4_Roof_Measurement",
        "4.5_Flat_Coverings_Type","4.6_Flat_Tile_Size","4.7_Flat_Measurement",
        "4.8_Other_Coverings_Type","4.9_Other_Tile_Size","4.10_Other_Measurement",
    ))
    if structured_roof and tile_cap == 0:
        tile_cap = 1
    caps["tile"] = tile_cap

    # Rainwater & roofline caps (using your 8.x/9.x/10.x ranges)
    gutter_action = (cleaned.get("9.1_Guttering","") or "").strip().lower()
    caps["guttering"] = 1 if has_gutter or gutter_action in {"replace","refix","clean"} else 0

    has_down_struct = any(not _is_noise(cleaned.get(k, "")) for k in (
        "10.1_RWP","10.2_RWP_Replace","10.3_RWP_Replace_Measurement","10.4_RWP_Refix",
    ))
    caps["downpipe"] = 1 if has_down or has_down_struct else 0

    fascia_struct = any(not _is_noise(cleaned.get(k, "")) for k in ("8.1_Fascia","8.2_Fascia_Measurement"))
    caps["fascia"] = 1 if has_fascia or fascia_struct else 0

    soffit_struct = any(not _is_noise(cleaned.get(k, "")) for k in ("8.3_Soffit","8.4_Soffit_Measurement"))
    caps["soffit"] = 1 if has_soffit or soffit_struct else 0

    # Ridge / verge / valley remain primarily text-led; 5.x fields are already in text_blob
    caps["ridge"]  = 1 if has_ridge else 0
    caps["verge"]  = 1 if has_verge else 0
    caps["valley"] = 1 if has_valley else 0

    # NEW families: conservative caps, using your 4.x mappings where relevant
    has_flat_struct = any(not _is_noise(cleaned.get(k, "")) for k in (
        "4.5_Flat_Coverings_Type","4.6_Flat_Tile_Size","4.7_Flat_Measurement",
    ))
    roof_type_val = (cleaned.get("4.1_Roof_Type","") or "").lower()
    flat_cap = 0
    if has_flat or has_flat_struct or "flat" in roof_type_val:
        flat_cap = 2
    caps["flat_roof"] = flat_cap

    caps["sheet_roof"]        = 1 if has_sheet else 0
    caps["green_roof"]        = 1 if has_green else 0
    caps["insulation"]        = 2 if has_insul else 0
    caps["structural_timber"] = 2 if has_struct else 0
    caps["cladding"]          = 1 if has_cladding else 0
    caps["solar_panels"]      = 1 if has_solar else 0
    caps["timber_treatment"]  = 1 if has_treat else 0
    caps["asbestos"]          = 1 if has_asb else 0
    caps["special"]           = 1 if has_special else 0

    # Inspection, scaffold, airbrick as before
    caps["inspection"] = 1
    caps["scaffold"]   = 1 if has_scaff else 0
    caps["airbrick"]   = 1 if has_airbrick else 0

    # Visit-type policy for inspection caps
    visit_type_text = (cleaned.get("1.1_Visit_Type") or "").strip().lower()
    if visit_type_text == "inspection and repair":
        caps["inspection"] = 1
    elif visit_type_text == "inspection only":
        caps["inspection"] = 1
    else:  # "repair only" or anything else
        caps["inspection"] = 0

    caps["scaffold"] = 1 if has_scaff else 0

    # Misc only if nothing else
    non_misc_total = sum(v for k, v in caps.items() if k != "misc")
    caps["misc"] = 1 if non_misc_total == 0 else 0
    return caps

def _collapse_near_duplicates(selected: List[Tuple[str, float, str, str, Dict[str,str]]], threshold: float = 0.90) -> List[Tuple[str, float, str, str, Dict[str,str]]]:
    out: List[Tuple[str, float, str, str, Dict[str,str]]] = []
    for code, score, fam, act, facets in selected:
        dup = False
        for c2, s2, fam2, act2, fac2 in out:
            if fam == fam2 and act == act2:
                if _sim_ratio(facets.get("facet_text",""), fac2.get("facet_text","")) >= threshold:
                    dup = True
                    break
        if not dup:
            out.append((code, score, fam, act, facets))
    return out


# =============================
# Step 7: INLINE RULES (minimal)
# =============================
def _apply_inline_rules(
    matcher: JobCodeMatcher,
    ranked6: List[Tuple[str, float]],
    global_pool: Dict[str, Dict],
    cleaned_texts: Dict[str, str],
    intents: List[Tuple[str, str]],
    tile_count: Optional[int],
) -> Tuple[List[Tuple[str, float]], Dict[str,int], Set[str], List[str], List[str]]:
    logs: List[str] = []
    excludes: Set[str] = set()
    cap_overrides: Dict[str, int] = {}
    special_adds: List[str] = []

    # Build the unified text
    full_text = " ".join((cleaned_texts or {}).values()).strip()
    full_text_lc = full_text.lower()

    # ---- Existing guardrails ----
    # Small-tiles guardrail
    if tile_count is not None and tile_count <= 10:
        cap_overrides["tile"] = max(1, cap_overrides.get("tile", 0))


    # ---------------------------------------------------------------------
    # NEW: INSPECTION CANONICALISATION DRIVEN BY VISIT TYPE
    # ---------------------------------------------------------------------
    visit_type_text = (cleaned_texts.get("1.1_Visit_Type") or "").strip().lower()
    if visit_type_text in {"inspection and repair", "inspection only"}:
        canonical_id = "221001"
        cap_overrides["inspection"] = max(1, cap_overrides.get("inspection", 0))

        # inject/promote canonical to #1
        present = any(c == canonical_id for c, _ in ranked6)
        if not present:
            ranked6 = [(canonical_id, 1e9)] + ranked6
            logs.append(f"[rules][inspect] Injected canonical {canonical_id} at #1 (visit_type={visit_type_text}).")
        else:
            ranked6 = [(c, s) for (c, s) in ranked6 if c != canonical_id]
            ranked6 = [(canonical_id, 1e9)] + ranked6
            logs.append(f"[rules][inspect] Promoted canonical {canonical_id} to #1 (visit_type={visit_type_text}).")

        # exclude any other inspection-family codes
        other_inspection: Set[str] = set()
        for code, _ in ranked6[1:]:
            facets = _code_facets(code, matcher)
            fam = _infer_family_from_facets(facets)
            if fam == "inspection" and code != canonical_id:
                other_inspection.add(code)
        if other_inspection:
            excludes.update(other_inspection)
            logs.append(f"[rules][inspect] Excluding non-canonical inspection codes: {sorted(other_inspection)}")


    # ---------------------------------------------------------------------
    # GUTTER CLEAN SPECIALISATION (existing)
    # ---------------------------------------------------------------------
    verb_near_gutter = (
        re.search(
            r"(clean(?:ed|ing)?|clear(?:ed|ing)?|wash(?:ed|ing)?|flush(?:ed|ing)?|"
            r"jet(?: ?wash)?(?:ed|ing)?|vac(?:ced)?|vacuum(?:ed|ing)?|unblock(?:ed|ing)?|"
            r"(?:clean|clear|empty)\s*out)"
            r"[^\.]{0,50}\b(gutter(?:s|ing)?)\b",
            full_text_lc, flags=re.I
        )
        or
        re.search(
            r"\b(gutter(?:s|ing)?)\b[^\.]{0,50}"
            r"(clean(?:ed|ing)?|clear(?:ed|ing)?|wash(?:ed|ing)?|flush(?:ed|ing)?|"
            r"jet(?: ?wash)?(?:ed|ing)?|vac(?:ced)?|vacuum(?:ed|ing)?|unblock(?:ed|ing)?|"
            r"(?:clean|clear|empty)\s*out)",
            full_text_lc, flags=re.I
        )
    )

    is_downpipe_context = re.search(
        r"\b(down ?pipe|rwp|rain ?water ?pipe|rainwater ?pipe)\b",
        full_text_lc, flags=re.I
    )

    is_gutter_clean = (verb_near_gutter is not None) and (is_downpipe_context is None)
    logs.append(f"[gc] trigger={bool(is_gutter_clean)} text='{full_text_lc[:120]}'")

    if is_gutter_clean:
        canonical_id = "603903"
        cap_overrides["guttering"] = max(1, cap_overrides.get("guttering", 0))
        cap_overrides["downpipe"]  = max(1, cap_overrides.get("downpipe", 0))
        present = any(c == canonical_id for c, _ in ranked6)
        if not present:
            ranked6 = [(canonical_id, 1e9)] + ranked6
            logs.append(f"[rules][gc] Injected canonical {canonical_id} into ranked pool at #1.")
        else:
            ranked6 = [(c, s) for (c, s) in ranked6 if c != canonical_id]
            ranked6 = [(canonical_id, 1e9)] + ranked6
            logs.append(f"[rules][gc] Promoted canonical {canonical_id} to #1.")

        others = [c for (c, _) in ranked6[1:]]
        if others:
            excludes.update(others)
            logs.append(f"[rules][gc] Excluding non-canonical candidates for gutter-clean: {others}")

        logs.append("[rules][gc] Hard override active → output will be ONLY 603903.")

    # ---------------------------------------------------------------------
    # NEW: VERTICAL-TILING GUARDRAIL
    # ---------------------------------------------------------------------
    # If the job text does NOT describe vertical tiles/cladding, exclude vertical-only codes.
    vertical_present = bool(re.search(
        r"\b(vertical|hung\s+tiles?|tile\s+hanging|hanging\s+tiles?|cladding)\b",
        full_text_lc, flags=re.I
    ))

    if not vertical_present:
        vertical_like: Set[str] = set()
        for code, _score in ranked6:
            facets = _code_facets(code, matcher)
            blob = " ".join([
                facets.get("short_desc","") or "",
                facets.get("facet_text","") or "",
            ]).lower()

            if re.search(r"\b(vertical|hung\s+tile|tile\s+hanging|cladding)\b", blob):
                vertical_like.add(code)

        if vertical_like:
            excludes.update(vertical_like)
            logs.append(f"[rules][vertical] No vertical indicators in text → excluding: {sorted(vertical_like)}")

    # ---------------------------------------------------------------------
    # NEW: HANGING SLATES — NE 5 NO vs OVER 5 NO (mutual exclusivity + selection)
    # ---------------------------------------------------------------------
    # Only one of these should survive. Decide using tile_count:
    # - None or <=5  → keep 207001 (NE 5 NO)
    # - >=6          → keep 207003 (OVER 5 NO)
    HANGING_NE = "207001"  # SLATE, RENEW HANGING SLATE NE 5 NO
    HANGING_OV = "207003"  # SLATE, RENEW HANGING SLATES OVER 5 NO

    in_scope = {c for c, _ in ranked6}
    present_candidates = {c for c in (HANGING_NE, HANGING_OV) if (c in in_scope or c in global_pool)}

    if present_candidates:
        chosen = HANGING_NE if (tile_count is None or tile_count <= 5) else HANGING_OV
        other  = HANGING_OV if chosen == HANGING_NE else HANGING_NE

        # exclude the non-chosen variant (even if not currently ranked; handled upstream)
        excludes.add(other)

        # gently request the chosen code be added if it isn't already in ranked but is known globally
        if chosen not in in_scope and chosen in global_pool:
            special_adds.append(chosen)

        logs.append(f"[rules][slate_ne_over] count={tile_count} ⇒ keep {chosen}, exclude {other}")

    # ---------------------------------------------------------------------
    # (Keep any other inline rules below)
    # ---------------------------------------------------------------------

    return ranked6, cap_overrides, excludes, special_adds, logs




# =============================
# Step 8: Recruitment rails (tiny editable mapping)
# =============================

CANONICAL_CODES = {
    # --- Guttering (keep CLEAN as-is) ---
    ("guttering", "CLEAN"): "603903",            # Clean out debris from gutters (per elevation)
    ("guttering", "REFIX_CAST_IRON"): "603305",  # Realign cast iron gutter
    ("guttering", "REFIX_PVC"): "603109",        # Realign PVCu gutter
    ("guttering", "REPLACE_CAST_IRON"): "603301",# Renew cast iron gutter complete
    ("guttering", "REPLACE_PVC"): "603101",      # Renew 112mm PVCu gutter complete

    # --- Downpipe ---
    ("downpipe", "REFIX_CAST_IRON"): "601505",   # Remove and refix cast iron downpipe complete
    ("downpipe", "REFIX_PVC"): "601121",         # Remove and refix PVCu downpipe complete
    ("downpipe", "REPLACE_CAST_IRON"): "601503", # Renew up to 100mm cast iron pipe
    ("downpipe", "REPLACE_PVC"): "601105",       # Renew PVCu downpipe

    # --- Fascia (timber / PVCu) ---
    ("fascia", "REFIX_PVC"): "303015",          # Refix fascia/soffit/barge (generic)
    ("fascia", "REFIX_TIMBER"): "303015",       # Refix fascia/soffit/barge (generic, used for timber too)
    ("fascia", "REPLACE_PVC"): "303005",        # Renew fascia/bargeboard in PVCu (≤300mm)
    ("fascia", "REPLACE_TIMBER"): "303001",     # Renew fascia/bargeboard in softwood (≤300mm)

    # --- Soffit (timber / PVCu) ---
    ("soffit", "REFIX_PVC"): "303015",          # Refix fascia/soffit/barge (generic)
    ("soffit", "REFIX_TIMBER"): "303015",       # Refix fascia/soffit/barge (generic)
    ("soffit", "REPLACE_PVC"): "303013",        # Renew soffit in PVCu (≤450mm)
    ("soffit", "REPLACE_TIMBER"): "303007",     # Renew soffit in softwood (≤450mm)

    # --- Ridge (half round focus) ---
    ("ridge", "RENEW_HALF_ROUND"): "203503",    # Renew half-round or roll-top ridge/hip tile to slating
    ("ridge", "REPOINT_HALF_ROUND"): "201717",  # Rake out and repoint ridge/hip/valley tiles

    # --- Verge ---
    ("verge", "REFIX"): "201505",               # Remove and refix verge tiles
    ("verge", "RENEW"): "201503",               # Renew tile verge
    ("verge", "REPLACE"): "201503",             # Renew tile verge (used as replace)
    ("verge", "REPOINT"): "201501",             # Rake out and repoint verge in mortar

    # --- Valley ---
    ("valley", "REFIX"): "231025",              # Repair leak to lead valley
    ("valley", "RENEW"): "201731",              # Renew valley tiles
    ("valley", "REPLACE"): "201731",            # Renew valley tiles (used as replace)
    ("valley", "REPOINT"): "201731",            # Fallback repoint/renew for valley tiles

    # --- Chimney (pointing / flaunch) ---
    ("chimney", "FLAUNCH"): "130009",           # Rebed pot and make good flaunching
    ("chimney", "POINTING"): "120029",          # Rake out and repoint chimney stack

    # --- Leadwork (flashing / apron) ---
    ("leadwork", "APRON"): "231011",            # Renew lead apron flashing (≈300mm girth)
    ("leadwork", "FLASHING"): "231005",         # Renew lead cover flashing (≈150mm girth)

    # --- Tile / roof coverings by type ---
    ("tile", "CLAY"): "201301",                 # Renew plain concrete or clay roof tiles (≤10)
    ("tile", "CONCRETE"): "201101",             # Renew concrete interlocking tiles (≤5)
    ("tile", "FELT"): "217001",                 # Renew felt roofing – 2-layer, plain/mineral finish
    ("tile", "GRP"): "215001",                  # Sweep and apply waterproofing compound (liquid covering)
    ("tile", "SLATE"): "203301",                # Renew natural slate to roof (≤5 slates)

    # --- Airbrick (existing family) ---
    ("airbrick", "REFIX"): "115009",            # Rebed loose vent/airbrick
    ("airbrick", "RENEW"): "115001",            # Renew airbrick with PVC unit

    # --- Flat roof / sheet / green roof / insulation ---
    ("flat_roof", "ANY"): "217001",             # Renew felt roofing – 2-layer, plain/mineral finish
    ("sheet_roof", "ANY"): "213001",            # Renew galvanised corrugated iron sheeting
    ("green_roof", "ANY"): "210001",            # Annual maintenance – sedum green roof
    ("insulation", "ANY"): "227005",            # Lay loft quilt insulation up to 270mm thick

    # --- Structural timber / cladding / solar / timber treatment ---
    ("structural_timber", "ANY"): "301103",     # Renew softwood floor joist (≈100mm deep)
    ("cladding", "ANY"): "307101",              # Renew 19mm timber shiplap cladding (≤2m²)
    ("solar_panels", "ANY"): "298001",          # Client inspection – roofing (neutral solar/roof visit fallback)
    ("timber_treatment", "ANY"): "305822",      # Chemical treat timber (roof timber treatment)

    # --- Asbestos / special ---
    ("asbestos", "ANY"): "Asbestos Removal Cost",        # Asbestos removal cost line (SPECIAL)
    ("special", "ANY"): "Supplied and Fitted Bird Guards",# Supply and fit bird guards (SPECIAL)

    # --- Inspection (keep as-is) ---
    ("inspection", "ANY"): "221001",            # Provide and erect ladder for inspection
}

def _recruit_missing_families(
    matcher: JobCodeMatcher,
    selected_codes: List[str],
    ranked6: List[Tuple[str, float]],
    global_pool: Dict[str, Dict],
    cleaned_texts: Dict[str, str],
    intents: List[Tuple[str, str]],
    family_caps: Dict[str, int],
    logs: List[str],
) -> Tuple[List[Tuple[str, float]], List[str]]:
    special_adds: List[str] = []

    def family_empty(fam: str) -> bool:
        return not any(
            c for c in selected_codes
            if _infer_family_from_facets(_code_facets(c, matcher)) == fam
        )

    def _norm_field(val: Optional[str]) -> str:
        return (val or "").strip().upper()

    # ---------------------------------------------
    # Extract structured hints from form fields
    # ---------------------------------------------
    def _extract_family_hints_from_fields(ct: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        hints: Dict[str, Dict[str, str]] = {}

        # --- Guttering: material + action from 9.x ---
        g_act = _norm_field(ct.get("9.1_Guttering"))
        if g_act:
            info: Dict[str, str] = {}
            mat = ""
            if "CLEAN" in g_act:
                info["base_action"] = "CLEAN"
                info["action_key"] = "CLEAN"
            elif "REFIX" in g_act:
                info["base_action"] = "REFIX"
                m = _norm_field(ct.get("9.4_Guttering_Refix"))
                if "CAST" in m:
                    mat = "CAST_IRON"
                elif "PVC" in m:
                    mat = "PVC"
            elif "REPLACE" in g_act:
                # Map REPLACE -> RENEW action group
                info["base_action"] = "RENEW"
                m = _norm_field(ct.get("9.2_Guttering_Replace"))
                if "CAST" in m:
                    mat = "CAST_IRON"
                elif "PVC" in m:
                    mat = "PVC"
            if mat:
                info["material"] = mat
                if "REFIX" in g_act:
                    info["action_key"] = f"REFIX_{mat}"
                elif "REPLACE" in g_act:
                    info["action_key"] = f"REPLACE_{mat}"
            if info:
                hints["guttering"] = info

        # --- Downpipe: material + action from 10.x ---
        d_act = _norm_field(ct.get("10.1_RWP"))
        if d_act:
            info = {}
            mat = ""
            if "CLEAN" in d_act:
                info["base_action"] = "CLEAN"
                info["action_key"] = "CLEAN"
            elif "REFIX" in d_act:
                info["base_action"] = "REFIX"
                m = _norm_field(ct.get("10.4_RWP_Refix"))
                if "CAST" in m:
                    mat = "CAST_IRON"
                elif "PVC" in m:
                    mat = "PVC"
            elif "REPLACE" in d_act:
                info["base_action"] = "RENEW"
                m = _norm_field(ct.get("10.2_RWP_Replace"))
                if "CAST" in m:
                    mat = "CAST_IRON"
                elif "PVC" in m:
                    mat = "PVC"
            if mat:
                info["material"] = mat
                if "REFIX" in d_act:
                    info["action_key"] = f"REFIX_{mat}"
                elif "REPLACE" in d_act:
                    info["action_key"] = f"REPLACE_{mat}"
            if info:
                hints["downpipe"] = info

        # --- Fascia: material from 8.1 ---
        f_mat = _norm_field(ct.get("8.1_Fascia"))
        if f_mat:
            info = {}
            if "PVC" in f_mat:
                info["material"] = "PVC"
            elif "TIMBER" in f_mat:
                info["material"] = "TIMBER"
            if info:
                hints["fascia"] = info

        # --- Soffit: material from 8.3 ---
        s_mat = _norm_field(ct.get("8.3_Soffit"))
        if s_mat:
            info = {}
            if "PVC" in s_mat:
                info["material"] = "PVC"
            elif "TIMBER" in s_mat:
                info["material"] = "TIMBER"
            if info:
                hints["soffit"] = info

        # --- Ridge: action from 5.3, default to HALF ROUND for SOR selection ---
        r_job = _norm_field(ct.get("5.3_Ridge_Job"))
        if r_job:
            info = {}
            if "REPOINT" in r_job:
                info["base_action"] = "REPOINT"
                info["action_key"] = "REPOINT_HALF_ROUND"
            elif "RE-NEW" in r_job or "RENEW" in r_job or "RE NEW" in r_job:
                info["base_action"] = "RENEW"
                info["action_key"] = "RENEW_HALF_ROUND"
            if info:
                hints["ridge"] = info

        # --- Chimney: POINTING / FLAUNCH from 7.1 ---
        c_job = _norm_field(ct.get("7.1_Chimney"))
        if c_job:
            info = {}
            if "POINT" in c_job:
                info["action_key"] = "POINTING"
            elif "FLAUNCH" in c_job:
                info["action_key"] = "FLAUNCH"
            if info:
                hints["chimney"] = info

        # --- Leadwork: FLASHINGS / APRON from 6.1 ---
        l_job = _norm_field(ct.get("6.1_Leadwork"))
        if l_job:
            info = {}
            if "FLASH" in l_job:
                info["action_key"] = "FLASHING"
            elif "APRON" in l_job:
                info["action_key"] = "APRON"
            if info:
                hints["leadwork"] = info

        # --- Tile: covering type from 4.2 / 4.8 ---
        cov = _norm_field(ct.get("4.2_Coverings_Type") or ct.get("4.8_Other_Coverings_Type"))
        if cov:
            info = {}
            if "SLATE" in cov:
                info["material"] = "SLATE"
            elif "CONCRETE" in cov:
                info["material"] = "CONCRETE"
            elif "CLAY" in cov:
                info["material"] = "CLAY"
            elif "FELT" in cov:
                info["material"] = "FELT"
            elif "GRP" in cov:
                info["material"] = "GRP"
            if info:
                hints["tile"] = info

        return hints

    family_hints = _extract_family_hints_from_fields(cleaned_texts or {})

    # ---------------------------------------------
    # Annotate pool with family + action
    # ---------------------------------------------
    annotated_pool: List[Tuple[str, float, str, str]] = []
    for code, score in ranked6:
        facets = _code_facets(code, matcher)
        fam = _infer_family_from_facets(facets)
        act = _normalize_action_group(
            _infer_action_from_text(facets["short_desc"])
            or _infer_action_from_text(facets["facet_text"])
        )
        annotated_pool.append((code, score, fam, act))

    # Median score across the pool – used as a strength reference
    median_score = 0.0
    if annotated_pool:
        scores_sorted = sorted(s for _, s, _, _ in annotated_pool)
        mid = len(scores_sorted) // 2
        if len(scores_sorted) % 2 == 1:
            median_score = scores_sorted[mid]
        else:
            median_score = 0.5 * (scores_sorted[mid - 1] + scores_sorted[mid])

    # Families we actively try to ensure are present (if caps allow)
    target_fams = [
        "guttering",
        "fascia",
        "inspection",
        "leadwork",
        "tile",
        "chimney",
        "ridge",
        "verge",
        "valley",
        "airbrick",
    ]

    # ---------------------------------------------
    # Helper: basic material/covering match
    # ---------------------------------------------
    def _material_matches(fam: str, hint: Dict[str, str], code_id: str) -> bool:
        mat = hint.get("material") or hint.get("covering")
        if not mat:
            return False
        facets = _code_facets(code_id, matcher)
        blob = (
            (facets.get("short_desc", "") or "")
            + " "
            + (facets.get("facet_text", "") or "")
        ).lower()

        if fam in {"guttering", "downpipe", "fascia", "soffit"}:
            if mat == "PVC":
                return "pvc" in blob or "u.p.v.c" in blob or "upvc" in blob
            if mat == "CAST_IRON":
                return "cast iron" in blob
            if mat == "TIMBER":
                return "timber" in blob or "softwood" in blob or "hardwood" in blob

        if fam == "tile":
            if mat == "SLATE":
                return "slate" in blob
            if mat == "CONCRETE":
                return "concrete" in blob
            if mat == "CLAY":
                return "clay" in blob or "plain tile" in blob
            if mat == "FELT":
                return "felt" in blob
            if mat == "GRP":
                return "grp" in blob or "liquid" in blob

        if fam == "leadwork":
            # Use action_key more than material, but as a weak hint:
            ak = hint.get("action_key") or ""
            if "APRON" in ak:
                return "apron" in blob
            if "FLASH" in ak:
                return "flash" in blob

        if fam == "chimney":
            ak = hint.get("action_key") or ""
            if "POINT" in ak:
                return "point" in blob or "repoint" in blob
            if "FLAUNCH" in ak:
                return "flaunch" in blob or "flaunching" in blob

        return False

    # ---------------------------------------------
    # Helper: find a "clear winner" from the pool
    # ---------------------------------------------
    def _pick_clear_family_winner(fam: str) -> Optional[Tuple[str, float]]:
        # Collect all candidates for this family
        fam_cands: List[Tuple[str, float, str]] = [
            (code, score, act)
            for (code, score, pfam, act) in annotated_pool
            if pfam == fam
        ]
        if not fam_cands:
            return None

        hint = family_hints.get(fam, {})

        # 1) Prefer candidates matching the hinted base_action, if any
        filtered = fam_cands
        base_action = hint.get("base_action")
        if base_action:
            by_action = [c for c in filtered if c[2] == base_action]
            if by_action:
                filtered = by_action

        # 2) Within that, prefer candidates whose SOR text matches the material/covering, if any
        mat = hint.get("material") or hint.get("covering")
        if mat:
            by_mat: List[Tuple[str, float, str]] = []
            for code, score, act in filtered:
                if _material_matches(fam, hint, code):
                    by_mat.append((code, score, act))
            if by_mat:
                filtered = by_mat

        fam_cands = filtered

        # Sort by score descending
        fam_cands.sort(key=lambda x: -x[1])
        best_code, best_score, _ = fam_cands[0]
        second_score = fam_cands[1][1] if len(fam_cands) > 1 else None

        # Threshold relative to global median
        strength_floor = 0.75 * median_score if median_score > 0.0 else 0.0

        # If best is very weak relative to the overall pool, treat as "no clear"
        if median_score > 0.0 and best_score < strength_floor:
            return None

        # Require a small margin over the next-best in the same family
        margin = 0.02
        if second_score is not None and best_score < second_score + margin:
            return None

        # Clear enough: return with score floored up a bit for stability
        final_score = max(best_score, strength_floor)
        return best_code, final_score

    # ---------------------------------------------
    # Main recruitment loop
    # ---------------------------------------------
    for fam in target_fams:
        # Respect caps
        if fam not in family_caps or family_caps[fam] <= 0:
            continue

        # If the family is already present in selected_codes, skip
        if not family_empty(fam):
            continue

        recruited: Optional[Tuple[str, float]] = None

        # 1) POOL-FIRST: see if there is a clear, strong winner in the scored pool
        winner = _pick_clear_family_winner(fam)
        if winner is not None:
            recruited = winner
            logs.append(f"[recruit] +{fam} via POOL_CLEAR id={winner[0]}")
        else:
            # 2) CANONICAL AS FALLBACK: only when no clear pool code exists
            hint = family_hints.get(fam, {})
            preferred_actions: List[str] = []

            # First, try any high-precision action_key derived from fields
            ak = hint.get("action_key")
            if ak:
                preferred_actions.append(ak)

            # Then fall back to the generic action groups
            preferred_actions += ["REFIX", "CLEAN", "RENEW", "REPOINT", "REBUILD", "ANY"]

            for act in preferred_actions:
                code_id = CANONICAL_CODES.get((fam, act))
                if not code_id:
                    continue
                rec = getattr(matcher, "job_code_data", {}).get(code_id)
                if rec is None:
                    continue

                # Inspection canonical can be injected even if overall scores are low
                if fam == "inspection":
                    floor = 0.0
                else:
                    floor = 0.75 * median_score if median_score > 0.0 else 0.0

                recruited = (code_id, max(floor, median_score))
                logs.append(f"[recruit] +{fam} via CANON_FALLBACK ({act}) id={code_id}")
                if fam == "inspection":
                    special_adds.append(code_id)
                break

            # 3) If canonical also fails, we do NOT force a weak pool code
            # (no further action; the family simply won't be recruited)

        # Add the recruited code if it isn't already in ranked6
        if recruited and all(code != recruited[0] for code, _ in ranked6):
            ranked6.append(recruited)

    ranked6.sort(key=lambda x: (-x[1], x[0]))
    return ranked6, special_adds



# =============================
# Public API
# =============================
def run_model(
    form_texts: Dict[str, str],
    sor_csv_path: str,
    fields_override: Optional[List[str]] = None,
    top_k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
    rules_registry: Optional[dict] = None,          # UNUSED
    add_client_inspection_rule: bool = True,        # kept for compat
) -> Tuple[List[str], List[str], List[str]]:
    logs: List[str] = []

    # -------- 1) Input prep --------
    fields = list((fields_override or form_texts.keys()))
    fields = [f for f in fields if f in form_texts]

    cleaned: Dict[str, str] = {}
    dropped: List[str] = []
    for key in fields:
        raw = form_texts.get(key, "")
        val = _canon(raw)
        if _is_noise(val):
            dropped.append(key); continue
        cleaned[key] = val

    if dropped:
        logs.append(f"[prep] Dropped {len(dropped)} noisy/empty fields: {sorted(dropped)}")
    else:
        logs.append("[prep] No fields dropped as noise.")

    if not cleaned:
        logs.append("[prep] All candidate fields empty/noisy; returning no suggestions at this stage.")
        return [], [], logs

    # -------- 2) Field weights --------
    weights = _select_field_weights(list(cleaned.keys()))
    logs.append("[weights] Snapshot: " + str({k: round(weights.get(k,1.0), 2) for k in cleaned.keys()}))

    # -------- 2a) Build matcher --------
    matcher = JobCodeMatcher(sor_csv_path, model_name=model_name)

    # -------- 2b) Sentence-aware recall + per-field cap --------
    per_field_pool: Dict[str, Dict[str, Dict]] = {}
    total_chunks = 0

    for field in cleaned.keys():
        text = cleaned[field]
        chunks = _chunk_text(field, text, max_chunks=MAX_CHUNKS_PER_FIELD)
        total_chunks += len(chunks)
        field_weight = weights.get(field, 1.0)

        field_map_raw: Dict[str, Dict] = {}
        for idx, chunk in enumerate(chunks):
            cw = _chunk_weight(field, idx, chunk)
            matches = matcher.match(chunk, top_k=TOPK_PER_CHUNK, lead_allowed=True) or []
            top_preview = [(c, round(s, 3)) for (c, s) in matches[:5]]
            logs.append(f"[recall] {field} chunk#{idx+1}/{len(chunks)}: '{chunk[:80]}' → top {len(top_preview)} {top_preview}")

            for code, score in matches:
                wscore = float(score) * field_weight * cw
                entry = field_map_raw.get(code)
                if entry is None or wscore > entry["score"]:
                    field_map_raw[code] = {
                        "score": wscore,
                        "field": field,
                        "first_chunk_ix": idx,
                        "sample_chunk": chunk[:160],
                        "hits": 1,
                    }
                else:
                    entry["hits"] += 1

        sorted_field = sorted(field_map_raw.items(), key=lambda kv: kv[1]["score"], reverse=True)
        capped_field = dict(sorted_field[:CAP_PER_FIELD_AFTER_RECALL])
        if len(sorted_field) > CAP_PER_FIELD_AFTER_RECALL:
            logs.append(f"[cap.field] {field}: kept {CAP_PER_FIELD_AFTER_RECALL} / {len(sorted_field)}")
        per_field_pool[field] = capped_field

    logs.append(f"[recall] Total chunks processed: {total_chunks}")

    # -------- 2c) Aggregate to global pool --------
    global_pool: Dict[str, Dict] = {}
    for field, fmap in per_field_pool.items():
        for code, info in fmap.items():
            if (code not in global_pool) or (info["score"] > global_pool[code]["score"]):
                global_pool[code] = dict(info)
            else:
                global_pool[code]["hits"] = global_pool[code].get("hits", 1) + info.get("hits", 0)

    logs.append(f"[recall] Global pool size: {len(global_pool)}")

    # Debug facet previews (top 5)
    try:
        tops = sorted(global_pool.items(), key=lambda kv: kv[1]["score"], reverse=True)[:5]
        previews = []
        for code, _ in tops:
            fac = _code_facets(code, matcher)
            previews.append(f"{code}:{(fac.get('facet_text') or '')[:100]}")
        logs.append("[debug] facet previews: " + " || ".join(previews))
    except Exception:
        pass

    # -------- 3) Text evidence & intents (soft) --------
    full_text_blob = " ".join(cleaned.get(k, "") for k in cleaned.keys()).strip()
    intents = _intent_pairs_from_text(full_text_blob)
    subtype_hint = _infer_tile_subtype(full_text_blob)
    tile_count = _extract_tile_count_from_text(full_text_blob)

    tokens = set(_tok(full_text_blob))
    soft_flags = {
        "leadwork": bool({"lead","flashing","apron","soaker","abutment","step","cover"} & tokens),
        "tile":     bool({"tile","tiles","slate","concrete","clay","plain"} & tokens),
        "guttering":bool({"gutter","guttering"} & tokens or _fuzzy_hit(full_text_blob, {"gutter","gutters","guttering"})),
        "downpipe": bool({"downpipe","rwp"} & tokens),
        "ridge":    bool({"ridge","hip"} & tokens),
        "chimney":  bool({"chimney"} & tokens),
        "fascia":   bool({"fascia","barge"} & tokens),
        "soffit":   bool({"soffit"} & tokens),
        "verge":    bool({"verge"} & tokens),
        "valley":   bool({"valley"} & tokens),
        "airbrick": bool({"airbrick","air","brick","vent"} & tokens or _fuzzy_hit(full_text_blob, {"airbrick","air brick","air-vent","air vent","airvent"})),
    }

    logs.append(f"[evidence] SOFT families: {sorted([k for k,v in soft_flags.items() if v])}")
    if intents:
        logs.append(f"[evidence] Intents (verb,material): {intents}")
    if subtype_hint:
        logs.append(f"[evidence] Tile subtype hint: {subtype_hint}")
    if tile_count is not None:
        logs.append(f"[evidence] Tile count hint: {tile_count}")

    # -------- 4) Lightweight scoring --------
    re_ranked4, score4_logs = _score_step4_lightweight(
        matcher=matcher,
        global_pool=global_pool,
        cleaned_texts=cleaned,
        intents=intents,
        subtype_hint=subtype_hint,
        tile_count=tile_count,
    )
    logs.extend(score4_logs)

    # -------- 6) Stronger blending --------
    re_ranked6, score6_logs = _score_step6_stronger(
        matcher=matcher,
        ranked4=re_ranked4,
        global_pool=global_pool,
        per_field_pool=per_field_pool,
        cleaned_texts=cleaned,
    )
    logs.extend(score6_logs)

    # -------- 6.5) Learned re-ranker (optional) --------
    re_ranked6 = _apply_learned_reranker(
        ranked6=re_ranked6,
        global_pool=global_pool,
        form_texts=form_texts,
        cleaned_texts=cleaned,
        logs=logs,
    )

    # -------- 7) INLINE RULES --------
    re_ranked7, cap_overrides, excludes, special_adds_rules, rule_logs = _apply_inline_rules(
        matcher=matcher,
        ranked6=re_ranked6,
        global_pool=global_pool,
        cleaned_texts=cleaned,
        intents=intents,
        tile_count=tile_count,
    )
    logs.extend(rule_logs)

    re_ranked7 = [(c, s) for (c, s) in re_ranked7 if c not in excludes]

    annotated: List[Tuple[str, float, str, str, Dict[str,str]]] = []
    for code, score7 in re_ranked7:
        fam, act, facets = _code_facets_to_family_action(code, matcher)
        annotated.append((code, score7, fam, act, facets))

    # Debug: family by column (top 5)
    if annotated:
        try:
            previews = []
            for code, _, fam, _, facets in annotated[:5]:
                fam2, src_field = _family_from_structured_facets(
                    facets.get("job_type",""),
                    facets.get("work_category",""),
                    facets.get("element",""),
                    facets.get("work_subcategory",""),
                )
                previews.append(f"{code}:{(fam2 or fam)}@{src_field or 'fallback'}")
            logs.append("[debug] family by column: " + " || ".join(previews))
        except Exception:
            pass

    # -------- 5D) Variable family caps (with RULE overrides) --------
    family_caps = _variable_family_caps_from_text(cleaned, tile_count)
    for k, v in cap_overrides.items():
        family_caps[k] = v
    logs.append(f"[cap.family] variable caps (+overrides): {family_caps}")

    # -------- 5E + 5G) Diversity & duplicate collapse --------
    selected: List[Tuple[str, float, str, str, Dict[str,str]]] = []
    seen_actions_per_family: Dict[str, Set[str]] = {}

    for code, score, fam, act, facets in annotated:
        cap = family_caps.get(fam, 0)
        if cap <= 0:
            continue
        fam_count = sum(1 for _,_,f2,_,_ in selected if f2 == fam)
        if fam_count >= cap:
            continue
        actions_seen = seen_actions_per_family.setdefault(fam, set())
        if act in actions_seen:
            continue
        if _collapse_near_duplicates([*selected, (code, score, fam, act, facets)], threshold=0.90)[-1][0] != code:
            continue
        selected.append((code, score, fam, act, facets))
        actions_seen.add(act)

    for code, score, fam, act, facets in annotated:
        cap = family_caps.get(fam, 0)
        if cap <= 0:
            continue
        fam_count = sum(1 for _,_,f2,_,_ in selected if f2 == fam)
        if fam_count >= cap:
            continue
        if any(c == code for c, *_ in selected):
            continue
        if _collapse_near_duplicates([*selected, (code, score, fam, act, facets)], threshold=0.90)[-1][0] != code:
            continue
        selected.append((code, score, fam, act, facets))

    # -------- 8) Recruitment rails (guarded) --------
    before_recruit = [c for c, *_ in selected]
    re_ranked7_sorted = sorted(re_ranked7, key=lambda x: (-x[1], x[0]))
    re_ranked8, special_adds_recruit = _recruit_missing_families(
        matcher=matcher,
        selected_codes=before_recruit,
        ranked6=re_ranked7_sorted,
        global_pool=global_pool,
        cleaned_texts=cleaned,
        intents=intents,
        family_caps=family_caps,
        logs=logs,
    )

    annotated2: List[Tuple[str, float, str, str, Dict[str,str]]] = []
    for code, score in re_ranked8:
        fam, act, facets = _code_facets_to_family_action(code, matcher)
        annotated2.append((code, score, fam, act, facets))

    selected2: List[Tuple[str, float, str, str, Dict[str,str]]] = []
    seen_actions_per_family2: Dict[str, Set[str]] = {}
    for code, score, fam, act, facets in annotated2:
        cap = family_caps.get(fam, 0)
        if cap <= 0:
            continue
        fam_count = sum(1 for _,_,f2,_,_ in selected2 if f2 == fam)
        if fam_count >= cap:
            continue
        actions_seen = seen_actions_per_family2.setdefault(fam, set())
        if act in actions_seen:
            continue
        if _collapse_near_duplicates([*selected2, (code, score, fam, act, facets)], threshold=0.90)[-1][0] != code:
            continue
        selected2.append((code, score, fam, act, facets))
        actions_seen.add(act)
    for code, score, fam, act, facets in annotated2:
        cap = family_caps.get(fam, 0)
        if cap <= 0:
            continue
        fam_count = sum(1 for _,_,f2,_,_ in selected2 if f2 == fam)
        if fam_count >= cap:
            continue
        if any(c == code for c, *_ in selected2):
            continue
        if _collapse_near_duplicates([*selected2, (code, score, fam, act, facets)], threshold=0.90)[-1][0] != code:
            continue
        selected2.append((code, score, fam, act, facets))

    MACRO_CAP = max(top_k, 10)
    selected2 = selected2[:MACRO_CAP]

    suggestions = [code for code, *_ in selected2]
    special_adds = list(dict.fromkeys(special_adds_rules + special_adds_recruit))

    logs.append(f"[output] Returning {len(suggestions)} suggestion(s) after Steps 7–8 (rules+recruit).")
    if special_adds:
        logs.append(f"[output] Special adds: {special_adds}")

    return special_adds, suggestions, logs
