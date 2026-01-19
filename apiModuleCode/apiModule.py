import requests
import apiModuleCode.configKeys as configKeys
import base64
import fitz
import unicodedata
import os
from io import BytesIO
import re
from uuid import uuid4
from backend.details.model import DetailsResponse
from PIL import Image
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()


def normalize(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text or "")
        .replace("­", "-")
        .replace("–", "-")
        .replace("\xa0", " ")
        .strip()
        .lower()
    )


# --- final string cleaner so no \xa0 or odd hyphens leak through ---
def clean_value(v: Optional[str]) -> Optional[str]:
    """
    Normalise any extracted string so that:
    - \xa0 → regular spaces
    - odd hyphens normalised
    - leading/trailing whitespace stripped
    """
    if v is None:
        return v
    return (
        unicodedata.normalize("NFKD", v)
        .replace("­", "-")
        .replace("–", "-")
        .replace("\xa0", " ")
        .strip()
    )


# ============================================================
# SECTION HEADERS — now includes "Engineer's Signature"
# ============================================================

SECTION_HEADERS = {
    *map(
        lambda s: s.lower(),
        [
            "General",
            "Asbestos Check",
            "Scaffolding",
            "Roof",
            "Ridge Tile",
            "Leadwork",
            "Chimney",
            "Roofline",
            "Rainwater Goods",
            "Other Works",
            "Access",
            "Other Issues",
            "Customer Vulnerability",
            "Engineer's Signature",
        ],
    )
}

# Known footer/footer-like noise from the Propeller PDF export that sometimes
# gets merged onto the end of real field values (e.g. "No Propeller Powered - ...").
# We strip anything from the first occurrence of one of these tokens onwards.
FOOTER_NOISE_TOKENS = [
    "propeller powered -",
    "field management and compliance solution",
    "www.propellerpowered.co.uk",
    "visit ref:",
    "page 1 / 2",
    "page 2 / 2",
]


def strip_footer_noise(value: str) -> str:
    """
    Remove Propeller PDF footer noise if it has been concatenated onto a
    legitimate field value. For example:
        "No Propeller Powered - Field Management..." -> "No"
    The comparison is done case-insensitively and everything from the first
    footer token onwards is dropped.
    """
    if not value:
        return value
    text = str(value)
    lower = text.lower()
    for token in FOOTER_NOISE_TOKENS:
        idx = lower.find(token)
        if idx != -1:
            text = text[:idx]
            lower = text.lower()
    # Trim common trailing separators and whitespace
    return text.strip(" |\t\r\n")


def extract_field_next_line(text_lines, label: str, fuzzy: bool = False) -> str:
    norm_label = normalize(label)
    for i, line in enumerate(text_lines):
        if normalize(line) == norm_label or (fuzzy and norm_label in normalize(line)):
            if i + 1 < len(text_lines):
                return (text_lines[i + 1] or "").strip()
    return ""


# ============================================================
# IMPROVED extract_field_value
#   - duplicate skip
#   - header/label stop
#   - footer stripping
# ============================================================


def extract_field_value(text_lines, label: str, fuzzy: bool = False) -> str:
    """
    Extract a field value given a label string.

    Behaviour:
    - Matches either an exact normalized label or, if fuzzy=True, a prefix match.
    - Skips the first occurrence if the next non-blank line is the same label
      (handles header + label duplication).
    - Avoids capturing section headers or other labels as values.
    - Stops at the next known label or section header.
    - Strips known Propeller footer noise if it has bled into the line.
    """

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())

    lab_norm = _norm(label)
    n = len(text_lines)

    # Normalized section headers (e.g., "roof", "leadwork", etc.)
    section_headers = {_norm(s) for s in SECTION_HEADERS}

    # Normalized known labels (populated from label_to_raw in the caller)
    try:
        known_labels = {_norm(k) for k in KNOWN_LABELS}
    except NameError:
        known_labels = set()

    for i, orig_line in enumerate(text_lines):
        line = orig_line or ""
        line_norm = _norm(line)

        # Match label
        is_exact = line_norm == lab_norm
        is_prefix = fuzzy and line.lower().startswith(label.lower())
        if not (is_exact or is_prefix):
            continue

        # OPTION A: SKIP first occurrence if the next non-blank line is the
        # same label (header + label duplicate pattern).
        j = i + 1
        while j < n and not (text_lines[j] or "").strip():
            j += 1
        if j < n and _norm(text_lines[j]) == lab_norm:
            # This looks like a header row; skip and let the next occurrence
            # (the real field label) be processed instead.
            continue

        parts = []

        # 1) Same-line capture: "Ridge Tile    No"
        m = re.search(
            rf"{re.escape(label)}\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE
        )
        if m:
            val = strip_footer_noise((m.group(1) or "").strip())
            if val and _norm(val) not in section_headers:
                parts.append(val)

        # 2) Continuation lines (up to a small window)
        for k in range(i + 1, min(i + 7, n)):
            cand = strip_footer_noise((text_lines[k] or "").strip())
            if not cand:
                break
            c_norm = _norm(cand)

            # Stop if we hit a header
            if c_norm in section_headers:
                break

            # Stop if we hit another known label
            if c_norm in known_labels:
                break

            parts.append(cand)

        candidate = " ".join(parts).strip()
        if candidate:
            return candidate

        # If we got here, label was found but no usable value; keep searching
        # in case another (better) occurrence exists later.
    return ""


def first_nonblank(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def first_nonblank_keys(raw: Dict[str, Any], *keys: str) -> Optional[str]:
    return first_nonblank(*(raw.get(k) for k in keys))


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================


def extract_instance_from_visit(
    visit_id: str, save_dir: str = "", local_id: Optional[str] = None
):

    if local_id is None:
        local_id = str(uuid4())

    # Authenticate
    auth_payload = {
        "username": os.getenv("USERNAME"),
        "password": os.getenv("PASSWORD"),
        "grant_type": os.getenv("GRANTTYPE"),
        "organisationcode": os.getenv("ORGCODE"),
    }
    token = requests.post(configKeys.liveAPI, data=auth_payload).json()[
        "access_token"
    ]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # Fetch visit
    visit_data = requests.get(
        configKeys.visit_url.format(visit_id), headers=headers
    ).json()
    visit_details = visit_data.get("visitDetails", {})
    documents = visit_details.get("documents", []) or []

    # Find responsive PDF
    apex_doc = next(
        (
            d
            for d in documents
            if "responsive repair"
            in (d.get("name", "") + d.get("description", "")).lower()
        ),
        None,
    )

    # Prepare output dicts
    allowed_fields = {f for f in DetailsResponse.model_fields if f != "job_id"}
    values = {field: None for field in allowed_fields}
    raw: Dict[str, Any] = {}

    if apex_doc:
        # Read PDF
        pdf_bytes = base64.b64decode(apex_doc["content"])
        with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc:
            full_text = "\n".join(page.get_text() for page in doc)

            if __name__ == "__main__":
                print("\n===== RAW PDF TEXT (debug) =====\n")
                print(full_text)
                print("\n===== END RAW PDF TEXT =====\n")

        lines = (full_text or "").splitlines()

        raw["job_ref"] = str(visit_id)

        # --------------------------------------------------------------
        # Client / Address block extraction
        # --------------------------------------------------------------
        client_line = None
        address_line = None

        def looks_like_phone(s: str) -> bool:
            digits = re.sub(r"[^\d]", "", s or "")
            return 10 <= len(digits) <= 13

        for i, l in enumerate(lines[:200]):
            if normalize(l) == "work carried out":
                blk = [
                    (x or "").strip()
                    for x in lines[i + 1 : i + 10]
                    if (x or "").strip()
                ]
                # First non-phone = client
                for b in blk:
                    if not looks_like_phone(b):
                        client_line = b
                        break
                if client_line:
                    after = blk[blk.index(client_line) + 1 :]
                    for b in after:
                        if not looks_like_phone(b):
                            address_line = b
                            break
                break

        if client_line:
            raw["client"] = client_line.replace("\xa0", " ")
        if address_line:
            raw["address"] = address_line.replace("\xa0", " ")

        # --------------------------------------------------------------
        # LABEL → RAW FIELD MAP (EXTENDED)
        # --------------------------------------------------------------
        label_to_raw = {
            # General
            "Job Ref:": "job_ref",
            "Client": "client",
            "Address": "address",
            "Type of Visit": "visit_type",
            "Description of Works": "work_desc",
            "Type of Property": "property_type",
            # Scaffolding
            "Is Scaffolding Required": "scaffold_required",
            "Scaffolding Required": "scaffold",
            "Type of Scaffolding Required": "scaffold_type",
            # Elevations (front / rear / gable)
            "Front Elevation Measurement (m²/LM)": "front_elevation_measurement",
            "Rear Elevation Measurement (m²/LM)": "rear_elevation_measurement",
            "Gable Elevation Measurement (m²/LM)": "gable_elevation_measurement",
            # Roof - pitched / flat / other
            "Type Of Roof": "roof_type",
            "Pitched - Type Of Coverings": "coverings_type_pitched",
            "Pitched - Tile Size": "tile_size_pitched",
            "Pitched - Measurement (m²/LM)": "roof_measurement_pitched",
            "Flat - Type Of Coverings": "coverings_type_flat",
            "Flat - Tile Size": "tile_size_flat",
            "Flat - Measurement (m²/LM)": "flat_measurement",
            "Other - Type Of Coverings": "coverings_type_other",
            "Other - Tile Size": "tile_size_other",
            "Other - Measurement (m²/LM)": "other_measurement",
            # Ridge tile
            "Ridge Tile": "ridge_tile",
            "Ridge Tile Type": "ridge_tile_type",
            "Ridge Job": "ridge_job",
            "Ridge Measurement (m²/LM)": "ridge_measurement",
            # Leadwork
            "Leadwork": "leadwork",
            "Lead Flashings Measurement (m²/LM)": "lead_flashings_measurement",
            "Lead Flashings Comments": "leadwork_comment",
            "Leadwork Renew Measurement (m²/LM)": "leadwork_renew_measurement",
            "Leadwork Renew Comments": "leadwork_renew_comments",
            "Leadwork Repoint Measurement (m²/LM)": "leadwork_repoint_measurement",
            "Leadwork Repoint Comments": "leadwork_repoint_comments",
            # Chimney
            "Chimney": "chimney",
            "Chimney Pointing Measurement (m²/LM)": "chimney_point_measurement",
            "Chimney Pointing Comments": "chimney_point_comments",
            "Chimney Repoint Measurement (m²/LM)": "chimney_repoint_measurement",
            "Chimney Repoint Comments": "chimney_repoint_comments",
            "Chimney Renew Measurement (m²/LM)": "chimney_renew_measurement",
            "Chimney Renew Comments": "chimney_renew_comments",
            "Chimney Flaunch Measurement (m²/LM)": "chimney_flaunch_measurement",
            "Chimney Flaunch Comments": "chimney_flaunch_comment",
            # Roofline
            "Fascia": "fascia",
            "PVC Fascia Measurement (m²/LM)": "pvc_fascia_measurement",
            "Timber Fascia Measurement (m²/LM)": "timber_fascia_measurement",
            "Soffit": "soffit",
            "PVC Soffit Measurement (m²/LM)": "pvc_soffit_measurement",
            "Timber Soffit Measurement (m²/LM)": "timber_soffit_measurement",
            # Rainwater Goods - guttering
            "Guttering": "guttering",
            "PVC Guttering Replacement (m²/LM)": "pvc_guttering_replacement",
            "Cast Iron Guttering Replacement (m²/LM)": "cast_iron_guttering_replacement",
            "PVC Guttering Refix (m²/LM)": "pvc_guttering_refit",
            "Cast Iron Guttering Refix (m²/LM)": "cast_iron_guttering_refit",
            "Guttering Replace Measurement (m²/LM)": "guttering_replace_measurement",
            "Guttering Refix Measurement (m²/LM)": "guttering_refix_measurement",
            "Guttering Clean": "guttering_clean",
            "Number of Elevations": "guttering_num_elevations",
            # Rainwater Goods - RWP
            "RWP": "rwp",
            "PVC RWP Replacement (m²/LM)": "pvc_rwp_replacement",
            "Cast Iron RWP Replacement (m²/LM)": "cast_iron_rwp_replacement",
            "RWP Replace Measurement (m²/LM)": "rwp_replace_measurement",
            "RWP Refix": "rwp_refix",
            "RWP Refix Measurement (m²/LM)": "rwp_refix_measurement",
            # Other works
            "Other Works Completed": "other_completed",
            "Other Works Needed": "other_needed",
            # Access
            "Alley Gate Key": "access_key",
            "Party Wall Notice Required": "wall_notice",
            # Other issues & vulnerability
            "Other Issues": "issues_present",
            "Other Issues Comments": "issues_comments",
            "Customer Vulnerability": "customer_vuln",
            "Customer Comments": "customer_comments",
        }

        # Make known labels available to extractor
        global KNOWN_LABELS
        KNOWN_LABELS = {normalize(k) for k in label_to_raw}

        # --------------------------------------------------------------
        # RUN EXTRACTION
        # --------------------------------------------------------------
        for label, field in label_to_raw.items():
            v = extract_field_value(lines, label, fuzzy=True)
            if v:
                raw[field] = v.strip()

        # --------------------------------------------------------------
        # Scrub accidental header captures
        # --------------------------------------------------------------
        def scrub(v):
            return None if (v and normalize(v) in SECTION_HEADERS) else v

        for k in ("ridge_tile", "issues_present", "customer_vuln"):
            raw[k] = scrub(raw.get(k))

        # --------------------------------------------------------------
        # REDUCE TO MODEL FIELDS (EXTENDED)
        # --------------------------------------------------------------
        reduced = {
            # General
            "visit_type": raw.get("visit_type"),
            "work_desc": raw.get("work_desc"),
            "property_type": raw.get("property_type"),
            # Scaffolding
            "scaffold_required": first_nonblank_keys(
                raw, "scaffold_required", "scaffold"
            ),
            "scaffold_type": raw.get("scaffold_type"),
            # Elevations (aggregate to a single measurement if present)
            "elevation_measurement": first_nonblank_keys(
                raw,
                "front_elevation_measurement",
                "rear_elevation_measurement",
                "gable_elevation_measurement",
            ),
            # Roof
            "roof_type": first_nonblank_keys(raw, "roof_type"),
            "coverings_type": first_nonblank_keys(
                raw,
                "coverings_type_pitched",
                "coverings_type_flat",
                "coverings_type_other",
            ),
            "tile_size": first_nonblank_keys(
                raw,
                "tile_size_pitched",
                "tile_size_flat",
                "tile_size_other",
            ),
            "roof_measurement": first_nonblank_keys(
                raw,
                "roof_measurement_pitched",
                "flat_measurement",
                "other_measurement",
            ),
            # Ridge Tile
            "ridge_tile": raw.get("ridge_tile"),
            "ridge_tile_type": raw.get("ridge_tile_type"),
            "ridge_job": raw.get("ridge_job"),
            "ridge_measurement": raw.get("ridge_measurement"),
            # Leadwork
            "leadwork": raw.get("leadwork"),
            "leadwork_measurement": first_nonblank_keys(
                raw,
                "lead_flashings_measurement",
                "leadwork_renew_measurement",
                "leadwork_repoint_measurement",
            ),
            "leadwork_comment": first_nonblank_keys(
                raw,
                "leadwork_comment",
                "leadwork_renew_comments",
                "leadwork_repoint_comments",
            ),
            # Chimney
            "chimney": raw.get("chimney"),
            "chimney_measurement": first_nonblank_keys(
                raw,
                "chimney_point_measurement",
                "chimney_repoint_measurement",
                "chimney_renew_measurement",
                "chimney_flaunch_measurement",
            ),
            "chimney_comment": first_nonblank_keys(
                raw,
                "chimney_point_comments",
                "chimney_repoint_comments",
                "chimney_renew_comments",
                "chimney_flaunch_comment",
            ),
            "chimney_flaunch_measurement": raw.get("chimney_flaunch_measurement"),
            "chimney_flaunch_comment": raw.get("chimney_flaunch_comment"),
            # Roofline
            "fascia": raw.get("fascia"),
            "fascia_measurement": first_nonblank_keys(
                raw, "pvc_fascia_measurement", "timber_fascia_measurement"
            ),
            "soffit": raw.get("soffit"),
            "soffit_measurement": first_nonblank_keys(
                raw, "pvc_soffit_measurement", "timber_soffit_measurement"
            ),
            # Rainwater Goods - guttering
            "guttering": raw.get("guttering"),
            "guttering_replace": first_nonblank_keys(
                raw,
                "guttering_replace",
                "pvc_guttering_replacement",
                "cast_iron_guttering_replacement",
            ),
            "guttering_replace_measurement": first_nonblank_keys(
                raw, "guttering_replace_measurement"
            ),
            "guttering_refix": first_nonblank_keys(
                raw,
                "guttering_refix",
                "pvc_guttering_refit",
                "cast_iron_guttering_refit",
            ),
            "guttering_refix_measurement": first_nonblank_keys(
                raw, "guttering_refix_measurement"
            ),
            "guttering_clean": raw.get("guttering_clean"),
            "guttering_num_elevations": raw.get("guttering_num_elevations"),
            # Rainwater Goods - RWP
            "rwp": raw.get("rwp"),
            "rwp_replace": first_nonblank_keys(
                raw,
                "rwp_replace",
                "pvc_rwp_replacement",
                "cast_iron_rwp_replacement",
            ),
            "rwp_replace_measurement": first_nonblank_keys(
                raw, "rwp_replace_measurement"
            ),
            "rwp_refix": raw.get("rwp_refix"),
            "rwp_refix_measurement": raw.get("rwp_refix_measurement"),
            # Other works
            "other_works_completed": first_nonblank_keys(
                raw, "other_completed", "other_works_completed"
            ),
            "other_works_needed": first_nonblank_keys(
                raw, "other_needed", "other_works_needed"
            ),
            # Access
            "access_key": raw.get("access_key"),
            "wall_notice": raw.get("wall_notice"),
            # Other issues & vulnerability
            "issues_present": raw.get("issues_present"),
            "issues_comments": raw.get("issues_comments"),
            "customer_vuln": raw.get("customer_vuln"),
            "customer_comments": raw.get("customer_comments"),
        }

        for f in allowed_fields:
            if reduced.get(f) is not None:
                values[f] = reduced[f]

    else:
        print("⚠️ No Apex Responsive document found.")

    # --------------------------------------------------------------
    # IMAGE EXTRACTION (unchanged)
    # --------------------------------------------------------------
    if not save_dir:
        save_dir = os.getcwd()
    save_path = os.path.join(save_dir, str(local_id))
    os.makedirs(save_path, exist_ok=True)

    image_list = []
    before_counter = after_counter = during_counter = 1

    for doc in documents:
        name = doc.get("name", "")
        desc = doc.get("description", "")
        content = doc.get("content", "")

        if "." in name:
            ext = name.split(".")[-1].lower()
        else:
            ext = "jpg"

        is_img = ext in {"jpg", "jpeg", "png"} or content.startswith("/9j/")
        if not is_img:
            continue

        d = desc.lower()
        if "before" in d:
            label = "before"
        elif "after" in d:
            label = "after"
        elif "during" in d:
            label = "during"
        else:
            continue

        counter = {
            "before": before_counter,
            "after": after_counter,
            "during": during_counter,
        }[label]
        filename = f"{visit_id}-{label}_photo-{counter}.jpg"
        path = os.path.join(save_path, filename)

        try:
            img_data = base64.b64decode(content)
            img = Image.open(BytesIO(img_data))
            img.thumbnail((1000, 1000))
            img.convert("RGB").save(path, "JPEG")

            image_list.append({"filename": filename, "path": path})

            if label == "before":
                before_counter += 1
            elif label == "after":
                after_counter += 1
            else:
                during_counter += 1

        except Exception as e:
            print(f"⚠️ Error saving image {name}: {e}")

    # --------------------------------------------------------------
    # FINAL STRING CLEANUP (kill \xa0 etc.) BEFORE filling blanks
    # --------------------------------------------------------------
    for k, v in raw.items():
        if isinstance(v, str):
            raw[k] = clean_value(v)

    for k, v in values.items():
        if isinstance(v, str):
            values[k] = clean_value(v)

    # Convert None → " "
    values = {k: (v if v is not None else " ") for k, v in values.items()}

    result = {
        "instance": DetailsResponse(job_id=uuid4(), **values),
        "extra_details": {
            "job_ref": raw.get("job_ref") or " ",
            "client": raw.get("client") or " ",
            "address": raw.get("address") or " ",
        },
        "images": image_list,
    }

    return result["extra_details"], result["instance"], result["images"]


# ============================================================
# MAIN RUNNER (prints raw PDF only here)
# ============================================================

if __name__ == "__main__":
    load_dotenv()
    visit_id = 5648118  # change this freely

    extra_details, instance, images = extract_instance_from_visit(
        visit_id, save_dir="apiModuleCode/", local_id="debug-run"
    )

    print("\nEXTRA DETAILS:", extra_details)
    print("\nINSTANCE:\n", instance)
    print("\nImages:", images)

