# sorCodeModel/mainModelLayer/theCondensedFunction.py
from typing import Dict, Any, List, Tuple
import os

# Central SOR path
from sorCodeModel.pathsAndImports import BM_SOR_CSV_PATH

# --- Require the new model (v2). If it's missing, raise a clear error, but
#     do NOT swallow non-import exceptions (so real bugs surface properly). ---
try:
    # Preferred: package-style import
    from sorCodeModel.mainModelLayer.jobmatcher_model_v2 import run_model as _run_model  # v2 only
    _RUNNER_NAME = "jobmatcher_model_v2"
except ImportError as e:
    # Fallback: try local/relative import (useful if package path isn't set up cleanly)
    try:
        from .jobmatcher_model_v2 import run_model as _run_model  # type: ignore
        _RUNNER_NAME = "jobmatcher_model_v2_local"
    except Exception as e2:
        # Re-raise with both error messages to make debugging easier
        raise ImportError(
            "jobmatcher_model_v2 is required but could not be imported via either\n"
            "  1) 'sorCodeModel.mainModelLayer.jobmatcher_model_v2', or\n"
            "  2) relative '.jobmatcher_model_v2'.\n"
            "Please ensure the file exists and PYTHONPATH / package layout are correct.\n"
            f"Primary ImportError: {e}\n"
            f"Fallback error: {e2}"
        ) from e2

from sorCodeModel.mainModelLayer.quantity_resolver import resolve_quantities
from sorCodeModel.mainModelLayer.formProcessing import DetailsResponse, flatten_survey_data

# Live SOR codes CSV (centralised)
DEFAULT_SOR_CSV = BM_SOR_CSV_PATH


def _union_preserve_order(primary: List[str], to_add: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in list(primary) + list(to_add):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _resolve_sor_path(sor_csv_path: str) -> str:
    path_in = (sor_csv_path or "").strip()
    if (not path_in) or (not os.path.exists(path_in)):
        candidate = DEFAULT_SOR_CSV
        print(f"[condensed] SOR path '{path_in}' missing → using '{candidate}'")
        return candidate
    return path_in


COMPLETED_TEXT_FIELDS: List[str] = [
    "1.2_Work_Description",
    "6.3_Leadwork_Comment",
    "7.3_Chimney_Comment",
    "7.5_Chimney_Flaunch_Comment",
    "11.1_Other_Works_Completed",
]

FUTURE_TEXT_FIELDS: List[str] = [
    "11.2_Other_Works_Needed",
    "13.1_Issues_Present",
    "13.2_Issues_Comments",
]


def _build_texts(instance: DetailsResponse) -> Dict[str, str]:
    return flatten_survey_data(instance)


def _run_selection_pass(
    texts: Dict[str, str],
    fields: List[str],
    sor_csv_path: str,
    top_k: int = 10,
) -> Tuple[List[str], List[str], List[str]]:
    special_adds, suggestions, logs = _run_model(
        form_texts=texts,
        sor_csv_path=sor_csv_path,
        fields_override=fields,
        top_k=top_k,
    )
    return special_adds, suggestions, logs


def condensed_function(
    instance: DetailsResponse,
    sor_csv_path: str = DEFAULT_SOR_CSV,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "codes": List[str],
        "quantities": Dict[str, Any],
        "future_codes": List[str],
        "logs": List[str],
      }
    """
    resolved_sor = _resolve_sor_path(sor_csv_path)
    print(f"[condensed] using runner={_RUNNER_NAME}  SOR='{resolved_sor}'")

    texts: Dict[str, str] = _build_texts(instance)

    # Completed / main pass
    special_adds_main, suggestions_main, logs_main = _run_selection_pass(
        texts=texts,
        fields=COMPLETED_TEXT_FIELDS,
        sor_csv_path=resolved_sor,
        top_k=top_k,
    )

    # Future/issues pass
    special_adds_future, suggestions_future, logs_future = _run_selection_pass(
        texts=texts,
        fields=FUTURE_TEXT_FIELDS,
        sor_csv_path=resolved_sor,
        top_k=top_k,
    )

    # Merge codes; preserve order; include any special adds at the front.
    codes_main: List[str] = _union_preserve_order(special_adds_main, suggestions_main)
    future_codes: List[str] = _union_preserve_order(special_adds_future, suggestions_future)

    # --- Visit-type policy (inspection handling is visit_type-driven) ---
    visit_type = (texts.get("1.1_Visit_Type") or "").strip().lower()
    inspection_code = "221001"

    if visit_type == "inspection only":
        # Ensure inspection present in completed
        if inspection_code not in codes_main:
            codes_main = [inspection_code] + codes_main
        # Move ALL non-inspection codes to future
        non_inspection = [c for c in codes_main if c != inspection_code]
        future_codes = _union_preserve_order(future_codes, non_inspection)
        codes_main = [inspection_code]  # completed contains inspection only

    elif visit_type == "inspection and repair":
        # Ensure inspection is included among completed, keep others as-is
        if inspection_code not in codes_main:
            codes_main = [inspection_code] + codes_main

    else:  # "repair only" (or anything else)
        # Make sure inspection is not in completed
        codes_main = [c for c in codes_main if c != inspection_code]

    # Quantities: legacy positional signature
    quantities = resolve_quantities(
        instance,
        codes_main,  # picked_codes
        texts,       # form_data (flattened form)
        texts,       # free_text_map (reuse same mapping)
        resolved_sor,
    )

    # Build combined logs and include which runner we used for traceability,
    # plus predicted lists for downstream visibility.
    logs: List[str] = []
    logs.append(f"[condensed] using runner={_RUNNER_NAME} SOR='{resolved_sor}'")
    if logs_main:
        logs.extend(logs_main)
    if logs_future:
        logs.extend(["[Future pass] " + x for x in logs_future])
    logs.append(f"[pred.completed] {codes_main}")
    logs.append(f"[pred.future] {future_codes}")

    return {
        "codes": codes_main,
        "quantities": quantities,
        "future_codes": future_codes,
        "logs": logs,
    }


# ---- Legacy shims (compat) ----
# NOTE: These accept **_ to swallow extra keyword arguments such as `images`
# and `extra_details` passed from older parts of the pipeline (training_ops,
# pipeline_demo, etc.). The extra data is not currently used by the condensed
# function, but keeping the signature flexible preserves compatibility.


def get_sor_codes_from_instance(
    instance: DetailsResponse,
    sor_csv_path: str = DEFAULT_SOR_CSV,
    top_k: int = 10,
    **_: Any,
) -> List[str]:
    result = condensed_function(instance=instance, sor_csv_path=sor_csv_path, top_k=top_k)
    return result.get("codes", [])


def get_future_sor_codes_from_instance(
    instance: DetailsResponse,
    sor_csv_path: str = DEFAULT_SOR_CSV,
    top_k: int = 10,
    **_: Any,
) -> List[str]:
    result = condensed_function(instance=instance, sor_csv_path=sor_csv_path, top_k=top_k)
    return result.get("future_codes", [])


def get_sor_codes_and_quantities_from_instance(
    instance: DetailsResponse,
    sor_csv_path: str = DEFAULT_SOR_CSV,
    top_k: int = 10,
    **_: Any,
) -> Dict[str, Any]:
    result = condensed_function(instance=instance, sor_csv_path=sor_csv_path, top_k=top_k)
    return {
        "codes": result.get("codes", []),
        "quantities": result.get("quantities", {}),
        "future_codes": result.get("future_codes", []),
        "logs": result.get("logs", []),
    }
