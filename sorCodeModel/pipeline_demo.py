# sorCodeModel/pipeline_demo.py
from __future__ import annotations
import pathsAndImports
from sorCodeModel.pathsAndImports import (
    BM_SOR_CSV_PATH,
    BM_FEEDBACK_CSV_PATH,
    BM_TEXT_CACHE_PATH,
    BM_PACK_CACHE_PATH,
    BM_FORM_CACHE_PATH,
)

import os
# --- Import the operations (kept out of this file to keep it tidy) ---
from sorCodeModel.learnLayer.training_ops import (
    update_feedback_for_visit,
    run_and_log_without_prompt,
    run_with_memory_enabled_once,
)

# --- Paths / constants 
SOR_CSV_PATH = "sorCodeModel/dataSOR/processedSORCodes2.csv"
FEEDBACK_CSV_PATH = BM_FEEDBACK_CSV_PATH




if __name__ == "__main__":
    # ==============================
    # Set your visit ID here:
    # ==============================
    #visit_id = 5648118  # AIRBRICKS JOB
    #visit_id = 5651181 # GUTTER CLEAN JOB
    #visit_id = 5693248 # ROOF COVER & FELT BATTENS
    #visit_id = 5695576 # Multi-Visit Gutter Job
#5695520 5718721 5759329 5881567
    visit_id = 5867133

    # ==============================
    # Choose ONE of the actions below by uncommenting it.
    # Leave others commented out. You can commit this file with
    # your preferred default commented selection.
    # ==============================

    # --- (1) UPDATE: write corrected codes into the last CSV row for this visit_id ---
    # Example format: "201303x1, 201307x4, 231011x6, 603903x2"
    '''corrected = "201303x42, 201307x42, 231011x62, 603903x22"
    ok = update_feedback_for_visit(visit_id, corrected, feedback_csv_path=FEEDBACK_CSV_PATH)
    print("Updated CSV" if ok else "No CSV row found for this visit_id")'''

    # --- (2) LOG ONLY: run model, save caches + CSV row with predicted_json, corrected_json left {} ---- WORKS
    #result = run_and_log_without_prompt(visit_id, sor_csv_path=SOR_CSV_PATH, feedback_csv_path=FEEDBACK_CSV_PATH)
    #print(f"[LOGGED] visit_id={result['visit_id']}  text_hash={result['text_hash']}  pack_hash={result['pack_hash']}")
    #print(f"Predicted codes: {list(result.get('quantities', {}).keys())}")
#123
    '''
  OUTPUT GUIDE OF FUNCTION (2)
  "visit_id": int,
  "codes": [...],                    # completed codes (back-compat)
  "quantities": {code: qty, ...},    # completed, flattened to {code: float}
  "completed_codes": [...],          # same as codes
  "completed_quantities": {...},     # same as quantities
  "future_codes": [...],             # from “future/issues” text
  "future_quantities": {code: {...}},# qty dicts per code for future set
  "text_hash": str,                  # stable hash of free-text
  "pack_hash": str                   # stable hash of evidence pack
    '''

    # --- (3) LOG WITH MEMORY ONCE: temporarily enable memory, run same as (2), then turn memory back off ---
    result = run_with_memory_enabled_once(visit_id, sor_csv_path=SOR_CSV_PATH, feedback_csv_path=BM_FEEDBACK_CSV_PATH)
 
    for code in result["future_codes"]:
                qty = result["future_quantities"].get(code, 0)
                print("Qty Dict:")
                print(qty)


    print(f"[LOGGED-MEM] visit_id={result['visit_id']}  text_hash={result['text_hash']}  pack_hash={result['pack_hash']}")
    print(f"Predicted codes (mem): {list(result.get('quantities', {}).keys())}")

    # Tip: keep only one of the above uncommented to avoid double-logging.
    pass
 





    # 5461011- 4754699 - Test Job 1 - [201303x1, 201307x4, 231011x6, 603903x2]
    # 5461025- 4894609 - Test Job 2 - SAME JOB SEE BELOW
    # 5461042- 4661913 - Test Job 3 - [201103 x 2, 201150 x 2.5, 201717 x 38.4, 201501 x 11] Maybe Expect Valley code instead of verge? (they made mistake on invoice)