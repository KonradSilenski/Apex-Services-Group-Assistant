 

import os

 

# Main SOR CSV file
BM_SOR_CSV_PATH = "sorCodeModel/dataSOR/processedSORCodes2.csv"

# Feedback CSV with operator corrections
BM_FEEDBACK_CSV_PATH = "/app/model/operator_feedback.csv"

# Memory cache files
BM_TEXT_CACHE_PATH = "/app/model/cache_text.json"
BM_PACK_CACHE_PATH = "/app/model/cache_pack.json"
BM_FORM_CACHE_PATH = "/app/model/cache_form.json"

# Directory containing all trained ML models (reranker, qty models, meta files)
BM_LEARN_MODEL_DIR = "/app/model/"

# Derived model paths
BM_RERANKER_MODEL_PATH = os.path.join(BM_LEARN_MODEL_DIR, "reranker.pkl")
BM_RERANKER_META_PATH  = os.path.join(BM_LEARN_MODEL_DIR, "reranker_meta.json")

BM_QTY_SRC_MODEL_PATH = os.path.join(BM_LEARN_MODEL_DIR, "qty_source.pkl")
BM_QTY_MAG_MODEL_PATH = os.path.join(BM_LEARN_MODEL_DIR, "qty_magnitude.pkl")
BM_QTY_META_PATH      = os.path.join(BM_LEARN_MODEL_DIR, "qty_meta.json")
 