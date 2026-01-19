# (full file)
import os
import csv
import json
from sentence_transformers import SentenceTransformer, util
import torch

class JobCodeMatcher:
    def __init__(self, job_code_file: str, model_name: str = 'all-MiniLM-L6-v2'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        ext = os.path.splitext(job_code_file)[1].lower()
        if ext == ".csv":
            self.job_code_data = self._load_from_csv(job_code_file)
        elif ext == ".json":
            with open(job_code_file, 'r') as f:
                self.job_code_data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")

        self.job_code_texts = [f"{code}: {info['description']}" for code, info in self.job_code_data.items()]
        self.job_code_codes = list(self.job_code_data.keys())
        self.job_code_embeddings = self.model.encode(self.job_code_texts, convert_to_tensor=True)

    def _load_from_csv(self, csv_path):
        job_code_dict = {}
        with open(csv_path, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                code = row.get("Code", "").strip()
                description = row.get("Medium Description", "").strip()
                job_type = row.get("Job Type", "").strip().lower()
                element = row.get("Element", "").strip().lower()
                category = row.get("Work Categories", "").strip().lower()
                if code and description:
                    job_code_dict[code] = {
                        "description": description,
                        "job_type": job_type,
                        "element": element,
                        "category": category
                    }
        return job_code_dict

    def match(self, report_text: str, top_k: int = None, boosts: dict = None, lead_allowed: bool = False, exclude_codes=None):
        report_embedding = self.model.encode(report_text, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(report_embedding, self.job_code_embeddings).squeeze()
        adjusted_scores = []
        boosts = boosts or {}
        exclude_codes = exclude_codes or set()

        for idx, code in enumerate(self.job_code_codes):
            if code in exclude_codes:
                continue
            info = self.job_code_data[code]
            if not lead_allowed:
                text_fields = [info.get("description", ""), info.get("job_type", ""), info.get("element", ""), info.get("category", "")]
                if any("lead" in field.lower() for field in text_fields):
                    continue
            base_score = cos_scores[idx].item()
            boost = boosts.get(code, 0.0)
            adjusted_scores.append((code, base_score + boost))

        sorted_results = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)
        return sorted_results if top_k is None else sorted_results[:top_k]
