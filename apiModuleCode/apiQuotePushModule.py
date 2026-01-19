 # apiModule_sors.py
"""
Exports all SORs from Propeller to a CSV.

- Authenticates exactly like apiModule.py (uses apiModuleCode.configKeys)
- Writes a CSV with all fields returned by the API
- Includes a simple test runner at the bottom

Usage (standalone):
    python apiModule_sors.py
"""

import os
import csv
import json
import requests
from datetime import datetime
import apiModuleCode.configKeys as configKeys


PROP_SORS_URL = configKeys.prop_sors_url


def _auth_headers() -> dict:
    """
    Authenticate using the same payload/flow as apiModule.py and return an
    Authorization header dict ready for use with requests.
    """
    auth_payload = {
        "username": configKeys.username,
        "password": configKeys.password,
        "grant_type": configKeys.grantType,
        "organisationcode": configKeys.orgCode,
    }
    resp = requests.post(configKeys.liveAPI, data=auth_payload, timeout=30)
    resp.raise_for_status()
    token = resp.json()["access_token"]
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def fetch_all_sors(headers: dict) -> list[dict]:
    """
    Fetch SORs from the /Admin/api/SORs endpoint.

    Notes:
    - The endpoint in Propeller typically returns the full list.
    - If your server uses paging (e.g., query params like ?page=N&size=M),
      you can adapt the loop below: it’s structured to allow easy paging.
    """
    url = PROP_SORS_URL
    all_rows: list[dict] = []

    # Basic (single-shot) request
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse SORs JSON: {e}")

    if isinstance(data, list):
        all_rows.extend(data)
    elif isinstance(data, dict) and "results" in data:
        # In case the API returns a wrapper object (rare)
        all_rows.extend(data.get("results", []))
    else:
        raise RuntimeError("Unexpected SORs response format.")

    # --- Example pagination hook (disabled by default) ---
    # next_link = data.get("next") if isinstance(data, dict) else None
    # while next_link:
    #     r = requests.get(next_link, headers=headers, timeout=60)
    #     r.raise_for_status()
    #     page = r.json()
    #     all_rows.extend(page.get("results", []) if isinstance(page, dict) else page)
    #     next_link = page.get("next") if isinstance(page, dict) else None

    return all_rows


def write_sors_to_csv(rows: list[dict], output_csv: str) -> str:
    """
    Write SOR list (list of dicts) to CSV.
    - Builds the header from the union of keys across all rows so we don’t lose fields.
    - Returns the output path.
    """
    if not rows:
        raise ValueError("No SOR rows to write.")

    # Build a stable header (union of keys, common fields first if present)
    key_union = set().union(*(row.keys() for row in rows))

    preferred_order = [
        "id",
        "code",
        "name",
        "reference",
        "customerreference",
        "serviceid",
        "categoryid",
        "category",
        "subcategoryid",
        "subcategory",
        "nvmminutes",
        "supplierid",
        "purchaseprice",
        "amount",
        "taxcode",
        "nominalcode",
        "accountreference",
        "alternativereference",
        "comment",
    ]
    # keep known ones in order, then append the unknown/rest
    header = [k for k in preferred_order if k in key_union] + [
        k for k in sorted(key_union) if k not in preferred_order
    ]

    _ensure_dir(os.path.dirname(output_csv) or ".")

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            # ensure all keys exist
            safe = {k: row.get(k, "") for k in header}
            writer.writerow(safe)

    return output_csv


def export_sors(output_dir: str = "sorCodeModel/dataSOR", filename: str | None = None) -> str:
    """
    High-level convenience function: authenticate, fetch, write CSV.
    Returns the path to the written CSV.
    """
    headers = _auth_headers()
    rows = fetch_all_sors(headers)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"propeller_sors_{timestamp}.csv"

    output_csv = os.path.join(output_dir, filename)
    return write_sors_to_csv(rows, output_csv)


# ---------------------- Test Runner ----------------------
if __name__ == "__main__":
    try:
        out_path = export_sors()
        print(f"✅ SOR export complete: {out_path}")
    except Exception as e:
        print(f"❌ Failed to export SORs: {e}")
