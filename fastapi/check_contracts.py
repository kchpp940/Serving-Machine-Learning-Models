"""
Contract regression scanner for FastAPI module.

Run:  python3 check_contracts.py

Ensures:
  - Success responses preserve the original contract (no envelope, no extra fields)
  - /predict_batch output is exactly {"predictions": [...]} — no count/status/drift
  - Error responses include top-level "detail" for Streamlit/Kivy compatibility
  - Error responses use unified envelope {"status":"error", "error":{...}}

If any assertion fails, exit code is non-zero.
"""

import sys
import os
import json
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import urllib.request

PORT = 18210
SAMPLE = {
    "enginesize": 130, "curbweight": 2548, "horsepower": 111,
    "highwaympg": 27, "carwidth": 64.1, "wheelbase": 88.6,
    "drivewheel": "rwd", "citympg": 21, "boreratio": 3.47,
    "cylindernumber": "four",
}

SUCCESS_CONTRACTS = {
    "POST /predict": {
        "required": {"prediction", "status"},
        "forbidden": {"data", "error", "predictions", "count"},
    },
    "POST /predict_batch": {
        "required": {"predictions"},
        "forbidden": {"count", "status", "data", "error", "prediction"},
    },
    "POST /explain": {
        "required": {"prediction", "feature_importance", "top_features", "status"},
        "forbidden": {"data", "error", "predictions", "count"},
    },
    "GET /schema": {
        "required": {"feature_order", "numeric_features", "categorical_features", "categorical_options"},
        "forbidden": {"data", "error", "status", "predictions", "count"},
    },
    "GET /status": {
        "required": {"status", "model_loaded"},
        "forbidden": {"data", "error", "predictions", "count"},
    },
}


def start_server():
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=PORT, log_level="warning")


def fetch(method, path, data=None):
    url = f"http://127.0.0.1:{PORT}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json"} if body else {},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def check_success_contract(label, body, contract):
    keys = set(body.keys()) if isinstance(body, dict) else set()
    missing = contract["required"] - keys
    extra = contract["forbidden"] & keys
    if missing:
        print(f"   FAIL: missing required fields: {missing}")
        return False
    if extra:
        print(f"   FAIL: forbidden fields found (envelope drift): {extra}")
        return False
    print(f"   OK: required={contract['required']}, no drift fields")
    return True


def check_error_detail(label, body):
    if not isinstance(body, dict):
        print(f"   FAIL: error response is not a dict")
        return False
    if "detail" not in body:
        print(f"   FAIL: missing top-level 'detail' (Streamlit/Kivy compat)")
        return False
    if body.get("status") != "error":
        print(f"   FAIL: missing status=error in error envelope")
        return False
    if "error" not in body or "code" not in body.get("error", {}):
        print(f"   FAIL: missing error.code in error envelope")
        return False
    print(f"   OK: detail='{body['detail'][:50]}...', status=error, error.code={body['error']['code']}")
    return True


def main():
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(4)

    passed = 0
    failed = 0

    # ---- Success contracts ----
    print("=" * 60)
    print("SUCCESS RESPONSE CONTRACT CHECKS")
    print("=" * 60)

    # POST /predict
    code, body = fetch("POST", "/predict", SAMPLE)
    label = "POST /predict"
    contract = SUCCESS_CONTRACTS[label]
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 200 and check_success_contract(label, body, contract):
        passed += 1
    else:
        failed += 1

    # POST /predict_batch — rows format
    code, body = fetch("POST", "/predict_batch", {"rows": [SAMPLE, SAMPLE]})
    label = "POST /predict_batch (rows)"
    contract = SUCCESS_CONTRACTS["POST /predict_batch"]
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 200 and check_success_contract(label, body, contract):
        passed += 1
    else:
        failed += 1

    # POST /predict_batch — items format
    code, body = fetch("POST", "/predict_batch", {"items": [SAMPLE]})
    label = "POST /predict_batch (items)"
    contract = SUCCESS_CONTRACTS["POST /predict_batch"]
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 200 and check_success_contract(label, body, contract):
        passed += 1
    else:
        failed += 1

    # POST /predict_batch — direct list
    code, body = fetch("POST", "/predict_batch", [SAMPLE])
    label = "POST /predict_batch (list)"
    contract = SUCCESS_CONTRACTS["POST /predict_batch"]
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 200 and check_success_contract(label, body, contract):
        passed += 1
    else:
        failed += 1

    # POST /explain
    code, body = fetch("POST", "/explain", SAMPLE)
    label = "POST /explain"
    contract = SUCCESS_CONTRACTS[label]
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 200 and check_success_contract(label, body, contract):
        passed += 1
    else:
        failed += 1

    # GET /schema
    code, body = fetch("GET", "/schema")
    label = "GET /schema"
    contract = SUCCESS_CONTRACTS[label]
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 200 and check_success_contract(label, body, contract):
        passed += 1
    else:
        failed += 1

    # GET /status
    code, body = fetch("GET", "/status")
    label = "GET /status"
    contract = SUCCESS_CONTRACTS[label]
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 200 and check_success_contract(label, body, contract):
        passed += 1
    else:
        failed += 1

    # ---- Error contracts ----
    print("\n" + "=" * 60)
    print("ERROR RESPONSE CONTRACT CHECKS (detail compat for Streamlit/Kivy)")
    print("=" * 60)

    # ValueError → 400
    bad = {**SAMPLE, "drivewheel": "xxx"}
    code, body = fetch("POST", "/predict", bad)
    label = "POST /predict (ValueError)"
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 400 and check_error_detail(label, body):
        passed += 1
    else:
        failed += 1

    # Validation error → 422
    code, body = fetch("POST", "/predict", {"enginesize": "not_a_number"})
    label = "POST /predict (validation error)"
    print(f"\n{label}: status={code}, keys={sorted(body.keys()) if isinstance(body, dict) else 'N/A'}")
    if code == 422 and check_error_detail(label, body):
        passed += 1
    else:
        failed += 1

    # ---- Batch predict: anti-regression for count/status ----
    print("\n" + "=" * 60)
    print("ANTI-REGRESSION: /predict_batch must NOT have count/status")
    print("=" * 60)

    code, body = fetch("POST", "/predict_batch", {"rows": [SAMPLE]})
    if isinstance(body, dict):
        has_count = "count" in body
        has_status = "status" in body
        if has_count or has_status:
            print(f"   FAIL: /predict_batch has drifted fields: count={has_count}, status={has_status}")
            failed += 1
        else:
            print(f"   OK: no count/status drift in /predict_batch")
            passed += 1
    else:
        print(f"   FAIL: response is not a dict")
        failed += 1

    # ---- Summary ----
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed > 0:
        print("CONTRACT REGRESSION DETECTED!")
        sys.exit(1)
    else:
        print("ALL CONTRACT CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
