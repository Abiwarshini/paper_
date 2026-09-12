import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5005"


def send_post(endpoint, payload):
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))
    except Exception as e:
        return 500, {"error": str(e)}


def send_get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))
    except Exception as e:
        return 500, {"error": str(e)}


def test_suite():
    print("=" * 70)
    print("      TESTING 4-MODEL AI MALNUTRITION PREDICTION SUITE (Port 5005)")
    print("=" * 70)

    # 1. Health Check
    print("\n[Test 1] Health Check (/health)...")
    status, data = send_get("/health")
    print(f"Status: {status}")
    print(f"Service: {data.get('service')}")
    print(f"Supported Models: {data.get('supported_models')}")
    assert status == 200, "Health check failed!"

    # Sample Child Profiles
    high_risk_child = {
        "child_age_months": 18,
        "child_sex": "Female",
        "birth_weight": 2.1,
        "birth_size": "Smaller than Average",
        "breastfeeding_duration": 8,
        "birth_order": 4,
        "mother_bmi": 17.2,
        "education": "No Education",
        "anc_visits": 1,
        "wealth_quintile": "Poorest",
        "residence": "Rural",
        "hhsize": 7,
        "sanitation_risk_index": 3,
        "diarrhea_recent": "Yes",
        "fever_recent": "Yes",
        "cough_recent": "Yes",
        "measles_vaccine": "No"
    }

    # 2. Consensus Endpoint (All 4 Models)
    print("\n[Test 2] Consensus Prediction (/predict) with High-Risk Child...")
    status, res = send_post("/predict", high_risk_child)
    print(f"Status: {status}")
    print(f"Best Model: {res.get('best_model')}")
    for m in ["xgboost", "transformer", "dnn", "tabnet"]:
        m_data = res.get(m, {})
        print(f"  {m.upper():12s} | Pred: {m_data.get('prediction')} | Prob: {m_data.get('percentage')}% | Risk: {m_data.get('overall_risk')}")
    assert status == 200, "/predict endpoint failed!"

    # 3. Individual Models
    for model_key in ["xgboost", "transformer", "dnn", "tabnet"]:
        print(f"\n[Test] Individual Endpoint (/predict/{model_key})...")
        st, r = send_post(f"/predict/{model_key}", high_risk_child)
        print(f"Status: {st} | Model: {r.get('model')} | Overall Risk: {r.get('overall_risk')} | Factors count: {len(r.get('top_factors', []))}")
        assert st == 200, f"/predict/{model_key} failed!"

    # 4. Model Benchmark Comparison
    print("\n[Test 4] Benchmark Comparison (/compare)...")
    status, bench = send_get("/compare")
    print(f"Status: {status} | Models Evaluated: {list(bench.get('overall_summary', {}).keys())}")
    assert status == 200, "/compare endpoint failed!"

    # 5. Missing Fields / Partial Payload Handling
    print("\n[Test 5] Partial Payload Robustness (/predict)...")
    partial_child = {"age_months": 20, "weight_kg": 9.2, "height_cm": 78.0}
    status, p_res = send_post("/predict", partial_child)
    print(f"Status: {status} | Fallback Handled: {p_res.get('status')}")
    assert status == 200, "Partial payload test failed!"

    print("\n" + "=" * 70)
    print(" ALL 4-MODEL API TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_suite()
