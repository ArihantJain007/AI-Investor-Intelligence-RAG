# evaluate_rag.py – Local RAG evaluation script for Investor Intelligence Platform

import os
import json
import re
import sys
from pathlib import Path

import requests

BASE_URL = os.getenv("RAG_EVAL_API_URL", "http://127.0.0.1:8000")
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"

QUESTIONS_PATH = Path(__file__).with_name("questions.json")
RESULTS_PATH = Path(__file__).with_name("results.json")

TOLERANCE = 0.05  # 5% tolerance for numeric comparisons


def normalize_number(text: str) -> float:
    """Convert a textual representation of a number into a float.
    Handles commas, currency symbols, and million/billion suffixes.
    """
    if not text:
        return 0.0
    t = text.lower().replace(" ", "")
    t = re.sub(r"[\$€£]", "", t)
    multipliers = {
        "k": 1e3,
        "m": 1e6,
        "million": 1e6,
        "b": 1e9,
        "bn": 1e9,
        "billion": 1e9,
        "t": 1e12,
        "trillion": 1e12,
    }
    for suffix, mult in multipliers.items():
        if t.endswith(suffix):
            number_part = t[: -len(suffix)]
            number_part = number_part.replace(",", "")
            try:
                return float(number_part) * mult
            except ValueError:
                pass
    t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return 0.0


def extract_numbers(text: str) -> list[float]:
    """Find all numeric substrings in a piece of text and normalise them."""
    pattern = r"[\$€£]?\d[\d,.]*\s*(?:k|m|b|bn|million|billion|trillion)?"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return [normalize_number(m) for m in matches]


def load_questions() -> list[dict]:
    if not QUESTIONS_PATH.is_file():
        print(f"Questions file not found: {QUESTIONS_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    questions = load_questions()
    results = []
    counters = {
        "total": len(questions),
        "api_success": 0,
        "factual_match": 0,
        "unsupported_correct": 0,
        "cross_company_pass": 0,
        "qualitative_success": 0,
    }
    # Flags for one‑time diagnostics
    first_kpi_reported = False
    first_unsupported_reported = False
    for q in questions:
        payload = {"question": q["question"]}
        if q.get("company"):
            payload["company"] = q["company"]
        if q.get("year"):
            payload["year"] = q["year"]
        try:
            resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=30)
        except Exception as e:
            results.append({"question": q, "error": str(e)})
            continue
        result_entry = {"question": q, "status_code": resp.status_code}
        if resp.ok:
            counters["api_success"] += 1
            data = resp.json()
            result_entry["response"] = data
            raw_answer = data.get("answer")
            if isinstance(raw_answer, list):
                answer_text = " ".join(item.get("text", "") for item in raw_answer)
            elif isinstance(raw_answer, str):
                answer_text = raw_answer
            else:
                answer_text = ""
            result_entry["answer_text"] = answer_text
            expected = q.get("expected_answer")
            behavior = q.get("expected_behavior")
            category = q.get("category", "")
            # Numeric KPI check
            if expected and re.search(r"\d", str(expected)):
                expected_num = normalize_number(str(expected))
                numbers_in_answer = extract_numbers(answer_text)
                if numbers_in_answer:
                    match = any(
                        abs(value - expected_num) / max(expected_num, 1) <= TOLERANCE
                        for value in numbers_in_answer
                    )
                    if match:
                        counters["factual_match"] += 1
            elif behavior == "unsupported":
                # Unsupported when answer indicates lack of data or inability to answer
                if any(word in answer_text.lower() for word in ["insufficient", "unsupported", "cannot answer", "no data", "no relevant context", "not found in the knowledge base", "does not contain", "not available", "cannot be answered"]):
                    counters["unsupported_correct"] += 1
            elif category == "cross_company":
                # Pass when prohibited company's name does NOT appear in the answer
                prohibited = q.get("prohibited_company")
                if prohibited and prohibited.lower() not in answer_text.lower():
                    counters["cross_company_pass"] += 1
            elif category == "qualitative":
                # Qualitative success if any non‑empty answer was returned
                if answer_text.strip():
                    counters.setdefault("qualitative_success", 0)
                    counters["qualitative_success"] += 1
        else:
            result_entry["error"] = f"HTTP {resp.status_code}"
        results.append(result_entry)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("--- RAG Evaluation Summary ---")
    print(f"Total questions                : {counters['total']}")
    print(f"Successful API calls           : {counters['api_success']}")
    print(f"Factual answers matched        : {counters['factual_match']}")
    print(f"Unsupported questions handled  : {counters['unsupported_correct']}")
    print(f"Cross-company isolation passed : {counters['cross_company_pass']}")
    print(f"Qualitative answer success       : {counters.get('qualitative_success', 0)}")
if __name__ == "__main__":
    evaluate()
