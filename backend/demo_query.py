"""
RAGMedX-MA — Live end-to-end demo query.

Usage (from d:\Ragmed_2\backend):
    d:\Ragmed_2\.venv-cu128-pilot\Scripts\python.exe demo_query.py

The server must already be running on http://127.0.0.1:8000 with
RAGMEDX_LLM_MODE=real before running this script.
"""

import httpx
import sys

BASE   = "http://127.0.0.1:8000"
EMAIL  = "hscr@ragmedx.dev"
PASSWD = "RagDoc99"

client = httpx.Client(base_url=BASE, timeout=300)

# ── Login ────────────────────────────────────────────────────────────────────
r = client.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWD})
if r.status_code != 200:
    print(f"Login failed ({r.status_code}): {r.text}")
    sys.exit(1)

token = r.json()["access_token"]
print(f"Login OK  (token length={len(token)})\n")

# ── Query ────────────────────────────────────────────────────────────────────
QUESTION = (
    "What is the role of amyloid beta and tau protein"
    "in Alzheimer's disease progression?"
)

print(f"Question: {QUESTION}")
print("Waiting for response (real GPU inference, takes ~2 min) ...")

response = client.post(
    "/api/v1/query/ask",
    headers={"Authorization": f"Bearer {token}"},
    json={"question": QUESTION, "include_agent_trace": True},
)

print(f"HTTP {response.status_code}\n")
data = response.json()

# ── Output ───────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"response_type : {data.get('response_type')}")
print(f"processing_ms : {data.get('processing_time_ms')}")

print("\n--- ANSWER (full) ---")
print(data.get("answer") or "(none)")

citations = data.get("citations") or []
print(f"\n--- CITATIONS ({len(citations)}) ---")
for c in citations:
    print(f"\n  {c.get('citation_marker')}  {c.get('title')}")
    print(f"  text: {(c.get('cited_text') or '')[:250]}")

# Pull critic verdict out of the agent trace
for step in ((data.get("agent_trace") or {}).get("steps") or []):
    if step.get("agent_type") == "critic":
        det = step.get("details") or {}
        print("\n--- CRITIC VERDICT ---")
        print(f"  passed              : {det.get('passed')}")
        print(f"  factual_consistency : {det.get('factual_consistency')}")
        print(f"  citation_support    : {det.get('citation_support')}")
        print(f"  hallucination_risk  : {det.get('hallucination_risk')}")
        break

client.close()
