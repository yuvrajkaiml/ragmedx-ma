"""
Phase 4a — Critic Analysis CLI (thin wrapper).

The analysis logic now lives in the importable, testable module
``backend/evaluation/agent_contribution.py``.  This script is kept as the
canonical entry point; it simply imports :func:`run_analysis` from that
module and runs it, writing a new timestamped::

    backend/results/critic_analysis_<timestamp>.json
    backend/results/critic_analysis_<timestamp>.md

Honesty note: the n30 result files record only ``{question, true_answer,
predicted_answer, response_type, evidence_count, evidence_titles,
elapsed_ms}`` per sample — no ``draft_answer`` or ``critic_verdict``.  So
per-question critic accept/reject/rewrite and ROUGE-L draft-vs-final are
NOT MEASURED (labelled as such in the output); only population-level
cross-config abstention deltas are reported, evidence_tier = SOFTWARE TEST.
No AUTOMATED MODEL JUDGE tier is produced.

Usage:
    python backend/scripts/generate_critic_analysis.py
"""

import sys
from pathlib import Path

# Make ``backend`` importable when run as a plain script (``python <script>``),
# where only the script's own directory is on sys.path by default.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from evaluation.agent_contribution import run_analysis  # noqa: E402


def main() -> None:
    """Run the critic analysis and write the timestamped JSON + Markdown outputs."""
    run_analysis()


if __name__ == "__main__":
    main()
