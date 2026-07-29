"""
Public interface for the Retrieval Engine.

The backend orchestrator will only call retrieve_evidence().
All retrieval logic remains encapsulated within this module.
"""

from typing import List, Tuple


async def retrieve_evidence(
    query: str,
    optimized_query: str | None = None,
    max_iterations: int = 3,
):
    """
    Main entry point for the Retrieval Engine.

    Parameters
    ----------
    query : str
        Original medical question.

    optimized_query : str | None
        Query rewritten by the agent pipeline.

    max_iterations : int
        Maximum retrieval iterations.

    Returns
    -------
    Placeholder response.
    """

    print(f"Retrieving evidence for: {query}")

    return [], []