# LangGraph orchestration

`MultiAgentOrchestrator` executes a compiled LangGraph `StateGraph`. The graph owns
control flow; individual agents only return state updates and trace steps.

```text
START
  |
Agent 1: Query Understanding
  |
Agent 2: Retrieval -> MGA
                  ^      |
                  |      +-- insufficient, pass < 3 -- focused sub-query --+
                  |      |
                  |      +-- insufficient / failed / capped --> Agent 6
                  |                                            Confidence
                  v                                              |
Agent 3: Evidence Re-Ranking / Validation <-- sufficient -------+
  |
  +-- contradiction / unsafe ------------------------------------+
  |
Agent 4: Generator <-- revision requested, round < 2 -- Agent 5: Critic
  |                                                          |
  +----------------------------------------------------------+
                                                             |
              verified --------------------------------------+
              rejected / revision-cap -----------------------+
                                                             v
                                                  Agent 6: Confidence
                                                             |
                                                             v
                                              Agent 7: Abstention Gate
                                                       |             |
                                                    PROCEED       ABSTAIN
                                                       |             |
                                                       +------> END <+
```

Every terminal route reaches Agent 7. Rejected or critic-cap states set
`force_abstention`; verified states still pass through Agent 7 so confidence and
other safety gates remain authoritative.

`response_builder.py` is deliberately **not** a graph node. Once `graph.ainvoke()`
returns terminal `AgentState`, `run_agent_pipeline()` calls `build_medical_response()`
to convert it into `MedicalResponse`. This preserves the external API and lets
`AgentTraceOutput` include all graph-node and agent-level trace entries for Pranav's
consumers before response formatting.
