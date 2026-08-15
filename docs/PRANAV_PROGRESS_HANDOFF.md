# RAGMedX-MA — Pranav Backend Progress Handoff

Last updated: 2026-08-08

## Ownership boundary

Pranav owns the backend API, authentication, orchestration, conversations,
WebSocket API, API schemas, documentation, and DevOps work.

Do **not** implement or alter Sana's retrieval-engine files under
`backend/app/retrieval/` unless the team explicitly agrees. Sana owns
ingestion, chunking, embeddings, vector storage, hybrid search, reranking,
and retrieval database models.

## Completed and verified

### App and authentication

- FastAPI app factory, CORS, SQLite initialization, and `/health` endpoint are
  working.
- JWT authentication works:
  - `POST /api/v1/auth/register`
  - `POST /api/v1/auth/login`
  - `POST /api/v1/auth/refresh`
- Authenticated profile APIs work:
  - `GET /api/v1/users/me`
  - `PATCH /api/v1/users/me`
- Fixed the `bcrypt 5.x` / `passlib 1.7.4` incompatibility by pinning
  `bcrypt==4.0.1` in `backend/requirements.txt`.

### Query contract

- Defined `MedicalQueryRequest` and the shared medical response contract.
- Added protected `POST /api/v1/query/ask`.
- Query service currently returns a safe `abstention` response until Sana's
  retrieval implementation and Yuvraj's agent pipeline are integrated.
- Removed an accidental duplicate `/api/v1/query/ask` route.

### Conversation API

- Added SQLAlchemy models:
  - `Conversation`
  - `ConversationMessage`
- Added Pydantic conversation contracts:
  - `ConversationCreate`
  - `ConversationUpdate`
  - `ConversationSummary`
  - `ConversationDetail`
  - `ConversationList`
- Added protected CRUD endpoints:

  | Method | Endpoint |
  |---|---|
  | POST | `/api/v1/chat/conversations` |
  | GET | `/api/v1/chat/conversations` |
  | GET | `/api/v1/chat/conversations/{conversation_id}` |
  | PATCH | `/api/v1/chat/conversations/{conversation_id}` |
  | DELETE | `/api/v1/chat/conversations/{conversation_id}` |

- Manual Swagger verification completed:
  - Create returned `201`.
  - Get-by-ID returned `200` with an empty `messages` list.
  - Delete returned `204 No Content`.
- API import, compilation, and route-registration checks passed with no
  duplicate path/method pairs.

### WebSocket streaming foundation

- Added `WebSocketEventType`, `QueryStreamSubscription`, and `WebSocketEvent`
  schemas.
- Added mock WebSocket endpoint:
  - `WS /api/v1/ws/query-stream?query_id=<UUID>`
- On connection, it sends an `agent_status` event.
- It responds to a `{"event_type": "ping"}` message with a `ping` event whose
  payload is `{"status": "pong"}`.
- Import and route-registration validation passed.

## Files changed or added during this work

- `backend/app/api/v1/endpoints/query.py`
- `backend/app/api/v1/endpoints/chat.py`
- `backend/app/api/v1/endpoints/__init__.py`
- `backend/app/api/v1/router.py`
- `backend/app/api/v1/websockets.py`
- `backend/app/core/security.py` (inspected; no code change needed)
- `backend/app/db/models.py`
- `backend/app/schemas/conversation.py`
- `backend/app/schemas/websocket.py`
- `backend/app/services/chat_service.py`
- `backend/requirements.txt`

### Retrieval files changed before ownership was clarified

These changes were made accidentally before confirming Sana's ownership. Do
not extend them without Sana's agreement:

- `backend/app/config.py` — retrieval configuration fields were added.
- `backend/app/schemas/retrieval.py` — new shared retrieval contracts added.
- `backend/app/retrieval/interface.py` — placeholder replaced by abstract
  embedder/vector-store interfaces.

Decide with Sana whether to retain, replace, or revert these three changes.

## Current architecture state

- API endpoints are under `backend/app/api/v1/`.
- Run the backend from the `backend` directory:

  ```powershell
  ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
  ```

- Swagger: `http://127.0.0.1:8000/docs`
- Database: SQLite, configured through `database_url` in `backend/app/config.py`.
- Tables are created automatically when the app starts.

## Important test workflow

1. Register via `POST /api/v1/auth/register`.
2. Copy the returned `access_token`.
3. In Swagger, click **Authorize** and paste the access token.
4. Use protected `/chat`, `/query`, and `/users` endpoints.

Never include an access token in commits, Markdown files, screenshots shared
publicly, or GitHub issues.

## Recommended next Pranav tasks

1. Finish the WebSocket test using the second-terminal script from the chat.
2. Add WebSocket authentication before production/frontend integration.
3. Add conversation message persistence when `/query/ask` produces a user
   question and a final assistant response.
4. Add conversation export (JSON, Markdown, then PDF).
5. Implement document-upload API contract and status endpoints; coordinate
   storage/indexing handoff with Sana.
6. Add admin metrics endpoints and centralized exception/logging handling.
7. When Sana and Yuvraj provide their public functions, update
   `QueryService` to call them while preserving the abstention fallback.
8. Add automated API tests with `pytest` and `pytest-asyncio`.
9. Before GitHub push: inspect Git status, commit only intended files, merge
   teammate work carefully, run checks, and push a clean branch.

## Key integration contracts

- Sana should expose `retrieve_evidence(query, max_iter=3)`.
- Yuvraj should expose `run_agent_pipeline(query, evidence)`.
- Pranav's `POST /api/v1/query/ask` must orchestrate those functions and
  return `MedicalResponse`.
- Frontend uses:
  - `POST /api/v1/query/ask`
  - Conversation CRUD endpoints
  - `WS /api/v1/ws/query-stream?query_id=<UUID>`

