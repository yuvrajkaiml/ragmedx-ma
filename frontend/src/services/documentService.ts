/**
 * RAGMedX-MA â€” Document Upload Service
 *
 * POST /api/v1/documents/upload  â€” upload a PDF, get back document_id
 * GET  /api/v1/documents/{id}/status â€” poll until status === 'indexed'
 * DELETE /api/v1/documents/{id}  â€” remove document + its index
 */

import { getAccessToken } from './authService';

const BASE = '/api/v1';

export type DocumentStatus = 'uploaded' | 'queued' | 'processing' | 'indexed' | 'failed' | 'deleted';

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  uploaded_at: string;
  message: string;
}

export interface DocumentStatusResponse {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  uploaded_at: string;
  error_message: string | null;
  chunk_count: number | null;
}

function authHeader(): Record<string, string> {
  const token = getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** Upload a PDF. Returns immediately with status=queued. */
export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    throw new Error('Only PDF files are supported.');
  }
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${BASE}/documents/upload`, {
    method: 'POST',
    headers: authHeader(),
    body: form,
  });
  const json = await res.json();
  if (!res.ok) throw new Error(json.message || json.detail || 'Upload failed');
  return json as DocumentUploadResponse;
}

/** Poll status until indexed/failed or timeout (default 120s). */
export async function pollUntilIndexed(
  documentId: string,
  onStatus?: (status: DocumentStatus, chunkCount: number | null) => void,
  maxWaitMs = 120_000,
): Promise<DocumentStatusResponse> {
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2_500));
    const res = await fetch(`${BASE}/documents/${documentId}/status`, {
      headers: authHeader(),
    });
    if (!res.ok) continue;
    const data: DocumentStatusResponse = await res.json();
    onStatus?.(data.status, data.chunk_count);
    if (data.status === 'indexed' || data.status === 'failed') return data;
  }
  throw new Error('Document indexing timed out after 120s.');
}

/** Delete a document and its index. */
export async function deleteDocument(documentId: string): Promise<void> {
  await fetch(`${BASE}/documents/${documentId}`, {
    method: 'DELETE',
    headers: authHeader(),
  });
}

