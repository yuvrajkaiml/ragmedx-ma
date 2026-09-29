import { AnswerData } from '../types/answer';

/**
 * REST API Request & Response Contracts for RAGMedX-MA
 * Target Endpoint: POST /api/v1/query/ask
 * 
 * Note: These are typed interfaces and stubs for future backend integration.
 * No actual network calls or fake responses are implemented here.
 */

export interface AskQueryRequest {
  query: string;
  modelTier?: 'Fast' | 'Balanced' | 'Thorough';
  confidenceThreshold?: number;
}

export interface AskQueryResponse {
  success: boolean;
  data?: AnswerData;
  error?: string;
}

// Service stub contract for future REST backend connection
export const submitAskQuery = async (_request: AskQueryRequest): Promise<AskQueryResponse> => {
  throw new Error('API service not connected. Backend integration expected in future phase.');
};
