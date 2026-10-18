/**
 * DEVELOPMENT / UI TESTING ONLY — NOT REAL MEDICAL DATA.
 * 
 * Mock data fixtures for testing completed RAGMedX-MA frontend UI components.
 * Contains fictional test values for answers, trace phases, evidence, history, and errors.
 */

import { AnswerData } from '../types/answer';
import { AgentTraceStep } from '../types/agent';
import { PassageItem, EvidenceSourceItem, DatasetItem } from '../types/evidence';
import { HistorySessionItem } from '../types/history';
import { ErrorType } from '../components/common/ErrorCard';

// Single toggle to enable/disable mock testing mode across the frontend
// Set to false when connecting to the real backend.
export const MOCK_MODE_ENABLED = false;

// ----------------------------------------------------------------------
// 1. MOCK VERIFIED ANSWER FIXTURE
// ----------------------------------------------------------------------
export const MOCK_VERIFIED_ANSWER: AnswerData = {
  id: 'mock-ans-001',
  query: '[MOCK TEST] What is the recommended treatment protocol for severe hypotension in synthetic shock syndrome?',
  state: 'verified',
  confidence: {
    score: 92,
    level: 'High',
  },
  summary: '[MOCK TEST DATA] First-line management involves aggressive intravenous crystalloid fluid resuscitation [1] accompanied by early initiation of norepinephrine vasopressor therapy [2].',
  body: '[MOCK TEST DATA] If mean arterial pressure remains below 65 mmHg despite adequate fluid challenge, secondary vasopressor agents such as vasopressin or epinephrine are indicated [3]. Close monitoring of cardiac output and systemic vascular resistance is recommended in intensive care settings.',
  createdAt: '2 mins ago',
  citations: ['cit-1', 'cit-2', 'cit-3'],
  citationObjects: [
    {
      id: 'cit-1',
      markerIndex: 1,
      sourceId: 'src-1',
      claimSnippet: '[MOCK TEST] Crystalloid fluid challenge recommendation',
      relevanceScore: 0.95,
    },
    {
      id: 'cit-2',
      markerIndex: 2,
      sourceId: 'src-2',
      claimSnippet: '[MOCK TEST] Norepinephrine first-line vasopressor protocol',
      relevanceScore: 0.91,
    },
    {
      id: 'cit-3',
      markerIndex: 3,
      sourceId: 'src-3',
      claimSnippet: '[MOCK TEST] Secondary vasopressor escalation guidelines',
      relevanceScore: 0.88,
    },
  ],
  sourceIds: ['src-1', 'src-2', 'src-3'],
};

// ----------------------------------------------------------------------
// 2. MOCK REVISED ANSWER FIXTURE
// ----------------------------------------------------------------------
export const MOCK_REVISED_ANSWER: AnswerData = {
  id: 'mock-ans-002',
  query: '[MOCK TEST] Can novel oral anticoagulants be safely administered in severe chronic kidney disease?',
  state: 'revised',
  confidence: {
    score: 74,
    level: 'Moderate',
  },
  summary: '[MOCK TEST DATA] Dose adjustments are strictly required for apixaban and rivaroxaban in severe renal impairment (CrCl 15-29 mL/min) [1]. Dabigatran is contraindicated when CrCl falls below 30 mL/min [2].',
  body: '[MOCK TEST DATA] Routine monitoring of renal function and anti-Xa levels may be warranted in vulnerable patient cohorts to prevent supratherapeutic bleeding risks.',
  revisionNote: '[MOCK TEST CRITIC REVISION] The initial draft overstated dabigatran safety in CrCl <30 mL/min. The critic agent corrected the recommendation and added dosage reduction warnings.',
  originalAnswer: '[MOCK TEST UNREVISED DRAFT] Oral anticoagulants can be administered across all stages of renal impairment without significant dosage changes.',
  createdAt: '10 mins ago',
  citations: ['cit-1', 'cit-2'],
  citationObjects: [
    {
      id: 'cit-1',
      markerIndex: 1,
      sourceId: 'src-1',
      claimSnippet: '[MOCK TEST] Apixaban renal dosage reduction criteria',
      relevanceScore: 0.89,
    },
    {
      id: 'cit-2',
      markerIndex: 2,
      sourceId: 'src-2',
      claimSnippet: '[MOCK TEST] Dabigatran clearance contraindications in renal failure',
      relevanceScore: 0.84,
    },
  ],
  sourceIds: ['src-1', 'src-2'],
};

// ----------------------------------------------------------------------
// 3. MOCK ABSTAINED ANSWER FIXTURE
// ----------------------------------------------------------------------
export const MOCK_ABSTAINED_ANSWER: AnswerData = {
  id: 'mock-ans-003',
  query: '[MOCK TEST] What is the efficacy of experimental peptide Alpha-7 in treating pediatric auto-inflammatory disorders?',
  state: 'abstained',
  confidence: {
    score: 31,
    level: 'Insufficient',
  },
  summary: '[MOCK TEST DATA] System gatekeeper abstained from providing a clinical recommendation due to insufficient and conflicting published phase-III trial evidence.',
  abstentionReason: '[MOCK TEST ABSTENTION REASON] Available literature consists of contradictory small-sample observational studies without randomized control validation.',
  partialFindings: [
    '[MOCK TEST] Phase-I safety trials showed mild tolerance in adult populations (N=24).',
    '[MOCK TEST] No peer-reviewed randomized controlled trials exist for pediatric cohorts.',
    '[MOCK TEST] Observational cohort data demonstrated inconsistent biomarker reduction.',
  ],
  recommendedNextActions: [
    '[MOCK TEST] Consult specialized pediatric rheumatology clinical guidelines.',
    '[MOCK TEST] Query ClinicalTrials.gov for active recruiting Phase-II/III protocols.',
    '[MOCK TEST] Re-evaluate query with broader keyword parameters.',
  ],
  createdAt: '15 mins ago',
  citations: [],
  sourceIds: [],
};

// ----------------------------------------------------------------------
// 4. MOCK AGENT TRACE (ALL 7 PHASES)
// ----------------------------------------------------------------------
export const MOCK_AGENT_TRACE_STEPS: AgentTraceStep[] = [
  {
    id: 'trace-1',
    phase: 'Query Understanding',
    agent: 'Retrieval',
    action: 'Deconstruct clinical question & extract medical concepts',
    status: 'completed',
    timestamp: '00:00.12',
    details: '[MOCK TEST] Identified core entities: Hypotension, Resuscitation, Norepinephrine. Expanded sub-terms.',
    subQueries: ['severe hypotension resuscitation', 'norepinephrine vasopressor protocol MAP 65'],
    queryOrPayload: '{"entities": ["hypotension", "resuscitation"], "intent": "treatment_protocol"}',
  },
  {
    id: 'trace-2',
    phase: 'Iterative Retrieval',
    agent: 'Retrieval',
    action: 'Execute hybrid dense-sparse vector search across PubMed & PMC corpus',
    status: 'completed',
    timestamp: '00:00.48',
    details: '[MOCK TEST] Retrieved 14 relevant document passages. Filtered out duplicate abstracts.',
    subQueries: ['crystalloid vs colloid resuscitation trials', 'second-line vasopressor escalation vasopressin'],
  },
  {
    id: 'trace-3',
    phase: 'Re-ranking',
    agent: 'Retrieval',
    action: 'Re-rank evidence passages using cross-encoder relevance model',
    status: 'completed',
    timestamp: '00:00.82',
    details: '[MOCK TEST] Top 3 passages selected based on clinical relevance score threshold (>0.85).',
  },
  {
    id: 'trace-4',
    phase: 'Generator',
    agent: 'Generator',
    action: 'Draft clinical answer grounded strictly in top-3 retrieved passages',
    status: 'completed',
    timestamp: '00:01.35',
    details: '[MOCK TEST] Synthesized initial answer draft with embedded [1], [2], [3] citation markers.',
    queryOrPayload: '[MOCK DRAFT TEXT] Resuscitation begins with crystalloid fluids followed by norepinephrine...',
  },
  {
    id: 'trace-5',
    phase: 'Critic',
    agent: 'Critic',
    action: 'Audit draft claims against evidentiary source passages for hallucination & bias',
    status: 'completed',
    timestamp: '00:01.90',
    details: '[MOCK TEST] Verified all claims against cited passages. No unsupported assertions detected.',
  },
  {
    id: 'trace-6',
    phase: 'Confidence Calibration',
    agent: 'Revision',
    action: 'Compute aggregate confidence score and evidentiary alignment index',
    status: 'completed',
    timestamp: '00:02.10',
    details: '[MOCK TEST] Calculated high confidence score (92%). Consensus alignment across 3 independent sources.',
  },
  {
    id: 'trace-7',
    phase: 'Abstention Gatekeeper',
    agent: 'Revision',
    action: 'Evaluate safety threshold & abstention gate criteria',
    status: 'completed',
    timestamp: '00:02.25',
    details: '[MOCK TEST] Passed abstention gate (Confidence > 80% threshold). Approved for final presentation.',
  },
];

// ----------------------------------------------------------------------
// 5. MOCK EVIDENCE PASSAGES & SOURCES
// ----------------------------------------------------------------------
export const MOCK_PASSAGES: PassageItem[] = [
  {
    id: 'pas-1',
    markerIndex: 1,
    sourceId: 'src-1',
    sourceTitle: '[MOCK TEST] Guidelines for Fluid Resuscitation in Acute Circulatory Failure',
    journal: 'Journal of Test Cardiology',
    year: 2024,
    snippet: '[MOCK TEST DATA] Initial fluid resuscitation should begin with at least 30 mL/kg of IV balanced crystalloid solution within the first 3 hours of hypotension onset.',
    gradeRating: 'HIGH',
    relevanceScore: 0.95,
  },
  {
    id: 'pas-2',
    markerIndex: 2,
    sourceId: 'src-2',
    sourceTitle: '[MOCK TEST] Comparative Efficacy of Vasopressors in Septic Shock',
    journal: 'Annals of Mock Medicine',
    year: 2023,
    snippet: '[MOCK TEST DATA] Norepinephrine is recommended as the first-choice vasopressor to target a mean arterial pressure (MAP) of 65 mmHg.',
    gradeRating: 'HIGH',
    relevanceScore: 0.91,
  },
  {
    id: 'pas-3',
    markerIndex: 3,
    sourceId: 'src-3',
    sourceTitle: '[MOCK TEST] Escalation Protocols for Refractory Vasoplegic Shock',
    journal: 'Critical Care Research Review',
    year: 2025,
    snippet: '[MOCK TEST DATA] Vasopressin (up to 0.03 units/min) can be added to norepinephrine with the intent of raising MAP or reducing norepinephrine dosage.',
    gradeRating: 'MODERATE',
    relevanceScore: 0.88,
    isContradictory: false,
  },
];

export const MOCK_SOURCES: EvidenceSourceItem[] = [
  {
    id: 'src-1',
    title: '[MOCK TEST] Guidelines for Fluid Resuscitation in Acute Circulatory Failure',
    journal: 'Journal of Test Cardiology',
    year: 2024,
    pmid: '38192001',
    doi: '10.1016/j.mockcard.2024.01.005',
    snippet: '[MOCK TEST DATA] Multi-center guideline recommendation for fluid administration in acute circulatory collapse.',
    relevanceScore: 0.95,
    citationCount: 42,
    tag: 'Guideline',
    url: 'https://example.com/mock-paper-1',
  },
  {
    id: 'src-2',
    title: '[MOCK TEST] Comparative Efficacy of Vasopressors in Septic Shock',
    journal: 'Annals of Mock Medicine',
    year: 2023,
    pmid: '37281092',
    doi: '10.1001/mockmed.2023.08.112',
    snippet: '[MOCK TEST DATA] Randomized double-blind trial comparing norepinephrine versus dopamine in shock management.',
    relevanceScore: 0.91,
    citationCount: 118,
    tag: 'RCT',
    url: 'https://example.com/mock-paper-2',
  },
  {
    id: 'src-3',
    title: '[MOCK TEST] Escalation Protocols for Refractory Vasoplegic Shock',
    journal: 'Critical Care Research Review',
    year: 2025,
    pmid: '39012344',
    doi: '10.1055/ccrr.2025.02.019',
    snippet: '[MOCK TEST DATA] Systematic review of combination vasopressor therapies in refractory shock.',
    relevanceScore: 0.88,
    citationCount: 19,
    tag: 'Combination therapy',
    url: 'https://example.com/mock-paper-3',
  },
];

// ----------------------------------------------------------------------
// 6. MOCK HISTORY SESSIONS
// ----------------------------------------------------------------------
export const MOCK_HISTORY_SESSIONS: HistorySessionItem[] = [
  {
    id: 'hist-001',
    question: '[MOCK TEST] What is the recommended treatment protocol for severe hypotension?',
    answerSummary: '[MOCK TEST DATA] First-line management involves aggressive IV crystalloid fluid resuscitation accompanied by early norepinephrine vasopressor therapy.',
    state: 'verified',
    confidenceScore: 92,
    confidenceLevel: 'High',
    date: 'Today, 2:15 PM',
    sourceCount: 3,
    fullAnswerData: MOCK_VERIFIED_ANSWER,
    agentTraceSteps: MOCK_AGENT_TRACE_STEPS,
    passages: MOCK_PASSAGES,
    sources: MOCK_SOURCES,
  },
  {
    id: 'hist-002',
    question: '[MOCK TEST] Can novel oral anticoagulants be safely administered in severe CKD?',
    answerSummary: '[MOCK TEST DATA] Dose adjustments are required for apixaban and rivaroxaban in severe renal impairment (CrCl 15-29 mL/min). Dabigatran is contraindicated.',
    state: 'revised',
    confidenceScore: 74,
    confidenceLevel: 'Moderate',
    date: 'Yesterday, 10:30 AM',
    sourceCount: 2,
    fullAnswerData: MOCK_REVISED_ANSWER,
    agentTraceSteps: MOCK_AGENT_TRACE_STEPS,
    passages: MOCK_PASSAGES.slice(0, 2),
    sources: MOCK_SOURCES.slice(0, 2),
  },
  {
    id: 'hist-003',
    question: '[MOCK TEST] What is the efficacy of experimental peptide Alpha-7 in pediatric disorders?',
    answerSummary: '[MOCK TEST DATA] System gatekeeper abstained from providing a clinical recommendation due to insufficient and conflicting published phase-III trial evidence.',
    state: 'abstained',
    confidenceScore: 31,
    confidenceLevel: 'Insufficient',
    date: 'Sep 25, 2026',
    sourceCount: 0,
    fullAnswerData: MOCK_ABSTAINED_ANSWER,
    agentTraceSteps: MOCK_AGENT_TRACE_STEPS,
    passages: [],
    sources: [],
  },
];

// ----------------------------------------------------------------------
// 7. MOCK ERROR STATE FIXTURES
// ----------------------------------------------------------------------
export const MOCK_ERRORS: Record<ErrorType, { title: string; message: string }> = {
  invalid_query: {
    title: '[MOCK TEST] Query Validation Error',
    message: 'The submitted query string was too short or did not contain recognizable medical terms.',
  },
  retrieval_failure: {
    title: '[MOCK TEST] Retrieval Exception',
    message: 'Zero evidence passages matched the query in the PubMed/PMC vector index.',
  },
  model_failure: {
    title: '[MOCK TEST] Multi-Agent Pipeline Error',
    message: 'The generator LLM timed out while synthesizing the grounded answer draft.',
  },
  out_of_domain: {
    title: '[MOCK TEST] Domain Boundary Warning',
    message: 'The question was flagged as non-medical/out-of-domain by the gatekeeper agent.',
  },
  contradictory_evidence: {
    title: '[MOCK TEST] Contradictory Evidence Alert',
    message: 'Top retrieved literature sources contained directly opposing therapeutic claims.',
  },
  timeout: {
    title: '[MOCK TEST] Request Timed Out',
    message: 'The pipeline took longer than expected on this machine before a response arrived.',
  },
  network: {
    title: '[MOCK TEST] Connection Problem',
    message: 'The frontend could not reach the backend API/WebSocket service.',
  },
  auth: {
    title: '[MOCK TEST] Sign In Required',
    message: 'Guest mode cannot run live queries; a signed-in account is required.',
  },
};

// ----------------------------------------------------------------------
// 8. MOCK DATASET TRANSPARENCY CARDS
// ----------------------------------------------------------------------
export const MOCK_DATASETS: DatasetItem[] = [
  {
    id: 'ds-pubmed',
    name: 'PubMed',
    description: '[MOCK DATASET] MEDLINE biomedical literature index from NLM.',
    coverage: 'Biomedical Abstracts & Citations',
    documentCount: '36.5M abstracts',
    lastUpdated: 'Mock 2026',
    url: 'https://pubmed.ncbi.nlm.nih.gov/',
  },
  {
    id: 'ds-pmc',
    name: 'PMC',
    description: '[MOCK DATASET] PubMed Central open-access full-text repository.',
    coverage: 'Full-text Peer-reviewed Articles',
    documentCount: '9.2M full articles',
    lastUpdated: 'Mock 2026',
    url: 'https://www.ncbi.nlm.nih.gov/pmc/',
  },
  {
    id: 'ds-medqa',
    name: 'MedQA',
    description: '[MOCK DATASET] Clinical USMLE-style diagnostic benchmark dataset.',
    coverage: 'Clinical Diagnosis Benchmarks',
    documentCount: '12.7K board questions',
    lastUpdated: 'Mock 2026',
    url: 'https://github.com/jair-ai/MedQA',
  },
  {
    id: 'ds-bioasq',
    name: 'BioASQ',
    description: '[MOCK DATASET] Large-scale biomedical semantic QA & fact retrieval benchmark.',
    coverage: 'Biomedical Fact QA & Task Benchmarks',
    documentCount: '4.5K challenge sets',
    lastUpdated: 'Mock 2026',
    url: 'http://bioasq.org/',
  },
];
