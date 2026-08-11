**UI/UX DESIGN BRIEF**

**RAGMedX-MA**

*A Multi-Agent Iterative Retrieval Framework with Peer-Reviewed Generation for High-Confidence Medical Question Answering*

Group No. 09 — Department of AIML

Shravani Waman · Pranav Sukale · Sana Thorat · Yuvraj K

Document prepared: July 2026

# **1\. Purpose of This Document**

This brief translates the RAGMedX-MA system architecture into a concrete user experience: what the user sees, in what order, and why each element earns their trust. Because the product answers medical questions, the interface has one job above all others — it must never let an answer look more certain than the evidence behind it. Every screen, label, and interaction below is built around that constraint.

Use this document as the shared reference for wireframing, prototyping, and the eventual frontend build in React.js. It does not prescribe pixel-level visuals; it defines structure, hierarchy, states, and content rules that any visual designer or developer can implement consistently.

# **2\. Project Recap**

RAGMedX-MA extends the original RAGMedX pipeline (FAISS + LangChain + PubMedBERT + local LLM) into a multi-agent, evidence-gated system with three upgrades that directly shape the UI:

-   Iterative retrieval — the system may issue several follow-up searches before answering a complex query, not just one pass.
-   Confidence & abstention — if retrieved evidence is weak or contradictory, the system declines to answer rather than guessing.
-   Multi-agent peer review — a Generator agent drafts the answer and a Critic agent checks it against retrieved literature before it reaches the user.

The UI's central challenge is making these three backend behaviors visible and legible, so the user experiences them as transparency and safety rather than latency and complexity.

# **3\. UX Goals & Design Principles**

## **3.1 Experience Goals**

-   Evidence before answer — the user should always be able to see what the system read before trusting what it says.
-   Calibrated confidence — visual weight of an answer should scale with the system's actual certainty, never higher.
-   Graceful refusal — an abstention should feel like responsible caution, not a broken product.
-   Inspectable reasoning — the multi-agent process (generate → critique → revise) should be optionally visible, never hidden or mysterious.
-   Fast perceived response — iterative retrieval takes longer than single-pass RAG, so the UI must narrate progress instead of showing a blank spinner.

## **3.2 Design Principles**

-   Clinical calm — quiet layout, generous whitespace, no marketing gloss. This is a tool for careful reading, not a consumer chat app.
-   Provenance is a first-class citizen — citations sit beside the answer, not buried in a footnote or a click-to-reveal tooltip.
-   Color carries meaning, not decoration — teal/green, amber, and red map consistently to verified, uncertain, and conflicting evidence everywhere in the app.
-   Progressive disclosure — the default view is simple (answer + confidence + top sources); agent traces, per-chunk scores, and retrieval logs live one level deeper for power users.
-   No false authority — the interface never lets typography or layout imply more medical certainty than the confidence score does.

# **4\. Target Users & Personas**

The system serves three distinct audiences with different depth needs from the same core screen.

**Persona**

**Primary Goal**

**Key Need from UI**

**Trust Trigger**

Dr. Anaya Kulkarni Physician / resident

Quick, literature-grounded answers during clinical review or teaching

Fast path to primary sources; ability to expand full evidence and filter by recency

Named journals, DOIs, and publication dates visible without extra clicks

Ravi, MBBS student

Study complex clinical vignettes and understand reasoning, not just the answer

Visible reasoning trace; ability to see how the agents reached a conclusion

Step-by-step agent trace framed as a learning aid

Meera, informed caregiver

Understand a family member's condition in plain language

Plain-language summary with a clear 'this is not medical advice' framing and an easy path to ask a licensed clinician

Honest abstention when evidence is thin, rather than a confident-sounding guess

*All three personas share one requirement that overrides their individual differences: they must be able to tell, within two seconds of reading an answer, how much to trust it.*

# **5\. Core User Flow**

The primary flow is a single continuous conversation, but it now has more visible internal steps than a standard chatbot:

-   1\. Query entry — user types a clinical question or selects a suggested example query.
-   2\. Retrieval in progress — status narration shows what the system is doing ("Searching PubMed and MedQA…", "Follow-up search: dosage guidelines…") rather than a generic spinner.
-   3\. Evidence assembled — a source list appears, each entry re-ranked and tagged with a relevance/quality signal before generation begins.
-   4\. Draft generated — the Generator agent's answer appears, visually marked as "under review" for a beat while the Critic agent checks it.
-   5\. Critic verdict — one of three outcomes is shown: Verified (aligned with evidence), Revised (Critic flagged an issue, answer was corrected), or Abstained (evidence insufficient or contradictory).
-   6\. Final answer card — confidence score, cited sources, and an expandable reasoning trace are presented together.
-   7\. Follow-up — user can ask a refining question, request "show your work," or flag the answer for feedback.

# **6\. Information Architecture**

## **6.1 Primary Navigation**

-   Ask (default landing) — the query and answer stream
-   History — past sessions, searchable by topic or date
-   Evidence Library — datasets and sources the system can draw from (PubMed, PMC, MedQA, BioASQ), shown for transparency
-   Settings — model/agent verbosity, confidence-threshold display, accessibility options

## **6.2 Answer Card Hierarchy**

-   Level 1 (always visible): question, answer summary, confidence badge, top 2–3 sources
-   Level 2 (one click): full evidence list with per-source relevance score, publication metadata
-   Level 3 (one click, labeled "Show agent reasoning"): Generator draft, Critic feedback, revision diff, retrieval queries issued

# **7\. Key Screens & States**

**Screen / State**

**Purpose**

**Key UI Elements**

Ask screen (empty state)

Orient a new user and set expectations

Search-style input, 3–4 example clinical questions, one-line disclaimer, dataset badges (PubMed, MedQA, BioASQ)

Retrieval-in-progress state

Keep the wait legible during iterative search

Step tracker ("Searching → Re-ranking → Drafting → Reviewing"), live text of the current sub-query

Answer card — Verified

Deliver a high-confidence, evidence-checked answer

Green/teal confidence badge, answer text, inline citation markers \[1\]\[2\], source cards with journal + year

Answer card — Revised

Show that peer review changed the answer, building trust in the process

Amber "Critic revised this answer" tag, optional toggle to view original draft vs. final

Abstention state

Refuse responsibly instead of guessing

Amber/red banner: "Insufficient reliable evidence to answer confidently," reason summary, suggested reformulation or related sources

Agent reasoning trace (expanded)

Give power users full transparency

Vertical timeline: Query → Retrieval agent results → Generator draft → Critic review notes → Final answer

Evidence Library

Let users audit what the system can and can't see

List of datasets with coverage description, last-updated date, and document counts

Session history

Support returning users and continuity of care/study

Chronological list, topic tags, quick re-open

# **8\. Core Components**

**Component**

**Behavior**

**Visual Treatment**

Confidence badge

Reflects the model's calibrated confidence score, not just a binary pass/fail

Pill-shaped label: "High confidence" (teal), "Moderate — verify further" (amber), "Insufficient evidence" (red), each with the numeric score on hover/tap

Citation chip

Inline marker linking a claim to its source

Small numbered chip \[1\] in-line with text; tapping opens a source card, never navigates away

Source card

Summarizes one retrieved document

Title, journal/publisher, year, one-line relevance note, external link icon

Agent trace timeline

Shows the Generator → Critic → Revision sequence

Collapsed by default; vertical stepper with short status labels, expandable per step

Abstain banner

Communicates a refusal without feeling like an error

Full-width card, calm amber tone (not red/error styling), plain-language reason, one suggested next action

Disclaimer footer

Persistent legal/safety framing

Thin, low-emphasis text fixed under every answer: "For informational purposes; not a substitute for professional medical advice."

# **9\. Visual System**

## **9.1 Color Palette**

Color is functional before it is aesthetic: the same three-color logic (teal = verified, amber = caution, red = conflict) must repeat identically across badges, banners, and the evidence library so users build a reliable mental model.

**Token**

**Swatch**

**Usage**

Primary — Clinical Navy

Headers, primary actions, chat AI messages, navigation

Secondary — Evidence Teal

Citations, verified badges, retrieval source tags

Caution — Abstain Amber

Low-confidence flags, 'insufficient evidence' banners

Alert — Conflict Red

Contradictory evidence, critic-agent rejections

Neutral Ink

Body text

Neutral Subtle

Secondary text, timestamps, metadata

## **9.2 Typography**

-   Headings — a humanist serif (e.g., Georgia/Source Serif) for a clinical, editorial tone rather than a startup-chat tone.
-   Body & UI text — a clean sans (e.g., Calibri/Inter) at 16–18px minimum for readability of dense medical text.
-   Citations and metadata — same sans, smaller size and the Subtle gray token, so they read as supporting evidence rather than competing with the answer.

## **9.3 Layout**

-   Single-column chat column (max ~720px) with evidence panel docked to the right on desktop, collapsible to a bottom sheet on mobile.
-   Generous line height (1.5+) and paragraph spacing for medical text that is often dense and technical.
-   No decorative accent bars or gradients — the interface should read as a clinical reference tool, not a marketing product.

# **10\. Interaction & Microcopy Guidelines**

-   Never state a percentage without a plain-language label next to it ("78% — Moderate confidence"), since raw numbers alone invite over-trust.
-   Abstention copy should name the gap, not just refuse: "The retrieved studies conflict on dosage for this age group" beats "I don't know."
-   Every revision by the Critic agent should be described in one short clause the user can read without expanding the trace, e.g., "Adjusted after reviewing conflicting evidence."
-   Keep the safety disclaimer present but understated — one line, consistent placement, never a modal that blocks reading.

# **11\. Accessibility & Trust Safeguards**

-   Color is never the only signal — every confidence badge and banner pairs color with a text label and an icon shape (check, dash, or triangle).
-   Full keyboard navigation for citation chips and the agent trace stepper; each is a focusable, announced element for screen readers.
-   Minimum WCAG AA contrast for all text, including the Subtle gray metadata tone against both white and Light-BG backgrounds.
-   Session history and evidence library respect data-privacy expectations for health-related queries — no third-party analytics on query content.

# **12\. Responsive Behavior**

-   Desktop — two-column layout: conversation on the left, persistent evidence/agent-trace panel on the right.
-   Tablet — single column with the evidence panel as a slide-over drawer.
-   Mobile — single column; sources and agent trace collapse into expandable accordions directly under the answer card to avoid a second screen.

# **13\. Next Steps for the Team**

-   Translate Section 7 into low-fidelity wireframes for the Ask screen, Answer card (all three states), and Agent trace view.
-   Validate the abstention and revised-answer copy with a few sample clinical queries pulled from MedQA before building the React components.
-   Build the Confidence badge, Citation chip, and Abstain banner as reusable React components first — they appear in every other screen.
-   Usability-test the "Show agent reasoning" disclosure with both a physician and a lay-caregiver persona to confirm progressive disclosure is working as intended.