# ACE System Development Roadmap

This document outlines the development plan for the Agentic Context Engineering (ACE) system, with a renewed focus on building concrete, functional, and testable components.

## Core Problem
The ACE system, as initially conceived, was overly ambitious and lacked a foundation of working code. The immediate problem is to pivot from aspirational whitepapers to practical engineering by building one useful component at a time.

---

## Phase 1: Multi-Tier Context Caching (DCR) - ✅ COMPLETE
This phase focused on implementing a functional multi-tier caching system for contexts.

### Goals
- [x] To build a robust and efficient caching system with four tiers (L1-L4).
- [x] To implement automatic tiering of contexts based on usage patterns.
- [x] To write comprehensive unit tests to ensure correctness and reliability.
- [x] To provide clear and honest documentation of the implemented system.

---

## Phase 2: LLM Integration - ✅ COMPLETE
This phase focused on connecting the ACE system to a real Large Language Model (LLM) to enable the processing of stored contexts.

### Goals
- [x] To create a dedicated client for communicating with an LLM API (e.g., OpenAI).
- [x] To securely manage API keys using environment variables.
- [x] To expose the LLM processing functionality via a new API endpoint.
- [x] To write unit tests for the LLM client using mocking.
- [x] To update all documentation to reflect the new capabilities.

---

## Phase 3: Basic Prompt Injection Detection
This phase focuses on implementing a functional and testable prompt injection detection mechanism.

### Goals
- [ ] To enhance the `InputSanitizationEngine` with concrete detection rules.
- [ ] To integrate the sanitization check into the API.
- [ ] To write unit tests to validate the detection logic.
- [ ] To document the new security feature.

### Development Plan & Checklist
- [ ] **1. Enhance Engine**: Implement detection logic in `InputSanitizationEngine`.
- [ ] **2. Write Unit Tests**: Create `tests/test_sanitization.py`.
- [ ] **3. Integrate with API**: Return sanitization status from the `store` endpoint.
- [ ] **4. Update Documentation**: Update `README.md` and `ROADMAP.md`.
- [ ] **5. Pre-Commit & Submission**: Run tests, get a code review, and submit.

---

## Future Phases
Once prompt injection detection is complete, the following concrete problems could be addressed:

- **A/B Testing Framework for Prompts**: Build a simple framework for testing the performance of different prompt variations.
