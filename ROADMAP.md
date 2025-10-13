# ACE System Development Roadmap

This document outlines the development plan for the Agentic Context Engineering (ACE) system, with a renewed focus on building concrete, functional, and testable components.

## Core Problem
The ACE system, as initially conceived, was overly ambitious and lacked a foundation of working code. The immediate problem is to pivot from aspirational whitepapers to practical engineering by building a single, useful component from the ground up.

## Phase 1: Multi-Tier Context Caching (DCR)
The first phase of development will focus on implementing a functional multi-tier caching system for contexts, as proposed in the `DynamicContextRepository` (DCR) concept.

### Goals
- To build a robust and efficient caching system with four tiers (L1-L4).
- To implement automatic tiering of contexts based on usage patterns.
- To write comprehensive unit tests to ensure correctness and reliability.
- To provide clear and honest documentation of the implemented system.

### Development Plan & Checklist

- [ ] **1. Implement Core DCR Logic**
  - [ ] Refactor the `DynamicContextRepository` class in `src/ace/core/implementation.py`.
  - [ ] Implement in-memory storage for L1 (hot) and L2 (warm) caches.
  - [ ] Implement a simulated persistent storage for L3 (cold) and L4 (archive) tiers (e.g., JSON files).
  - [ ] Implement the `store` method with logic to place new contexts in the L1 cache.
  - [ ] Implement the `retrieve` method with logic to search through tiers (L1 -> L4).

- [ ] **2. Implement Tier Management**
  - [ ] Implement logic to promote contexts from lower to higher tiers upon access.
  - [ ] Implement a cache eviction policy for L1 and L2 caches (e.g., LRU - Least Recently Used).
  - [ ] Implement a background process or periodic task to demote contexts from higher to lower tiers based on inactivity.

- [ ] **3. Write Unit Tests**
  - [ ] Create `tests/test_dcr.py`.
  - [ ] Write tests for storing and retrieving contexts.
  - [ ] Write tests for tier promotion (e.g., accessing an L2 item should move it to L1).
  - [ ] Write tests for tier demotion (e.g., inactive L1 items should move to L2).
  - [ ] Write tests for cache eviction (e.g., adding a new item to a full L1 cache should evict the LRU item).

- [ ] **4. Refactor CLI Demo**
  - [ ] Update the `main` function in `src/ace/core/implementation.py` to be a realistic demonstration of the caching system.
  - [ ] The demo should add, retrieve, and show the state of the cache to illustrate tiering.

- [ ] **5. Update Documentation**
  - [ ] Revise `README.md` to describe the functional DCR.
  - [ ] Update `docs/ace-architecture-v2.md` to be an honest representation of the *implemented* DCR, removing fabricated performance numbers and unsupported claims.

- [ ] **6. Pre-Commit & Submission**
  - [ ] Run all tests and ensure they pass.
  - [ ] Request a code review.
  - [ ] Record learnings.
  - [ ] Submit the changes with a descriptive commit message.

## Future Phases (Post-DCR)
Once the DCR is complete and verified, the following concrete problems could be addressed:

- **Integration with a real LLM API**: Connect the ACE system to an actual LLM (e.g., OpenAI, Claude) to process contexts.
- **Basic Prompt Injection Detection**: Implement a simple token-based detection mechanism for prompt injection attacks.
- **A/B Testing Framework for Prompts**: Build a simple framework for testing the performance of different prompt variations.
