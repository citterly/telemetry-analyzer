# 🤝 Rules of Engagement — Telemetry Analyzer

This document defines how we work together on the Telemetry Analyzer project.  
It ensures consistent process, clear expectations, and efficient collaboration.

---

## 1. Start with a Work Package (WP)
- Every task begins with a **Work Package (WP)** definition.
- A WP must include:
  - **Scope**: What we’re trying to achieve.
  - **Required Files**: Pulled from `docs/inventory.md` (respecting ≤25 file limit).
  - **Artifacts**: What’s expected at the end (tests, docs, outputs).
  - **Acceptance Criteria**: Clear, testable definition of “done.”

---

## 2. File Handling
- Upload only the **minimum file set** needed for the WP.
- If more context is needed, I will flag specific missing files.
- After finishing a WP:
  - Merge code/doc changes into the repo.
  - Update `docs/inventory.md` if new files were added or descriptions changed.
  - **Update governance docs** (`roadmap.md`, `phases.md`, `overview.md`) to reflect WP closure and next steps.

---

## 3. Workflow
1. **Plan →** Define WP (scope, files, criteria).
2. **Prep →** Confirm required files using inventory.
3. **Work →** Iterative development & review.
4. **Deliver →** Output artifacts (tests, reports, code changes).
5. **Close →** Commit to repo, update inventory, mark WP complete, update roadmap + phases + overview.

---

## 4. Artifacts at Each Phase
- **Foundation:** smoke tests, DLL check logs.
- **Data Access:** metadata JSON, export validation.
- **Analysis:** lap/gear reports, CSV/plots.
- **Integration:** pipeline JSON outputs.
- **UI:** screenshots/templates, API responses.
- **Docs:** updated roadmap + inventory.

---

## 5. Communication Rules
- If scope creeps, stop and reframe into a new WP.
- Always confirm the **file manifest** before starting work.
- Keep check-ins short and clear: “✅ ready” vs. “⚠️ need X.”
- I will flag:
  - Missing dependencies.
  - Files that don’t belong (e.g., sample data).
  - Opportunities for splitting into smaller WPs.
  - **Deprecated modules/folders** — must be labeled as “deprecated” in `inventory.md` until removed or migrated.

---

## 6. Success Metrics
- Each WP deliverable runs without error.
- Tests confirm expected behavior.
- Inventory stays current.
- Roadmap reflects actual progress.

---

## 7. CI/CD & Automation
- CI runs `pytest` + smoke tests on each commit.
- Pre-commit hook checks inventory consistency.
- Versioning follows `v0.x` for increments, `v1.0` for first stable release.


---

## 8. Standard Prompts
For consistency, use the standard prompts defined in `docs/prompts.md`.  
These include kickoff, closeout, backlog capture, refactor/review, testing, and deployment prompts.  
