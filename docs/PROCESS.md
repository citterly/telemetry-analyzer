# Development Process

## Three-Phase Pipeline

Every non-trivial feature follows a **Plan → Execute → Test** pipeline. Each phase is a separate entry in `features.json` with explicit dependencies.

### Phase 1: Plan (`-plan`)

**Goal**: Produce a design doc before writing any code.

**Inputs**: Codebase exploration, existing patterns, requirements from features.json description.

**Activities**:
1. Read all files that will be affected
2. Identify existing patterns to follow or break
3. Write a design doc in `docs/architecture/{feature-id}.md` with:
   - **Problem statement**: What's wrong / what's missing
   - **Current state**: How things work today (with file paths and line numbers)
   - **Proposed changes**: Exact files, classes, methods to add/modify/remove
   - **Acceptance criteria**: Measurable conditions that define "done"
   - **Migration path**: How to transition without breaking existing functionality
   - **Risks**: What could go wrong, what's the rollback plan
4. Commit the design doc

**Output**: Design doc committed. Feature status → passing.

**Rules**:
- NO code changes in plan phase (only the design doc)
- Design doc must reference specific files and line numbers
- Acceptance criteria must be testable

---

### Phase 2: Execute (`-exec`)

**Goal**: Implement the changes described in the design doc.

**Inputs**: Design doc from plan phase. Blocked until plan phase is "passing".

**Activities**:
1. Re-read the design doc
2. Implement changes following the spec exactly
3. Run existing tests to verify no regressions
4. Update `features.json` and `claude-progress.txt`
5. Commit

**Output**: Code changes committed. Feature status → passing.

**Rules**:
- Follow the design doc — deviations must be noted in the commit message
- Run full test suite before marking passing
- If the design doc was wrong, update it in the same commit with a note

---

### Phase 3: Test (`-test`)

**Goal**: Verify the implementation matches the acceptance criteria from the design doc.

**Inputs**: Design doc + implementation. Blocked until exec phase is "passing".

**Activities**:
1. Re-read the acceptance criteria from the design doc
2. Write tests that verify each criterion
3. Run tests, fix any failures
4. Write a compliance summary as a comment in the test file
5. Commit

**Output**: Tests committed. Feature status → passing.

**Rules**:
- Every acceptance criterion must have at least one test
- Tests should be independent of implementation details where possible
- Include both positive (it works) and negative (it rejects bad input) cases

---

## Feature Naming Convention

```
{category}-{number}-{phase}
```

Examples:
- `arch-001-plan` → Plan phase for vehicle config unification
- `arch-001-exec` → Execute phase
- `arch-001-test` → Test/verify phase

Categories:
- `arch` — Architecture improvements
- `feat` — New functionality
- `safeguard` — Traceability and validation
- `cleanup` — Code quality

---

## Dependency Rules

```
arch-001-plan  ←  arch-001-exec  ←  arch-001-test
     (must pass)       (must pass)
```

- Exec is blocked by its own plan
- Test is blocked by its own exec
- Cross-feature dependencies are declared in features.json `notes` field
  (e.g., "Depends on arch-001-exec for unified vehicle config")

---

## Boot-up Ritual Integration

The boot-up ritual in CLAUDE.md picks work in this order:

1. **Failing features** — fix regressions first
2. **Planned features, unblocked** — highest priority first
3. **All passing** — stop and report

A feature is "unblocked" when all features it depends on are "passing".

The dependency chain is:
- `-plan` features have no blockers (always eligible)
- `-exec` features are blocked until their `-plan` is passing
- `-test` features are blocked until their `-exec` is passing

---

## Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Feature registry | `features.json` | Single source of truth for all work items |
| Progress log | `claude-progress.txt` | Session-by-session work history |
| Design docs | `docs/architecture/*.md` | Specs for each feature |
| Safeguard spec | `docs/SAFEGUARD_SYSTEM.md` | Traceability system design |
| Process doc | `docs/PROCESS.md` (this file) | Development process definition |
| Auto-memory | `~/.claude/.../memory/MEMORY.md` | Cross-session knowledge |
