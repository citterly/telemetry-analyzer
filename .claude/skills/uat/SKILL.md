---
name: uat
description: "Perform User Acceptance Testing by tracing code paths from the user's perspective. Simulates button clicks, form submissions, and page navigation through the full stack to find broken wiring, missing error handling, and UX issues."
disable-model-invocation: true
argument-hint: "[page-name | flow-name | all | list]"
---

# User Acceptance Testing Skill

## Usage
`/uat <target>`

Where `<target>` is one of:
- A **page name**: `analysis`, `gg-diagram`, `corner-analysis`, `sessions`, `session-import`, `vehicles`, `dashboard`, `parquet`, `queue`, `files`, `upload`
- A **user flow**: `session-selection`, `audit-mode`, `vehicle-switching`, `lap-comparison`, `import-flow`
- `all` — run UAT on **every** target in parallel, then produce a combined summary
- `list` — show all available targets with brief descriptions

The argument is available as `$ARGUMENTS`.

## Instructions

You are performing a User Acceptance Test by tracing through code from the user's perspective. Your job is to simulate what happens when a real user interacts with the software — following every button click, dropdown selection, form submission, and page navigation through the full code path.

### Phase 1: Identify the Target

Parse `$ARGUMENTS` to determine what to test.

If `$ARGUMENTS` is `list` or empty, output this table and stop:

| Target | Type | Description |
|--------|------|-------------|
| `analysis` | Page | Main analysis dashboard — run shifts/laps/gears/power |
| `gg-diagram` | Page | G-G friction circle visualization |
| `corner-analysis` | Page | Turn-by-turn corner metrics |
| `sessions` | Page | Session browser with filters |
| `session-import` | Page | Import wizard for new sessions |
| `session-detail` | Page | Individual session view |
| `vehicles` | Page | Vehicle config editor |
| `dashboard` | Page | Home page with stats |
| `parquet` | Page | Data viewer with pagination |
| `queue` | Page | Extraction job queue |
| `files` | Page | File browser |
| `upload` | Page | XRK file upload |
| `track-map` | Page | GPS track visualization |
| `session-selection` | Flow | Selecting a session and context persistence across pages |
| `audit-mode` | Flow | Enabling audit mode and trace propagation |
| `vehicle-switching` | Flow | Changing active vehicle and re-running analysis |
| `lap-comparison` | Flow | Selecting two laps and viewing delta |
| `import-flow` | Flow | Full upload → import → analyze workflow |
| `all` | Meta | Run every target above in parallel |

If `$ARGUMENTS` is `all`, execute the **Parallel UAT** procedure below, then stop (do NOT continue to Phase 2).

#### Parallel UAT Procedure

1. **Launch subagents in parallel** using the Task tool with `subagent_type: "general-purpose"`. Launch ALL of the following targets as separate background agents in a **single message** (this is critical — they must all be launched at once for true parallelism):

   **Pages (13):** `analysis`, `gg-diagram`, `corner-analysis`, `sessions`, `session-import`, `session-detail`, `vehicles`, `dashboard`, `parquet`, `queue`, `files`, `upload`, `track-map`

   **Flows (5):** `session-selection`, `audit-mode`, `vehicle-switching`, `lap-comparison`, `import-flow`

   Each agent's prompt should be:
   ```
   You are performing a User Acceptance Test on the "{target}" {page|flow} of the telemetry-analyzer project at /home/chris/projects/telemetry-analyzer.

   Follow the full UAT procedure from .claude/skills/uat/SKILL.md — execute Phase 2 through Phase 5 for the target "{target}". Read the SKILL.md file first to understand the full procedure.

   IMPORTANT:
   - Do NOT execute Phase 6 (bug recording to features.json) — the coordinator will handle that.
   - Do NOT skip any files — read every file needed to trace the full code path.
   - At the end, output ONLY the Phase 5 report (the UAT Report section).
   - End your report with a machine-readable summary line in this exact format:
     UAT_SUMMARY|{target}|interactions:{N}|working:{N}|broken:{N}|ux_concerns:{N}|edge_cases:{N}
   - For each Critical or Major bug, also output a line in this format:
     UAT_BUG|{target}|{severity}|{short_title}|{files_csv}|{description}
   ```

   Use `run_in_background: true` for all agents. Name each agent `uat-{target}`.

2. **Wait for all agents to complete.** Use the Read tool to check each agent's output file. Poll periodically until all are done.

3. **Collect results.** Parse the `UAT_SUMMARY` and `UAT_BUG` lines from each agent's output.

4. **Execute Phase 6 once.** Read `features.json`, collect all Critical and Major bugs from all agents, deduplicate, assign sequential IDs (`bug-{target}-NNN`), and write them all in a single update to `features.json`.

5. **Output combined report.** Structure it as:

---

## UAT Full Sweep Report

**Tested**: {date}
**Targets tested**: 18 (13 pages + 5 flows)

### Aggregate Summary
| Category | Total |
|----------|-------|
| Interactions traced | {sum} |
| Working correctly | {sum} |
| Broken | {sum} |
| UX concerns | {sum} |
| Edge cases flagged | {sum} |

### Per-Target Results
| Target | Type | Interactions | Working | Broken | UX | Edge |
|--------|------|-------------|---------|--------|-----|------|
| `analysis` | Page | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

### All Critical/Major Bugs
{Numbered list of all bugs recorded to features.json, grouped by severity}

### Cross-Target Patterns
{Any patterns that appear across multiple targets — e.g., "5 pages have no loading spinner", "3 pages don't handle missing session context"}

---

Otherwise, proceed to Phase 2.

### Phase 2: Read All Relevant Code

This is the most important phase. You must read the actual source files — never guess at what they contain.

For a **page target**, read these files in order:
1. The template file: `templates/{target}.html` (map target name to actual filename)
2. The base template: `templates/base.html`
3. Any component templates (check for `{% include %}` directives in the template)
4. Any page-specific JS files referenced via `<script>` tags or `{% block %}` content
5. Shared JS modules: `static/js/audit.js`, `static/js/session_selector.js`
6. Shared CSS: `static/css/audit.css` (if audit panel is present)
7. The router(s) that serve the page and handle its API calls — find them in `src/main/routers/`
8. Any services or analyzers invoked by those router handlers — find them in `src/services/` and `src/features/`
9. Shared dependencies: `src/main/deps.py`

For a **flow target**, read:
1. All templates involved in the flow (may span multiple pages)
2. All JS files involved
3. All API endpoints touched during the flow
4. All backend services/analyzers invoked

**Do NOT skip files. Read everything needed to trace the full path. If you are unsure which router handles an endpoint, grep for the URL pattern.**

### Phase 3: Enumerate All User Interactions

List every interactive element on the page or in the flow:

- **Buttons**: what text they show, what `onclick` or event listener they have
- **Links**: `<a>` tags, where they navigate
- **Dropdowns / selects**: what options they contain, what `onchange` handler fires
- **Form fields**: input types, any validation (`required`, `pattern`, JS validation)
- **Toggles / checkboxes**: what state they control
- **Clickable table rows or cards**: click handlers on dynamic content
- **Modals / dialogs**: what triggers them, what actions they contain
- **Page load behavior**: what runs on `DOMContentLoaded` or `$(document).ready`
- **Auto-refresh / polling**: any `setInterval` or periodic fetches

For each element, note:
- **What the user sees** (button text, placeholder, label)
- **What they expect to happen** (common sense UX expectation)

### Phase 4: Trace Each Interaction

For every interaction identified in Phase 3, trace the complete code path:

```
USER ACTION: {what they click/type/select}
  → Frontend: {JS event handler, function name, file:line}
  → API Call: {method} {URL} with {params/body}
  → Backend: {router file:line} → {service/analyzer call}
  → Response: {JSON shape, key fields}
  → Rendering: {how JS processes response and updates DOM}
  → Error Path: {what happens on network error, 400, 404, 500}
```

At each step, check for these categories of issues:

#### Functional Issues (BROKEN)
- Event handler references a function that doesn't exist or is misspelled
- JS calls an API URL that doesn't match any route definition
- Frontend expects `response.results` but backend returns `response.analysis`
- API requires a parameter that the frontend doesn't send
- Conditional checks wrong property name (`if (data.laps)` vs `data.lap_data`)
- Async operations without proper sequencing (race conditions)
- Template references a variable that the route doesn't pass in context
- Route returns redirect but JS expects JSON (or vice versa)

#### UX Concerns (NEEDS ATTENTION)
- No loading spinner or indicator during API calls
- Silent failures (API error with no user-visible feedback)
- Confusing empty states (blank area vs "No data available" message)
- State lost on navigation (selected session forgotten)
- Inconsistent patterns (different error handling on different pages)
- Missing confirmation for destructive actions (delete without "Are you sure?")
- No visual feedback after successful save/submit
- Disabled buttons without explanation of why they're disabled
- Form submittable with invalid/incomplete data

#### Edge Cases
- Zero items: 0 laps, 0 sessions, no GPS data, empty parquet
- Missing data: parquet file with missing columns the analyzer expects
- Invalid state: no vehicle selected, no session in context
- Server unavailable: what does the UI show?
- Double-click: what if the user triggers the same action twice rapidly?
- Stale state: data changed on server between page load and action

### Phase 5: Report Findings

Structure the output as follows:

---

## UAT Report: `{target}`

**Tested**: {date}
**Files examined**: {list of files read}

### Summary
| Category | Count |
|----------|-------|
| Interactions traced | N |
| Working correctly | N |
| Broken | N |
| UX concerns | N |
| Edge cases flagged | N |

### Page Load Sequence
Step-by-step description of what happens when the page first loads, with file:line references.

### Interaction Trace Results

Group by functional area. For each interaction:

#### {N}. {Interaction Name} — {WORKS / BROKEN / UX CONCERN}

**User action**: {plain English description}
**Code path**: `template.html:L##` → `script block/file:functionName()` → `{METHOD} /api/endpoint` → `router.py:handler():L##` → response
**Expected**: {what should happen}
**Actual**: {what the code actually does}
**Issue** (if any): {specific description with file:line references}
**Severity**: Critical / Major / Minor / Cosmetic
**Suggested fix**: {brief, actionable description}

### Cross-Cutting Concerns
- **Error handling**: Is it consistent across all interactions? Are errors shown to the user?
- **Loading states**: Does the user know when something is in progress?
- **State management**: Does context persist correctly across interactions?
- **Data freshness**: Can stale data cause problems?

### Verification Checklist

Generate a numbered checklist the user can walk through manually in their browser:

```
[ ] 1. Navigate to {URL}, verify {expected initial state}
[ ] 2. Click "{button text}", verify {expected result}
[ ] 3. With no session selected, click "{button}", verify {graceful handling}
[ ] 4. Open browser dev tools Network tab, trigger {action}, verify {API call details}
...
```

Number each item. Include both happy-path and error-path checks. Order them so the user can walk through sequentially without backtracking.

---

### Phase 6: Record Bug Features

After producing the UAT report, record all **Critical** and **Major** severity bugs as feature entries in `features.json`. Minor and Cosmetic findings stay in the report only — they do not get feature entries.

For each Critical or Major finding from Phase 5:

1. **Read** the current `features.json`
2. **Check for existing** `bug-{target}-*` entries to avoid duplicates and to determine the next available NNN sequence number (e.g., if `bug-analysis-001` and `bug-analysis-002` exist, the next is `bug-analysis-003`)
3. **Build a feature entry** with this exact schema:

```json
{
  "id": "bug-{target}-{NNN}",
  "name": "{Short bug title from the finding}",
  "category": "bugfix",
  "description": "{Full finding description including code path, expected vs actual behavior, and suggested fix}",
  "status": "planned",
  "priority": 1,
  "tests": [],
  "files": ["{affected_file_1.py}", "{affected_file_2.html}"],
  "notes": "Found during UAT of {target}. Severity: {Critical|Major}",
  "source_uat": "{target}"
}
```

Priority rules:
- **Critical** bugs get `"priority": 1`
- **Major** bugs get `"priority": 10`

4. **Add** the new bug entries to the `features` array in `features.json`
5. **Update metadata counts**: increment `total_features` and `planned` by the number of bugs added
6. **Write** the updated `features.json`
7. **Output a summary** after the UAT report:

```
Recorded N bugs to features.json: bug-{target}-001, bug-{target}-002, ...
```

If no Critical or Major bugs were found, output:

```
No Critical/Major bugs found — nothing to record in features.json.
```

**Important**: If a finding duplicates an existing `bug-{target}-*` entry (same code path and same root cause), skip it and note the duplicate in the summary.

---

### Quality Standards

- **Be specific.** Include file paths and line numbers for every code reference. "Button doesn't work" is useless. "`Run Analysis` button at `analysis.html:142` calls `runAnalysis()` which builds URL with `selectedFile` but `selectedFile` is null when no file is selected, causing a fetch to `/api/analyze/report/null` which returns 404" is useful.
- **Follow actual code, not assumptions.** Read the function bodies. If there's a try/catch, report what the catch block actually does. If there's no try/catch, flag the missing error handling.
- **Don't fabricate issues.** Only report problems you can demonstrate exist in the code you read. If you can't tell whether something works without running it (CSS layout, animation timing), say "Unable to verify without runtime testing" and move on.
- **Prioritize.** Report broken functionality first, then UX concerns, then edge cases.
- **Be honest about confidence.** Mark findings as "Confirmed" (clear from code), "Likely" (strong evidence but needs runtime check), or "Possible" (edge case that may or may not trigger).
- **Do NOT fix bugs inline.** UAT is a read-only audit. Record Critical/Major bugs to `features.json` via Phase 6 and move on. The actual fix happens later when the boot-up ritual picks up the bug feature through the normal Plan → Execute → Test pipeline.
