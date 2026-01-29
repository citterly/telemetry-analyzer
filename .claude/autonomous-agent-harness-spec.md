# Autonomous Agent Harness Implementation Spec

## Overview

Implement Anthropic's two-agent pattern for long-running autonomous coding, as described in their engineering blog post. This harness enables Claude to work across multiple sessions without losing progress.

## Source Material

- **Anthropic Blog Post**: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- **Quickstart Repo**: https://github.com/anthropics/claude-quickstarts/blob/main/autonomous-coding/autonomous_agent_demo.py
- **Claude Agent SDK Docs**: https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk

## The Problem This Solves

1. **One-shotting failure**: Agent tries to build everything at once, runs out of context mid-implementation
2. **Premature victory**: Agent sees partial progress and declares "done" without finishing
3. **Amnesia between sessions**: Each new session has no memory of previous work

## Architecture: Two-Agent Pattern

### Agent 1: Initializer Agent
Runs ONCE at project start. Creates the scaffolding:

```
initializer/
├── Responsibilities:
│   ├── Parse user prompt into structured feature list
│   ├── Create features.json with all features marked "failing"
│   ├── Create claude-progress.txt (empty log)
│   ├── Create init.sh (environment bootstrap script)
│   ├── Make initial git commit
│   └── Set up test harness
```

### Agent 2: Coding Agent  
Runs for EVERY subsequent session. Makes incremental progress:

```
coding_agent/
├── Boot-up ritual (MUST happen every session):
│   ├── Run pwd to confirm directory
│   ├── Read git log (recent commits)
│   ├── Read claude-progress.txt
│   ├── Read features.json
│   ├── Run init.sh to start dev environment
│   └── Run existing tests to verify clean state
│
├── Work phase:
│   ├── Pick ONE failing feature (highest priority)
│   ├── Implement it
│   ├── Write/update tests
│   └── Run end-to-end verification
│
├── Clean-up ritual (MUST happen before terminating):
│   ├── Update feature status in features.json
│   ├── Append to claude-progress.txt
│   ├── Commit to git with descriptive message
│   └── Ensure environment is in "clean state"
```

## Required Artifacts

### 1. features.json
```json
{
  "project_name": "example-app",
  "features": [
    {
      "id": "feat-001",
      "name": "User authentication",
      "description": "Basic login/logout with session management",
      "status": "failing",
      "priority": 1,
      "tests": ["test_login", "test_logout", "test_session"],
      "notes": ""
    },
    {
      "id": "feat-002", 
      "name": "Dashboard view",
      "description": "Main dashboard showing user stats",
      "status": "failing",
      "priority": 2,
      "tests": ["test_dashboard_loads", "test_stats_display"],
      "notes": ""
    }
  ],
  "metadata": {
    "created_at": "2025-01-29T00:00:00Z",
    "last_updated": "2025-01-29T00:00:00Z",
    "total_features": 2,
    "passing": 0,
    "failing": 2
  }
}
```

### 2. claude-progress.txt
```
=== Session 1 (2025-01-29 10:00) ===
Initializer agent run.
- Created project structure
- Set up features.json with 5 features
- Created init.sh bootstrap script
- Initial commit: abc123

=== Session 2 (2025-01-29 10:30) ===
Working on: feat-001 (User authentication)
- Implemented login endpoint
- Added session middleware
- Tests passing: 3/3
- Status: feat-001 marked PASSING
- Commit: def456

=== Session 3 (2025-01-29 11:00) ===
Working on: feat-002 (Dashboard view)
- Created dashboard component
- Connected to user stats API
- Tests passing: 2/2
- Status: feat-002 marked PASSING
- Commit: ghi789
```

### 3. init.sh
```bash
#!/bin/bash
# Bootstrap script for development environment

# Install dependencies
npm install 2>/dev/null || pip install -r requirements.txt 2>/dev/null

# Start development server (background)
npm run dev &>/dev/null &
# OR: python manage.py runserver &>/dev/null &

# Wait for server to be ready
sleep 3

# Run smoke test
curl -s http://localhost:3000/health || echo "Warning: Health check failed"

echo "Environment ready"
```

## Implementation Steps for Claude Code

### Step 1: Clone/Review Reference Implementation
```bash
git clone https://github.com/anthropics/claude-quickstarts.git
cd claude-quickstarts/autonomous-coding
# Review autonomous_agent_demo.py
```

### Step 2: Create Project Structure
```
autonomous-harness/
├── README.md
├── harness/
│   ├── __init__.py
│   ├── initializer.py      # Initializer agent logic
│   ├── coding_agent.py     # Coding agent logic
│   ├── artifacts.py        # Feature list, progress log management
│   └── prompts/
│       ├── initializer_prompt.md
│       └── coding_agent_prompt.md
├── templates/
│   ├── features.json       # Template for feature tracking
│   ├── claude-progress.txt # Template for progress log
│   └── init.sh             # Template bootstrap script
├── examples/
│   └── simple_webapp/      # Example project to test harness
└── run.py                  # Main entry point
```

### Step 3: Implement Core Components

#### initializer.py
- Takes user prompt as input
- Expands prompt into detailed feature list
- Creates all scaffolding artifacts
- Makes initial git commit
- Returns control to user

#### coding_agent.py
- Boot-up ritual (read all artifacts)
- Feature selection logic (priority-based)
- Work loop with verification
- Clean-up ritual (update artifacts, commit)
- Session termination

#### artifacts.py
- `FeatureList` class: CRUD operations on features.json
- `ProgressLog` class: Append-only log management
- `GitManager` class: Commit, log reading

### Step 4: Create Prompt Templates

#### initializer_prompt.md
```markdown
You are an initializer agent. Your job is to set up the scaffolding for a coding project.

Given the user's request, you must:
1. Break it down into discrete, testable features
2. Create a features.json file with all features marked "failing"
3. Create a claude-progress.txt file
4. Create an init.sh script to bootstrap the environment
5. Set up a basic test harness
6. Make an initial git commit

Do NOT implement any features. Only create the scaffolding.

User request: {user_prompt}
```

#### coding_agent_prompt.md
```markdown
You are a coding agent. You work in sessions, making incremental progress.

BOOT-UP RITUAL (do this first, every time):
1. Run `pwd` to confirm your location
2. Run `git log --oneline -5` to see recent work
3. Read claude-progress.txt to understand history
4. Read features.json to see what's done vs. pending
5. Run `./init.sh` to start the environment
6. Run tests to verify clean state

WORK PHASE:
1. Pick ONE failing feature (highest priority)
2. Implement it completely
3. Write or update tests
4. Verify with end-to-end testing

CLEAN-UP RITUAL (do this before stopping):
1. Update features.json (mark feature passing/failing)
2. Append session summary to claude-progress.txt
3. Commit with descriptive message
4. Ensure no broken code remains

Current feature list:
{features_json}

Recent progress:
{progress_log}
```

## Key Design Principles

1. **Externalize goals**: Turn "do X" into machine-readable backlog with pass/fail criteria
2. **Make progress atomic**: One feature per session, always testable
3. **Leave campsite clean**: Every session ends in a mergeable state
4. **Standardize boot-up**: Same ritual every time (read memory, run checks, then act)
5. **Tests are source of truth**: Feature status tied to test results, not agent judgment

## Testing the Harness

1. Create a simple test project (e.g., "build a todo app")
2. Run initializer agent
3. Verify artifacts created correctly
4. Run coding agent 3-4 times
5. Verify incremental progress in git log
6. Verify features.json updates correctly
7. Verify claude-progress.txt captures session summaries

## Optional Enhancements

- **Puppeteer MCP integration**: For end-to-end browser testing
- **Rollback capability**: If tests fail, revert to last good commit
- **Priority recalculation**: Adjust feature priority based on dependencies
- **Parallel workers**: Multiple coding agents on independent features (advanced)

## Success Criteria

- [ ] Initializer creates all required artifacts from a single prompt
- [ ] Coding agent correctly reads and updates all artifacts
- [ ] Each session makes exactly ONE feature's worth of progress
- [ ] Git history shows clean, incremental commits
- [ ] Progress log accurately reflects work done
- [ ] Harness can run 5+ sessions without human intervention
- [ ] Failed sessions don't corrupt project state

---

## Notes for Claude Code

This spec follows the exact pattern it's describing. You have:
- Clear goals (the success criteria)
- Structured memory (this document)
- Atomic tasks (the implementation steps)
- Test criteria (success criteria checklist)

Treat this spec as your "initializer" - it sets the stage. Now be the coding agent and implement it incrementally.
