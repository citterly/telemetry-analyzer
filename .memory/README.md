# .memory - Cross-Machine Development Context

This folder maintains state and instructions for development across multiple machines (Linux dev, Windows test).

## How It Works

1. **NEXT_TASK.md** - Instructions for the receiving machine
2. **RESULT.md** - Results/findings to send back
3. Other files as needed for context

## Workflow

```
Machine A                          Machine B
─────────                          ─────────
Write NEXT_TASK.md
git push                    →      git pull
                                   Read NEXT_TASK.md
                                   Do the work
                                   Write RESULT.md
                                   git push
git pull                    ←
Read RESULT.md
Continue development
```

## Current Context

- **Linux:** Main development, code changes, analysis
- **Windows:** Required for AIM DLL testing (XRK extraction)
