## Continuity Ledger

Maintain `CONTINUITY.md` (repo root) as the canonical session briefing. It survives context compaction — do not rely on earlier chat unless reflected in the ledger.

### When to read
- Session start
- After compaction
- Context feels stale

### When to update
- Goal/constraints/decisions change
- Task completed or significant progress
- Before ending session

### Skip updates for
- Quick Q&A
- Mid-task work
- Minor clarifications

### Format
Keep short. Facts only, no transcripts. Bullets. Mark uncertainty as `UNCONFIRMED`.

```markdown
## Continuity Ledger
- Goal (incl. success criteria):
- Constraints/Assumptions:
- Key decisions:
- Done:
- Now:
- Next:
- Open questions (UNCONFIRMED if needed):
- Working set (files/ids/commands):
```

### In replies
Show brief "Ledger Snapshot" (Goal + Now/Next) at session start, after compaction, or when state changes. Skip for quick answers or mid-task flow.


## Task Tracking

### Beads (`bd`) — source of truth for tasks

**Beads owns:**
- Task list (epics, tasks, bugs)
- Status (open, in_progress, done)
- Dependencies and blockers
- Per-task notes

**Ledger owns:**
- Goal, constraints, key decisions
- Current state summary
- Pointers to current/next bead
- Cross-task notes

If Beads not initialized: `bd init`

### In-session tracking
Use your environment's todo tool for 3-7 step execution plans:
- Claude Code: `TodoWrite`
- Codex: inline checklist
- Other: equivalent mechanism

Do NOT put micro-steps in CONTINUITY.md.

### Session workflow
1. Read `CONTINUITY.md`
2. Check for dirty state: `git status` and `git stash list`
   - If uncommitted changes: show user, ask "Continue, commit, or stash?"
3. Run `bd ready` → work on top item
4. If nothing ready: `bd blocked` → resolve or create prerequisite tasks
5. Update bead status immediately after progress
6. Update ledger if goal/constraints/decisions change

### Task quality
Each task: small (one burst), verifiable (clear done condition), correctly linked (deps reflect reality).


## Scope Boundaries

### Ask first
- Architectural changes
- New dependencies
- Deleting files/functionality
- Changes affecting >5 files
- Deviating from request

### Proceed autonomously
- Bug fixes with clear scope
- Small features in single module
- Refactors (no behavior change)
- Test additions
- Docs for changed code

### If task grows beyond scope
Complete minimal version, create new beads for discovered work, confirm with user.


## Error Handling

- `bd` fails → install: `npm install -g @beads/bd`
- Tests fail → fix before marking done; if blocked, note in bead, ask user
- Ledger conflicts with code → trust code, update ledger
- Unclear requirements → ask user, add to Open Questions as `UNCONFIRMED`


## Session Completion

**Commit message format:**
```
type: short description

types: feat, fix, perf, refactor, test, docs, chore, wip
```

**Agent does automatically:**
1. Commit all changes (using format above)
2. Update Beads (status, new issues for remaining work)
3. Update CONTINUITY.md
4. Run quality gates (tests, lints, build)
5. `bd sync`


## GitHub Authentication

Expected: `GITHUB_TOKEN` env var set in `~/.zshrc`.

Verify: `echo $GITHUB_TOKEN | head -c 10` (should show `github_pat`)

**NEVER:**
- Commit tokens/secrets
- Echo/print/log full token values
- Store tokens in code or config

If auth fails: ask user to check `GITHUB_TOKEN` in `~/.zshrc` and run `source ~/.zshrc`.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
