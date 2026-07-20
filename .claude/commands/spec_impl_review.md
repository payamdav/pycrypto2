---
description: Full pipeline — spec_writer writes a spec, user approves it, code_writer implements, code_review verifies, findings get fixed
argument-hint: task description, or path to a task file
---

Orchestrate the full spec → implement → review pipeline for the task below. You run the spec stage yourself; implementation and review are delegated to the project's agents.

**Stage 1 — spec (inline).** Act as the spec_writer agent per `agents/roles/spec_writer.md`: read `AGENTS.md` (repo root) and ALL files under `agents/`, gather repo context, ask the user clarifying questions when requirements are ambiguous, and write exactly one spec file under `ai_chats/`.

**Approval gate.** Present the spec path plus a short summary (key design decisions, assumptions, open questions) and STOP — wait for the user's go-ahead. Apply any corrections they request to the spec before continuing. Never start Stage 2 without explicit approval.

**Stage 2 — implement.** Launch the `code_writer` agent, naming the spec file. It implements and verifies per its definition.

**Stage 3 — review.** Launch the `code_review` agent on the files changed in Stage 2, against the same spec.

**Stage 4 — fix loop.** If the review reports issues: pass the findings to `code_writer` to fix, then re-run `code_review` on the result. At most 2 fix rounds; if issues remain after that, stop and report them to the user instead of looping.

**Final report.** Spec path, files created/changed, verification results, review verdict, remaining issues (if any). Commit/push only when the user or session instructions ask for it.

Task: $ARGUMENTS
