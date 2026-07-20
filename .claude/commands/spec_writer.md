---
description: Run the spec_writer role — turn a task description into an implementation-ready spec in ai_chats/
argument-hint: task description, or path to a task file
---

Act as the **spec_writer** agent defined in `agents/roles/spec_writer.md` and follow it exactly, including its capability limits. This role runs here in the main conversation because it must be able to ask the user clarifying questions.

Read `AGENTS.md` (repo root) and ALL files under `agents/` first, then browse the repo for context relevant to the task. Ask the user clarifying questions whenever requirements are ambiguous, incomplete, contradictory, or risky. Produce exactly one markdown spec file under `ai_chats/` with a meaningful filename; change nothing else.

Task: $ARGUMENTS
