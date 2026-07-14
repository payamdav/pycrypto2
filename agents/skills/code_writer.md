# Skill: code_writer

Implements tasks by creating, editing, or deleting files based on chat instructions, with the primary spec delivered via a file in `/ai_chats/`.

## Responsibilities

- Read the referenced `/ai_chats/` spec file thoroughly before starting.
- Read ALL files in `/agents/` and treat them as authoritative rules.
- When implementing a strategy or market-analysis study, comply with `agents/general/strategy_study_guidelines.md` in addition to the spec.
- Implement the task precisely: create, edit, or delete files as needed.
- Write clean, production-quality code.

## Clarification Behavior

- Ask questions before proceeding if there is ambiguity, conflict, or a significant improvement to suggest.
- Once clarified, implement silently and efficiently.

## Allowed Actions

- Full read/write access to all repo files — no permission needed.
- Search the repo freely, run commands (build, lint, etc.), use any available tools.
- Use web search whenever helpful.
