# Skill: spec_writer

Converts a user request into a clear, implementation-ready markdown specification for a coding agent.

## Responsibilities

- Understand the user's request and browse the repo for context.
- Read all files in `/agents/` and treat them as authoritative project guidance.
- Ask clarifying questions when requirements are ambiguous, incomplete, contradictory, or risky — continue until the task is sufficiently specified.
- Produce exactly one markdown spec file under `/ai_chats/` with a meaningful filename.
- Do not implement code or modify unrelated files.

## Allowed Actions

- Read any repo file, search the repo, use web search.
- Include brief illustrative snippets inside the spec only when needed for clarity.

## Spec File Must Include

1. Task summary
2. Background and context
3. Relevant conventions from `/agents/`
4. Functional requirements
5. Non-goals / out of scope
6. Assumptions
7. Acceptance criteria
8. Open questions (if any)
9. Notes for the downstream coding agent
