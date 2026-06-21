# Skill: code_review

Reviews code changes for correctness, completeness, and full compliance with `/agents/` rules and any `/ai_chats/` specification.

## Responsibilities

- Read ALL files in `/agents/` — treat them as authoritative rules.
- Read the related `/ai_chats/` spec (if present) and verify the implementation matches it.
- Answer any questions posed in the session context.
- Review only the files changed in the task — not the entire codebase.

## Review Checklist

- All spec requirements implemented?
- All `/agents/` rules followed?
- Free of bugs, logic errors, and edge-case failures?
- Clean, readable, well-structured code?
- Security or performance concerns?
- All session questions answered?

## Output Format

- Summary of findings.
- Issues listed with file and line references.
- Explicit compliance/non-compliance statement per applicable rule.
- Answers to any session questions.

## Allowed Actions

- Full read access to all repo files, free repo search, run commands, use web search.
