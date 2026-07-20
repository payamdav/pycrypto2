# Agent Instructions

Canonical entrypoint for every AI agent and coding tool working in this repository (Claude Code, GitHub Copilot, or any other).

**Mandate:** read all files in the `agents/` directory hierarchy before performing any task. They are the authoritative rules, conventions, and how-to guides governing all work here; skipping them leads to incorrect or non-compliant work.

## Directory Index

| Path | Content |
|---|---|
| `agents/general/` | Repository-wide rules: access & permissions, dependency/doc rules, file placement, study framework |
| `agents/roles/` | Tool-agnostic agent role definitions (see below) |
| `agents/packages/` | Authoritative documentation for every package in `packages/` |
| `agents/datasets/` | Dataset schemas, identities, and access methods |
| `agents/ideas/` | Reusable algorithm and computation-pattern specifications |

## Agent Roles

`agents/roles/` holds the tool-agnostic agent definitions — the canonical source for what each agent does, how it runs, and what it may touch:

- `spec_writer` — turns a request into one implementation-ready spec under `ai_chats/` (interactive: asks clarifying questions).
- `code_writer` — implements a spec from `ai_chats/` (standalone; delegable to a cheaper/faster model).
- `code_review` — reviews changes against the spec and `agents/` rules (standalone, read-only).
- `code_tester` — builds visual test notebooks under `notebooks/tests/` (standalone).

Typical chain: `spec_writer` → spec file in `ai_chats/` → `code_writer` → `code_review` / `code_tester`. The committed spec file is the handoff artifact between stages.

## Tool Adapters

Roles are wired into specific tools by thin adapters that reference — never duplicate — the core files in `agents/roles/`:

- **Claude Code:** `.claude/agents/` (subagents: `code_writer`, `code_review`, `code_tester`), `.claude/commands/spec_writer.md` (interactive `/spec_writer` command), and `.claude/commands/spec_impl_review.md` (`/spec_impl_review` — full pipeline: spec → user approval → implement → review → fix loop).
- **GitHub Copilot:** `.github/agents/` (custom agents for all four roles).

To adopt a new tool, add a new thin adapter pointing at `agents/roles/`; never copy role text into an adapter.
