# Agent Access & Permissions

## Full Project Access

Every AI agent has full, unconditional access to all local files in this project. No permission is required before editing files, creating files, deleting files, or performing any other file system operation within the project.

Agents are permitted and expected to freely use any available tools and to perform any operation, including but not limited to:

- Reading, writing, editing, and deleting any project files
- Git operations (commit, branch, merge, rebase, reset, etc.)
- GitHub operations (creating/closing issues, pull requests, comments, etc.)
- Internet search and web fetching
- Refactoring code across any files or packages
- Running scripts and commands

Do not ask for permission before taking any of these actions. Just do the work.

---

## Testing

Unless testing is **explicitly mentioned** in the task instructions, agents are **not** responsible for writing or running tests. Agents must ensure the quality and correctness of the code they produce, but executing test suites is not a default duty.

---

## Debugging

Agents must **not** perform debugging unless they are **explicitly asked to do so**. If code has a bug and debugging was not requested, note the issue but do not investigate or attempt to fix it unless the task says to.
