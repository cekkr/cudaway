# AI Reference

This document is a fast-access knowledge base for AI agents working on fayasm. Update it alongside `README.md` whenever you modify code or documentation so the next agent inherits the latest context.

## Collaboration Rules

- After every code edit, refresh both `README.md` and `AI_REFERENCE.md` with any relevant behavioural, architectural, or tooling changes.
- Log fresh research or experiments under `studies/` and cross-reference them here to avoid repeating the same investigations.
- Prefer incremental changes: keep commits small, document breaking changes, and run the available tests before yielding control.

## Build & Test Checklist

- Configure and build with CMake (>= 3.10): ...
- Invoke the provided helper: `./build.sh` (cleans `build/`, regenerates, runs tests).
- Ensure new tests land alongside new features; runtime code lacks extensive coverage, so favour regression tests around control flow and stack behaviour.

## Core Code Map

- `src/*` – Main C++ source code folder
- `studies/*` – Contains documents with studies about the project
- `src/fa_ops.*` – Basic concept on which the entire project is based

### Gaps Worth Watching

- Many opcode handlers are placeholders; when fleshing them out, add targeted tests.
- Module loading currently depends on file descriptors; embedding scenarios may require in-memory alternatives.
- Memory and trap semantics are preliminary—plan for bounds checking and better error propagation.

## Research Archive (studies/)

Use these references before re-running the same explorations:

(Insert here most studied files by you during researches after prompts)

Keep this index synchronized when new material lands in `studies/`.

## When Starting a New Task

- Skim outstanding TODOs in source files (search for `TODO` or `FIXME`).
- Validate whether relevant studies already cover the topic; if not, add a new entry both under `studies/` and above.
- Outline expected tests; if the suite lacks coverage, note the gap here so the next agent can prioritise it.

## Contact & Credits

- Project owner: Riccardo Cecchini (Mozilla Public License Version 2.0, 2025).

If any workflow rule changes, reflect it here immediately so human collaborators and AI agents remain aligned.