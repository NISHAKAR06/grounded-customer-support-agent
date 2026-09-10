# Contributing Guidelines

## 1. Engineering Standards

To preserve the rigorous academic and engineering standards of the **Grounded Customer Support Agent**, all contributions must comply with the following standards:

### Code Quality & OOP Principles
- **Type Annotations**: All function parameters, return values, and class attributes must include Python type hints.
- **Pydantic Validation**: All API schemas, configuration settings, and pipeline payloads must be validated using Pydantic v2 models.
- **Single Responsibility**: Every class and module must have a clear, isolated responsibility. Business logic must never reside within FastAPI route handlers.
- **Structured Logging**: Use the application logger (`app.core.logging`) with contextual metadata (such as `run_id`).
- **Clean Error Handling**: Raise domain exceptions defined in `app.core.exceptions` instead of catching generic `Exception`.

---

## 2. Commit Permission Rule

Before any Git commit is executed:
1. Review `git status` and `git diff` to ensure no sensitive files (`.env`, credentials, raw datasets) are staged.
2. Verify that all automated tests in `tests/` pass with zero failures.
3. Verify that code formatting complies with Ruff and Black.
4. Prepare a conventional commit message following the format: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
5. Seek explicit authorization before running `git commit`.

---

## 3. Testing Requirements

- Any new service method must be accompanied by unit tests under `tests/unit/`.
- Any new API endpoint must be accompanied by integration tests under `tests/api/`.
- Tests must be deterministic and fast, using fixtures located in `tests/fixtures/`.
