package config

pass: summary: {
	description: "Generate a summary of the code"
	inline_prompt: _prompt
}

let _prompt = """
Summarize this Go function's behavior.
{{template "function-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Respond with JSON:
{
  "purpose": "One sentence describing what this function does",
  "behavior": "How it behaves - side effects, return conditions, error cases",
  "invariants": ["preconditions", "postconditions", "guarantees"],
  "security": ["security-relevant properties for callers to know"]
}

- purpose: what the function does from the caller's perspective, not how. Do not restate the function name.
- behavior: edge cases, side effects, non-obvious return values, error conditions. Do not repeat the happy path from purpose.
- invariants: preconditions callers must satisfy, postconditions guaranteed on success. Empty array if none.
- security: whether inputs are validated, auth is enforced, sensitive data is handled. Empty array if not applicable.
"""
