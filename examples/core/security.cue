package config

pass: security: {
	description: "Security audit"
	inline_prompt: _prompt
}

let _prompt = """
Audit this Go function for security vulnerabilities.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Flag any of:
- SQL injection: unparameterized queries, string concatenation in SQL. ORDER BY columns must be allowlisted.
- Command injection: exec.Command must use separate args. Never sh -c with concatenated input.
- Path traversal: file operations with user-controlled paths without sanitization.
- XSS: use html/template for HTML output. Never raw string formatting.
- SSRF: user-controlled URLs passed to HTTP requests.
- Insecure deserialization: untrusted data in unmarshal calls.
- Hardcoded secrets: API keys, passwords, tokens in source. Load from environment or secret manager.
- Weak randomness: use crypto/rand for tokens and keys. Never math/rand.
- Information disclosure: technical errors exposed to users. Send generic messages to clients, log details server-side.
- IDOR: resource accessed by caller-supplied ID without verifying the caller's permission.

{{- template "issues-format" .}}
"""
