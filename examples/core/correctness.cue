package config

pass: correctness: {
	description: "Error handling, nil safety, resource management"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for error handling, nil safety, and resource management bugs.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Flag any of:

Error handling:
- Error assigned to _ or return value not checked
- Same error both logged and returned. Choose one: log or return, never both.
- Error wrapped without adding new context. "failed to X" restates the error. Only wrap when adding information the caller lacks.
- Error compared with == or type assertion instead of errors.Is / errors.As
- Error string not lowercase or has trailing punctuation
- Panic used for a recoverable error
- Error variable shadowed in an inner scope

Nil and bounds safety:
- Nil pointer dereference without guard
- Slice/array index without bounds check
- Map lookup assuming the key exists
- Type assertion without comma-ok

Resource management:
- File, connection, or response body not closed. defer Close() immediately after open.
- defer inside a loop (defers accumulate until the function returns)
- Resource not released on an error path
- Long operation ignoring context cancellation
- External call without a timeout. Use context.WithTimeout.

{{- template "issues-format" .}}
"""
