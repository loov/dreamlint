package config

pass: baseline: {
	description: "General logical correctness"
	inline_prompt: _prompt
}

let _prompt = """
Check whether this Go function correctly implements what it claims to do.
{{template "function-context" .}}
{{- template "summary-context" .}}

Flag any of:

Intent vs. implementation mismatch:
- Behavior contradicts the declared purpose, invariants, or documented error conditions
- Code path returns success when it should error, or vice versa
- Precondition from the summary that the code does not enforce

Logic errors:
- Inverted condition (== vs !=, < vs >)
- Wrong logical operator (&& vs ||)
- Off-by-one in loop bounds, indices, or size calculations
- Unreachable code after unconditional return, panic, or continue
- Switch/if-else that silently omits a case that must be handled
- Unguarded integer overflow or underflow

Control flow:
- Loop that cannot terminate under reachable inputs
- Early return that skips required cleanup or postcondition
- Error path that falls through to success-path code

{{- template "issues-format" .}}
"""
