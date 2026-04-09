package config

pass: maintainability: {
	description: "Complexity, naming, structure"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for maintainability issues.
{{template "function-context" .}}
{{- template "summary-context" .}}

Flag any of:

Complexity:
- Function longer than 50 lines
- Nesting deeper than 3 levels
- More than 5 parameters
- Complex boolean expression
- Multiple responsibilities in one function

Readability:
- Magic numbers without named constants
- Complex logic without explanatory comment
- Inconsistent naming conventions

API design (exported functions only):
- Too many parameters. Use functional options for constructors with optional config.
- Boolean parameters that obscure meaning at the call site
- Pass-through method whose entire body forwards to another call

Naming:
- Wrong acronym casing: userID not userId, URL not Url, ServeHTTP for exported
- Receiver named self, this, or the full type name. Use one or two letters (s for Service).
- Sentinel error not following ErrXxx convention
- Enum value not prefixed with the type name: RoleAdmin not Admin
- Constant in ALL_CAPS instead of MixedCaps: MaxRetries not MAX_RETRIES
- Name too vague to understand at the call site
- Boolean name that is not a predicate (blinkStatus instead of cursorVisible)
- Comment restating what the code already shows rather than explaining why
- Exported type, variable, or method missing a godoc comment

Use severity "low" for style, "medium" for real maintainability problems.
{{- template "issues-format" .}}
"""
