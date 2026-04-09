package config

pass: authorization: {
	description: "Authorization enforcement violations"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for authorization enforcement violations.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}

The transport layer authenticates and stores identity on context. Service and data layers
extract identity and enforce authorization. Flag violations of this boundary:

- Service or repository method calls context.WithValue to set user identity. Only the transport layer sets auth context.
- Authorization by fetching all rows then filtering in application code. Embed the constraint in the SQL WHERE clause.
- Service method accepts userID as an explicit parameter instead of extracting it from context as a mandatory constraint.
- Authorization enforced only in middleware with no check in the service or data layer. Background jobs, CLI, and tests bypass middleware.

{{- template "issues-format" .}}
"""
