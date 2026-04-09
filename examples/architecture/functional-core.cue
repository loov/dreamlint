package config

pass: "functional-core": {
	description: "Functional core / imperative shell violations"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for functional core / imperative shell violations.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Domain/business logic functions should accept values and return values without performing I/O.
Do not flag functions whose purpose is I/O (repositories, handlers, workers, wiring code).

I/O mixed into domain logic — flag if computation and I/O coexist in the same function:
- database/sql, sqlx, pgx, or ORM calls alongside business logic
- http.Get, http.Post, http.Client.Do, or gRPC calls alongside computation
- time.Now() — inject current time as a parameter for determinism
- rand.Float64(), rand.Intn() — inject the source or value
- os.Getenv, os.ReadFile, os.Open — read at the boundary, pass values in
- Concrete dependency type where an interface would allow testing. But do not flag prematurely: start concrete, extract an interface only when a second consumer or test demands it.

Mutation:
- Method mutates its receiver when it could return a new value
- Function modifies a slice or map passed as a parameter instead of returning a new one

Shell absorbing logic:
- Sorting, filtering, ranking, or business rule evaluation inline in a handler or service method instead of delegated to a pure function
- Three or more conditional branches that depend only on values already in scope (no I/O needed) — extract to a pure function

{{- template "issues-format" .}}
"""
