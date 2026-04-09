package config

pass: layout: {
	description: "Application layout conventions"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for application layout violations.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Flag any of:

init() functions:
Avoid init(). It is acceptable only for driver/codec registration (sql.Register,
image.RegisterFormat) and initializing computed data without I/O. Flag any other use.

main() functions:
main() should only call run() and handle the error. Flag:
- Business logic, flag parsing, dependency construction, or os.Getenv in main()
- Anything beyond: if err := run(...); err != nil { os.Exit(1) }

Global state:
Dependencies must be passed explicitly. Flag:
- DB connection, HTTP client, cache, or config read from a package-level variable instead of a parameter or struct field
- Singleton initialized via package-level var instead of a constructor

{{- template "issues-format" .}}
"""
