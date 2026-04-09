package config

pass: "domain-types": {
	description: "Domain type and CRUD convention violations"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for domain type and CRUD convention violations.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}

Flag any of:

FindByID / FindOne:
- Returns (nil, nil) when entity does not exist. Return a sentinel error so callers can distinguish "not found" from other failures.

FindMany / List:
- Result slice declared with var (encodes as JSON null). Initialize with make([]*T).
- Returns an error for an empty result set. An empty list is not an error.

Create:
- Does not set ID, CreatedAt, UpdatedAt on the input pointer on success
- Returns a new entity value instead of populating the caller's struct

Update:
- Does not return the updated object on error (callers need it to replay a form)
- Accepts individual fields instead of an *Update struct with pointer fields
- Update struct uses non-pointer fields (nil cannot express "do not change")

Delete:
- Accepts more than the primary key. Auth checks belong inside the implementation.

Filter structs:
- Filter field is a non-pointer type (nil cannot express "not filtered on")
- Raw offset for pagination. Use a cursor (last-seen ID) to avoid skips under concurrent writes.

Interface contracts:
- Service method's first parameter is not context.Context
- Cache/decorator wrapper re-implements domain logic instead of delegating

Only report actual violations.
{{- template "issues-format" .}}
"""
