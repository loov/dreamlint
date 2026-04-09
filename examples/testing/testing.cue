package config

pass: testing: {
	description: "Test mechanics violations"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go test code for test mechanics violations.
{{template "function-context" .}}
{{- template "summary-context" .}}

Flag any of:

Helpers:
- Test helper missing t.Helper(). Failure output will point inside the helper instead of the call site.
- Helper returns an error instead of calling t.Fatalf
- Cleanup using defer instead of t.Cleanup. t.Cleanup runs even after t.FailNow.

Assertions:
- Third-party assertion or mock library (testify, gomock, gocheck, ginkgo, gomega). Use stdlib only.
- reflect.DeepEqual for struct comparison. Use cmp.Diff/cmp.Equal.
- Mock used to test a pure function. Pure functions need no test doubles.

Naming and structure:
- Vague test name (TestSuccess, TestCancel). Name like a bug report: TestOrderService_Cancel_RefundsPartialShipment.
- Table-driven cases without named subtests via t.Run
- Table test loop body over 10 lines or with branching. Keep it trivial; split into separate subtests.
- Sub-cases not wrapped in t.Run
- Test of unexported function as primary strategy. Test the exported API.

Timing:
- time.Sleep with a fixed duration. Use select with time.After.
- Async condition polled in a sleep loop. Use a channel.

Isolation:
- t.SetEnv used. It disables t.Parallel() for the binary. Inject getenv instead.
- t.Parallel() with reads/writes to package-level variables
- t.Parallel() absent when the test has a fully isolated environment

Structure:
- Complex expected output hardcoded inline. Use golden files with -update flag.
- Integration test not guarded by //go:build integration

Interfaces:
- Interface defined in the implementing package instead of at the point of use
- Interface wider than the methods actually called

{{- template "issues-format" .}}
"""
