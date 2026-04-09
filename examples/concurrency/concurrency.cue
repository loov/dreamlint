package config

pass: concurrency: {
	description: "Concurrency issues"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for concurrency bugs and design issues.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Flag any of:

Design:
- Concurrency where a synchronous call chain would suffice. Default to no concurrency.
- Goroutine spawned inside a callee instead of by the caller. The caller who writes `go` owns the lifecycle.
- Raw sync.WaitGroup with Add/Done when errgroup.Group would be better. Use sync.WaitGroup.Go or errgroup.
- Worker pool (N goroutines pulling from a channel) when a semaphore/limiter would suffice.

Safety:
- Race condition on shared state without synchronization
- Goroutine with no shutdown path. Every `go` needs a corresponding wait.
- Missing mutex lock/unlock. Place the mutex directly above the fields it guards.
- Deadlock potential from lock ordering
- Send on closed channel, or operation on nil channel
- Signal channel using chan bool instead of chan struct{}. Broadcast via close(done).
- sync.WaitGroup misuse: Add after Wait, wrong count
- Context cancellation ignored in long operations
- sync.Mutex copied by value
- Missing recover() at goroutine boundary

{{- template "issues-format" .}}
"""
