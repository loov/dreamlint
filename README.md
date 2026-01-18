# dreamlint

dreamlint is a staged Go code review tool that analyzes functions in dependency order using LLMs. It works around context limits by building a callgraph, grouping mutually recursive functions, and analyzing them bottom-up so that callee summaries inform caller analysis.

The name reflects the nature of LLM-based code review: the issues it finds might be real, might be hallucinated, or might be somewhere in between. Consider it a fever dream review.

This project was created with AI assistance using Claude.

## Installation

```
go install github.com/loov/dreamlint@latest
```

## Usage

```
dreamlint [flags] [packages...]
```

By default, dreamlint analyzes all packages in the current module using the configuration in `dreamlint.cue`.

Flags:

```
-config string    path to config file (default "dreamlint.cue")
-format string    output format: json, markdown, sarif, or all (default "all")
-resume           resume from existing partial report
-prompts string   directory to load prompts from (overrides builtin prompts)
```

## Prompts

To write custom prompts see the builtin prompts in [analyze/prompts](analyze/prompts).

## Configuration

Create a [`dreamlint.cue`](dreamlint.cue) file in your project root.

Each analysis pass can specify its own LLM configuration to use different models for different tasks. See [`config/schema.cue`](config/schema.cue) for details.

## How It Works

dreamlint extracts all functions from the specified packages and builds a callgraph using Class Hierarchy Analysis. It then computes strongly connected components using Tarjan's algorithm to group mutually recursive functions. These groups are sorted in reverse topological order so that callees are analyzed before their callers.

For each analysis unit, dreamlint first generates a summary describing the function's purpose, behavior, invariants, and security properties. This summary is cached and passed to callers during their analysis. Then it runs each configured analysis pass (security, error handling, cleanliness) and collects issues.

Results are written as JSON for programmatic consumption, Markdown for human review, and SARIF for integration with code analysis tools.
