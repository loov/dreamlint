# reviewmod

reviewmod is a staged Go code review tool that analyzes functions in dependency order using LLMs. It works around context limits by building a callgraph, grouping mutually recursive functions, and analyzing them bottom-up so that callee summaries inform caller analysis.

This project was created with AI assistance using Claude.

## Installation

```
go install github.com/loov/reviewmod@latest
```

## Usage

```
reviewmod [flags] [packages...]
```

By default, reviewmod analyzes all packages in the current module using the configuration in `reviewmod.cue`.

Flags:

```
-config string    path to config file (default "reviewmod.cue")
-format string    output format: json, markdown, sarif, or all (default "all")
-resume           resume from existing partial report
-prompts string   directory to load prompts from (overrides builtin prompts)
```

## Configuration

Create a [`reviewmod.cue`](reviewmod.cue) file in your project root.

Each analysis pass can specify its own LLM configuration to use different models for different tasks. See [`config/schema.cue`](config/schema.cue) for details about configuration.

## How It Works

reviewmod extracts all functions from the specified packages and builds a callgraph using Class Hierarchy Analysis. It then computes strongly connected components using Tarjan's algorithm to group mutually recursive functions. These groups are sorted in reverse topological order so that callees are analyzed before their callers.

For each analysis unit, reviewmod first generates a summary describing the function's purpose, behavior, invariants, and security properties. This summary is cached and passed to callers during their analysis. Then it runs each configured analysis pass (security, error handling, cleanliness) and collects issues.

Results are written as JSON for programmatic consumption, Markdown for human review, and SARIF for integration with code analysis tools.

## License

MIT
