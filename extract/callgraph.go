// extract/callgraph.go
package extract

import (
	"fmt"
	"slices"

	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// BuildCallgraph builds a callgraph using CHA analysis from loaded packages.
// Returns a map from function ID to list of callee IDs.
func BuildCallgraph(p *Packages) map[string][]string {
	return BuildCallgraphFromPackages(p.Pkgs)
}

// BuildCallgraphFromPackages builds a callgraph from a slice of packages.
func BuildCallgraphFromPackages(pkgs []*packages.Package) map[string][]string {
	// Build SSA
	prog, _ := ssautil.AllPackages(pkgs, ssa.SanityCheckFunctions)
	prog.Build()

	// Build callgraph using CHA
	cg := cha.CallGraph(prog)

	// Convert to our format
	graph := make(map[string][]string)

	for fn, node := range cg.Nodes {
		if fn == nil {
			continue
		}

		callerID := funcID(fn)
		if callerID == "" {
			continue
		}

		// Initialize entry even if no callees
		if _, ok := graph[callerID]; !ok {
			graph[callerID] = []string{}
		}

		for _, edge := range node.Out {
			if edge.Callee.Func == nil {
				continue
			}

			calleeID := funcID(edge.Callee.Func)
			if calleeID == "" {
				continue
			}

			// Avoid duplicates
			if !slices.Contains(graph[callerID], calleeID) {
				graph[callerID] = append(graph[callerID], calleeID)
			}
		}
	}

	return graph
}

func funcID(fn *ssa.Function) string {
	if fn.Pkg == nil {
		return ""
	}

	pkg := fn.Pkg.Pkg.Path()
	name := fn.Name()

	// Handle methods
	if recv := fn.Signature.Recv(); recv != nil {
		return fmt.Sprintf("%s.(%s).%s", pkg, recv.Type().String(), name)
	}

	return fmt.Sprintf("%s.%s", pkg, name)
}
