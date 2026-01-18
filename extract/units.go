// extract/units.go
package extract

import (
	"sort"
	"strings"
)

// AnalysisUnit is the atomic unit of analysis
type AnalysisUnit struct {
	ID        string
	Functions []*FunctionInfo
	Callees   []string
}

// BuildAnalysisUnits creates analysis units from functions and callgraph
// Units are returned in topological order (callees before callers)
func BuildAnalysisUnits(funcs []*FunctionInfo, graph map[string][]string) []*AnalysisUnit {
	// Build function lookup
	funcMap := make(map[string]*FunctionInfo)
	for _, f := range funcs {
		id := f.Package + "." + f.Name
		if f.Receiver != "" {
			id = f.Package + ".(" + f.Receiver + ")." + f.Name
		}
		funcMap[id] = f
	}

	// Filter graph to only include internal functions
	internalGraph := make(map[string][]string)
	for caller, callees := range graph {
		if _, ok := funcMap[caller]; !ok {
			continue
		}
		internalCallees := []string{}
		for _, callee := range callees {
			if _, ok := funcMap[callee]; ok {
				internalCallees = append(internalCallees, callee)
			}
		}
		internalGraph[caller] = internalCallees
	}

	// Add nodes with no outgoing edges
	for id := range funcMap {
		if _, ok := internalGraph[id]; !ok {
			internalGraph[id] = []string{}
		}
	}

	// Compute SCCs
	sccs := TarjanSCC(internalGraph)

	// Build units from SCCs
	units := make([]*AnalysisUnit, 0, len(sccs))
	sccMap := make(map[string]int) // function ID -> SCC index

	for i, scc := range sccs {
		for _, id := range scc {
			sccMap[id] = i
		}
	}

	// First pass: Create all units and assign their IDs
	// Also build a map from function ID to unit ID
	funcToUnitID := make(map[string]string)

	for _, scc := range sccs {
		unit := &AnalysisUnit{
			Functions: make([]*FunctionInfo, 0, len(scc)),
			Callees:   []string{},
		}

		// Collect functions
		for _, id := range scc {
			if f, ok := funcMap[id]; ok {
				unit.Functions = append(unit.Functions, f)
			}
		}

		// Build ID: simpler ID for single-function units
		if len(unit.Functions) == 1 {
			f := unit.Functions[0]
			unit.ID = f.Package + "." + f.Name
			if f.Receiver != "" {
				unit.ID = f.Package + ".(" + f.Receiver + ")." + f.Name
			}
		} else {
			// Build ID from sorted function names for multi-function SCCs
			sortedSCC := make([]string, len(scc))
			copy(sortedSCC, scc)
			sort.Strings(sortedSCC)
			unit.ID = strings.Join(sortedSCC, "+")
		}

		// Map each function in this SCC to the unit ID
		for _, id := range scc {
			funcToUnitID[id] = unit.ID
		}

		units = append(units, unit)
	}

	// Second pass: Populate Callees using the unit ID map
	for i, scc := range sccs {
		seenCallees := make(map[string]bool)
		for _, id := range scc {
			for _, callee := range internalGraph[id] {
				calleeUnitIdx := sccMap[callee]
				if calleeUnitIdx != i {
					calleeUnitID := funcToUnitID[callee]
					if !seenCallees[calleeUnitID] {
						units[i].Callees = append(units[i].Callees, calleeUnitID)
						seenCallees[calleeUnitID] = true
					}
				}
			}
		}
	}

	return units
}
