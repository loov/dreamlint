package scipextract

import (
	"slices"
	"strings"

	"github.com/scip-code/scip/bindings/go/scip"

	"github.com/loov/dreamlint/extract"
)

// buildCallgraph walks each function's enclosing range and records an
// F -> id[o.Symbol] edge for every non-definition occurrence whose symbol
// names another function. Any callee symbol not in the internal set becomes
// an ExternalFunc, looked up in index.ExternalSymbols when available.
//
// The SCIP reference-as-call-edge approximation is coarse (a function
// pointer pushed onto a vtable counts as a call), but it keeps the
// downstream SCC ordering meaningful.
func buildCallgraph(
	docs []*scip.Document,
	index *scip.Index,
	funcs []*extract.FunctionInfo,
	symbolToID map[string]string,
) (map[string][]string, map[string]*extract.ExternalFunc) {
	graph := make(map[string][]string, len(funcs))
	for _, id := range symbolToID {
		if _, ok := graph[id]; !ok {
			graph[id] = nil
		}
	}

	externalSeen := make(map[string]bool)
	external := make(map[string]*extract.ExternalFunc)

	for _, doc := range docs {
		defRanges := definitionRanges(doc)
		for _, sym := range doc.Symbols {
			callerID, ok := symbolToID[sym.Symbol]
			if !ok {
				continue
			}
			callerRange, ok := defRanges[sym.Symbol]
			if !ok {
				continue
			}
			for _, occ := range doc.Occurrences {
				if occ.Symbol == "" || occ.Symbol == sym.Symbol {
					continue
				}
				if occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0 {
					continue
				}
				or, err := scip.NewRange(occ.Range)
				if err != nil {
					continue
				}
				if !callerRange.Contains(or.Start) {
					continue
				}
				if calleeID, ok := symbolToID[occ.Symbol]; ok {
					if calleeID == callerID {
						continue
					}
					if !slices.Contains(graph[callerID], calleeID) {
						graph[callerID] = append(graph[callerID], calleeID)
					}
					continue
				}
				// External reference.
				extID := externalID(occ.Symbol)
				if externalSeen[extID] {
					if !slices.Contains(graph[callerID], extID) {
						graph[callerID] = append(graph[callerID], extID)
					}
					continue
				}
				if ext := buildExternalFunc(occ.Symbol, index); ext != nil {
					external[extID] = ext
					externalSeen[extID] = true
					if !slices.Contains(graph[callerID], extID) {
						graph[callerID] = append(graph[callerID], extID)
					}
				}
			}
		}
	}

	return graph, external
}

// definitionRanges returns a map from function symbol to its enclosing range
// (or the definition range as a fallback) for every function-like symbol
// defined in doc.
func definitionRanges(doc *scip.Document) map[string]scip.Range {
	out := make(map[string]scip.Range)
	for _, occ := range doc.Occurrences {
		if occ.SymbolRoles&int32(scip.SymbolRole_Definition) == 0 {
			continue
		}
		if occ.Symbol == "" {
			continue
		}
		if _, exists := out[occ.Symbol]; exists {
			continue
		}
		if len(occ.EnclosingRange) > 0 {
			if r, err := scip.NewRange(occ.EnclosingRange); err == nil {
				out[occ.Symbol] = r
				continue
			}
		}
		if r, err := scip.NewRange(occ.Range); err == nil {
			out[occ.Symbol] = r
		}
	}
	return out
}

// externalID is the key used in the callgraph and ExternalFunc map for
// symbols defined outside the indexed documents. We reuse the full SCIP
// symbol string so it's unambiguous across languages.
func externalID(symbol string) string {
	return "scip:" + symbol
}

// buildExternalFunc returns a shallow ExternalFunc for a reference to
// symbol. Metadata is pulled from index.ExternalSymbols when available;
// otherwise we synthesize it from the parsed descriptors.
func buildExternalFunc(symbol string, index *scip.Index) *extract.ExternalFunc {
	sym, err := scip.ParseSymbol(symbol)
	if err != nil {
		return nil
	}
	// Filter to function-like symbols.
	if len(sym.Descriptors) == 0 {
		return nil
	}
	last := sym.Descriptors[len(sym.Descriptors)-1]
	if last.Suffix != scip.Descriptor_Method {
		return nil
	}

	ext := &extract.ExternalFunc{
		Package: packageName(sym),
		Name:    last.Name,
	}

	if info := findExternalSymbolInfo(symbol, index); info != nil {
		if info.DisplayName != "" {
			ext.Name = info.DisplayName
		}
		if info.SignatureDocumentation != nil {
			ext.Signature = info.SignatureDocumentation.Text
		}
		ext.Godoc = strings.Join(info.Documentation, "\n\n")
	}

	return ext
}

// findExternalSymbolInfo returns the SymbolInformation for symbol in
// index.ExternalSymbols, or nil if absent.
func findExternalSymbolInfo(symbol string, index *scip.Index) *scip.SymbolInformation {
	if index == nil {
		return nil
	}
	for _, info := range index.ExternalSymbols {
		if info.Symbol == symbol {
			return info
		}
	}
	return nil
}
