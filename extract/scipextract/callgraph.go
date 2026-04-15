package scipextract

import (
	"math"
	"slices"
	"sort"
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
	docRanges map[*scip.Document]map[string]scip.Range,
	symbolToID map[string]string,
) (map[string][]string, map[string]*extract.ExternalFunc) {
	graph := make(map[string][]string, len(symbolToID))
	for _, id := range symbolToID {
		if _, ok := graph[id]; !ok {
			graph[id] = nil
		}
	}

	externalSeen := make(map[string]bool)
	external := make(map[string]*extract.ExternalFunc)

	for _, doc := range docs {
		defRanges := docRanges[doc]
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

// definitionRanges returns a map from function symbol to the source range
// that should be considered "inside" that function for callgraph purposes.
//
// Preferred: the occurrence's EnclosingRange. Some indexers (notably
// scip-clang) don't emit EnclosingRange, in which case we fall back to a
// "span to next function" heuristic: the definition range of function F
// extends until the next function definition in the same document. This
// recovers useful call edges for top-level functions but will mis-attribute
// calls in nested functions.
//
// Only symbols present in internal are considered — this keeps file/
// namespace definitions from segmenting the per-function spans.
func definitionRanges(doc *scip.Document, internal map[string]string) map[string]scip.Range {
	type entry struct {
		sym          string
		r            scip.Range
		hasEnclosing bool
	}
	var entries []entry
	seen := make(map[string]bool)
	for _, occ := range doc.Occurrences {
		if occ.SymbolRoles&int32(scip.SymbolRole_Definition) == 0 {
			continue
		}
		if occ.Symbol == "" {
			continue
		}
		if _, ok := internal[occ.Symbol]; !ok {
			continue
		}
		if seen[occ.Symbol] {
			continue
		}
		seen[occ.Symbol] = true

		var r scip.Range
		hasEnc := false
		if len(occ.EnclosingRange) > 0 {
			if rr, err := scip.NewRange(occ.EnclosingRange); err == nil {
				r = rr
				hasEnc = true
			}
		}
		if !hasEnc {
			rr, err := scip.NewRange(occ.Range)
			if err != nil {
				continue
			}
			r = rr
		}
		entries = append(entries, entry{sym: occ.Symbol, r: r, hasEnclosing: hasEnc})
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].r.Start.Less(entries[j].r.Start)
	})

	out := make(map[string]scip.Range, len(entries))
	for i, e := range entries {
		if !e.hasEnclosing {
			if i+1 < len(entries) {
				// Stop at the start of the next function's line, exclusive.
				e.r.End = scip.Position{Line: entries[i+1].r.Start.Line}
			} else {
				e.r.End = scip.Position{Line: math.MaxInt32}
			}
		}
		out[e.sym] = e.r
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
	switch last.Suffix {
	case scip.Descriptor_Method, scip.Descriptor_Macro:
		// callable
	default:
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
