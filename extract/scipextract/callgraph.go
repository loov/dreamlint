package scipextract

import (
	"math"
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

	extSymIndex := buildExternalSymbolIndex(index)
	calleeSeen := make(map[string]map[string]bool, len(symbolToID))
	externalSeen := make(map[string]bool)
	external := make(map[string]*extract.ExternalFunc)

	addEdge := func(callerID, calleeID string) {
		s := calleeSeen[callerID]
		if s == nil {
			s = make(map[string]bool)
			calleeSeen[callerID] = s
		}
		if s[calleeID] {
			return
		}
		s[calleeID] = true
		graph[callerID] = append(graph[callerID], calleeID)
	}

	for _, doc := range docs {
		defRanges := docRanges[doc]

		// Build a sorted list of caller ranges for this document so we
		// can walk occurrences once instead of once per symbol.
		type caller struct {
			id string
			r  scip.Range
		}
		var callers []caller
		for _, sym := range doc.Symbols {
			callerID, ok := symbolToID[sym.Symbol]
			if !ok {
				continue
			}
			r, ok := defRanges[sym.Symbol]
			if !ok {
				continue
			}
			callers = append(callers, caller{id: callerID, r: r})
		}
		sort.Slice(callers, func(i, j int) bool {
			return callers[i].r.Start.Less(callers[j].r.Start)
		})

		// Walk each occurrence once, attributing it to every caller
		// whose range contains it.
		for _, occ := range doc.Occurrences {
			if occ.Symbol == "" {
				continue
			}
			if occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0 {
				continue
			}
			or, err := scip.NewRange(occ.Range)
			if err != nil {
				continue
			}

			// Resolve the callee once — it's the same regardless of
			// which caller contains this occurrence.
			calleeID, isInternal := symbolToID[occ.Symbol]
			if !isInternal {
				extID := externalID(occ.Symbol)
				if !externalSeen[extID] {
					ext := buildExternalFunc(occ.Symbol, extSymIndex)
					if ext == nil {
						continue
					}
					external[extID] = ext
					externalSeen[extID] = true
				}
				calleeID = extID
			}

			// Scan callers sorted by start position; stop once the
			// caller starts past the occurrence.
			for _, c := range callers {
				if or.Start.Less(c.r.Start) {
					break
				}
				if !c.r.Contains(or.Start) {
					continue
				}
				if calleeID != c.id {
					addEdge(c.id, calleeID)
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
// recovers useful call edges for top-level functions.
//
// Known limitation: when a nested function (closure / lambda) is
// defined inside an outer function and the indexer omits EnclosingRange,
// the outer function's fallback range is truncated at the start of the
// inner function. Calls after the inner definition are lost from the
// outer's callgraph. When EnclosingRange IS present, both ranges
// overlap and calls inside the inner body appear in both caller entries.
// Fixing this requires attributing to the innermost enclosing range,
// which is deferred until an indexer produces this pattern in practice.
//
// Only symbols present in internal are considered — this keeps file/
// namespace definitions from segmenting the per-function spans.
func definitionRanges(doc *scip.Document, internal map[string]string) map[string]scip.Range {
	return collectDefinitionRanges(doc, func(sym string) bool {
		_, ok := internal[sym]
		return ok
	}, func(_ int32) int32 { return math.MaxInt32 })
}

// collectDefinitionRanges scans doc for Definition occurrences whose
// symbol passes include, computes a source range for each (preferring
// EnclosingRange, falling back to span-to-next-definition), and returns
// the result keyed by symbol string.
//
// lastFallbackEnd returns the End.Line for the last entry when
// EnclosingRange is absent. It receives the entry's Start.Line so
// callers can compute either an absolute value (math.MaxInt32 for
// functions) or a relative one (startLine + N for types).
func collectDefinitionRanges(doc *scip.Document, include func(string) bool, lastFallbackEnd func(startLine int32) int32) map[string]scip.Range {
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
		if occ.Symbol == "" || !include(occ.Symbol) || seen[occ.Symbol] {
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
				e.r.End = scip.Position{Line: entries[i+1].r.Start.Line}
			} else {
				e.r.End = scip.Position{Line: lastFallbackEnd(e.r.Start.Line)}
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

// buildExternalSymbolIndex builds a map from symbol string to
// SymbolInformation for all external symbols in the index. Called
// once per extraction so lookups during callgraph construction are O(1).
func buildExternalSymbolIndex(index *scip.Index) map[string]*scip.SymbolInformation {
	if index == nil {
		return nil
	}
	m := make(map[string]*scip.SymbolInformation, len(index.ExternalSymbols))
	for _, info := range index.ExternalSymbols {
		m[info.Symbol] = info
	}
	return m
}

// buildExternalFunc returns a shallow ExternalFunc for a reference to
// symbol. Metadata is pulled from extSymIndex when available;
// otherwise we synthesize it from the parsed descriptors.
//
// Filtering mirrors isFunctionSymbol: when the index carries
// SymbolInformation for this external, the full Kind+descriptor check
// is applied; otherwise we fall back to the descriptor suffix alone.
func buildExternalFunc(symbol string, extSymIndex map[string]*scip.SymbolInformation) *extract.ExternalFunc {
	sym, err := scip.ParseSymbol(symbol)
	if err != nil || len(sym.Descriptors) == 0 {
		return nil
	}

	info := extSymIndex[symbol]
	if info != nil {
		if !isFunctionSymbol(info) {
			return nil
		}
	} else if !isCallableDescriptor(sym.Descriptors[len(sym.Descriptors)-1]) {
		return nil
	}

	last := sym.Descriptors[len(sym.Descriptors)-1]
	ext := &extract.ExternalFunc{
		Package: packageName(sym),
		Name:    last.Name,
	}

	if info != nil {
		if info.DisplayName != "" {
			ext.Name = info.DisplayName
		}
		if info.SignatureDocumentation != nil {
			ext.Signature = info.SignatureDocumentation.Text
		}
		ext.Doc = strings.Join(info.Documentation, "\n\n")
	}

	return ext
}
