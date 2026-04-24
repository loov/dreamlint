// Package scipextract builds dreamlint analysis units from a pre-generated
// SCIP index (https://github.com/sourcegraph/scip). It consumes a .scip file
// produced by any SCIP indexer (scip-typescript, scip-java, scip-python,
// scip-clang, rust-analyzer --scip, ...) and shapes the data into
// extract.Result so the downstream LLM pipeline is unchanged.
package scipextract

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/scip-code/scip/bindings/go/scip"
	"google.golang.org/protobuf/proto"

	"github.com/loov/dreamlint/extract"
)

// Extractor consumes a .scip index and produces analysis units.
type Extractor struct {
	// IndexPath is the absolute or relative path to the .scip file.
	IndexPath string

	// PathFilters are optional filepath.Match globs (matched against
	// Document.RelativePath). An empty slice means no filtering.
	PathFilters []string

	// ProjectRoot overrides the project root resolved from the index metadata.
	// Most callers should leave this zero.
	ProjectRoot string
}

// Extract parses the SCIP index and returns analysis units. Step 4 deliberately
// produces one AnalysisUnit per function with empty Callees; the call graph is
// built in a follow-up step.
func (e *Extractor) Extract(ctx context.Context) (*extract.Result, error) {
	data, err := os.ReadFile(e.IndexPath)
	if err != nil {
		return nil, fmt.Errorf("read scip index: %w", err)
	}

	var index scip.Index
	if err := proto.Unmarshal(data, &index); err != nil {
		return nil, fmt.Errorf("unmarshal scip index: %w", err)
	}

	root := e.ProjectRoot
	if root == "" && index.Metadata != nil {
		root = stripFileURI(index.Metadata.ProjectRoot)
	}

	docs := filterDocuments(index.Documents, e.PathFilters)

	src := newSourceCache(root)

	var warnings []string

	// Pass 1: identify function and type symbols and assign ids. No body
	// extraction yet — we need the per-doc definition ranges (computed in
	// pass 2) before we can pick the right span for each function/type.
	type fnEntry struct {
		info    *scip.SymbolInformation
		doc     *scip.Document
		absPath string
		fn      *extract.FunctionInfo
	}
	type typeEntry struct {
		info    *scip.SymbolInformation
		doc     *scip.Document
		absPath string
		ti      *extract.TypeInfo
	}
	var fnEntries []fnEntry
	var typeEntries []typeEntry
	var funcs []*extract.FunctionInfo
	symbolToID := make(map[string]string)
	typesBySymbol := make(map[string]*extract.TypeInfo)
	typeSymbolSet := make(map[string]bool)
	for _, doc := range docs {
		absPath := filepath.Join(root, doc.RelativePath)
		for _, sym := range doc.Symbols {
			switch {
			case isFunctionSymbol(sym):
				if _, dup := symbolToID[sym.Symbol]; dup {
					warnings = append(warnings, fmt.Sprintf("duplicate function symbol %s in %s (first wins)", sym.Symbol, doc.RelativePath))
					continue
				}
				fn := buildFunctionInfo(sym, doc, absPath)
				if fn == nil {
					continue
				}
				symbolToID[sym.Symbol] = fn.ID()
				fnEntries = append(fnEntries, fnEntry{info: sym, doc: doc, absPath: absPath, fn: fn})
				funcs = append(funcs, fn)
			case isTypeSymbol(sym):
				if _, dup := typesBySymbol[sym.Symbol]; dup {
					warnings = append(warnings, fmt.Sprintf("duplicate type symbol %s in %s (first wins)", sym.Symbol, doc.RelativePath))
					continue
				}
				ti := buildTypeInfo(sym, doc, absPath)
				if ti == nil {
					continue
				}
				typesBySymbol[sym.Symbol] = ti
				typeSymbolSet[sym.Symbol] = true
				typeEntries = append(typeEntries, typeEntry{info: sym, doc: doc, absPath: absPath, ti: ti})
			}
		}
	}

	// Pass 2: per-doc definition ranges (with span-to-next-function fallback).
	docRanges := make(map[*scip.Document]map[string]scip.Range, len(docs))
	docTypeRanges := make(map[*scip.Document]map[string]scip.Range, len(docs))
	for _, doc := range docs {
		docRanges[doc] = definitionRanges(doc, symbolToID)
		docTypeRanges[doc] = typeDefinitionRanges(doc, typeSymbolSet)
	}

	// Pass 3: bodies use the shared ranges so indexers without
	// EnclosingRange (e.g. scip-clang) still get useful function bodies.
	for _, e := range fnEntries {
		body, warn := extractBody(e.info, e.doc, e.absPath, docRanges[e.doc], src)
		e.fn.Body = body
		if warn != "" {
			warnings = append(warnings, warn)
		}
	}
	for _, e := range typeEntries {
		e.ti.Body = extractTypeBody(e.info, e.doc, e.absPath, docTypeRanges[e.doc], src)
	}

	// Link methods to their receiver type (by package + receiver name,
	// which matches how types are keyed). External receivers stay
	// unlinked.
	types := make(map[string]*extract.TypeInfo, len(typeEntries))
	for _, e := range typeEntries {
		types[typeID(e.ti)] = e.ti
	}
	for _, fn := range funcs {
		if fn.Receiver == "" {
			continue
		}
		id := fn.Package + "." + fn.Receiver
		ti, ok := types[id]
		if !ok {
			continue
		}
		fn.ReceiverType = id
		ti.Methods = append(ti.Methods, fn.ID())
	}

	graph, external := buildCallgraph(docs, &index, docRanges, symbolToID)
	units := extract.BuildAnalysisUnits(funcs, graph)

	return &extract.Result{
		Units:    units,
		External: external,
		Types:    types,
		Language: pickLanguage(docs),
		Warnings: warnings,
	}, nil
}


// filterDocuments keeps documents whose RelativePath matches any of the globs.
// Empty filters means keep all.
func filterDocuments(docs []*scip.Document, filters []string) []*scip.Document {
	if len(filters) == 0 {
		return docs
	}
	var out []*scip.Document
	for _, doc := range docs {
		for _, pat := range filters {
			ok, err := filepath.Match(pat, doc.RelativePath)
			if err == nil && ok {
				out = append(out, doc)
				break
			}
		}
	}
	return out
}

// pickLanguage returns a display-friendly language name derived from the
// most common Document.Language across the filtered documents. When the
// indexer leaves Document.Language empty (scip-typescript, for example),
// fall back to inferring from symbol schemes.
func pickLanguage(docs []*scip.Document) string {
	counts := make(map[string]int)
	for _, doc := range docs {
		if doc.Language != "" {
			counts[doc.Language]++
		}
	}
	if len(counts) == 0 {
		return inferLanguageFromScheme(docs)
	}
	// Deterministic: pick the highest count, ties broken by name.
	type kv struct {
		lang  string
		count int
	}
	entries := make([]kv, 0, len(counts))
	for l, c := range counts {
		entries = append(entries, kv{l, c})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].count != entries[j].count {
			return entries[i].count > entries[j].count
		}
		return entries[i].lang < entries[j].lang
	})
	return displayLanguage(entries[0].lang)
}

// inferLanguageFromScheme maps the scheme of the first parseable symbol to
// a display language. Best-effort — returns "" if no scheme is recognized.
func inferLanguageFromScheme(docs []*scip.Document) string {
	for _, doc := range docs {
		for _, info := range doc.Symbols {
			sym, err := scip.ParseSymbol(info.Symbol)
			if err != nil {
				continue
			}
			switch sym.Scheme {
			case "scip-typescript":
				return "TypeScript"
			case "scip-java", "semanticdb":
				return "Java"
			case "scip-python":
				return "Python"
			case "scip-ruby":
				return "Ruby"
			case "rust-analyzer":
				return "Rust"
			case "scip-go":
				return "Go"
			case "scip-clang":
				return "C++"
			}
		}
	}
	return ""
}

// displayLanguage maps SCIP Language enum strings to human-readable names.
// Indexers are inconsistent about case (scip-clang emits "CPP",
// rust-analyzer emits lowercase "rust"), so normalize here.
func displayLanguage(scipLang string) string {
	switch strings.ToLower(scipLang) {
	case "cpp":
		return "C++"
	case "csharp":
		return "C#"
	case "objectivec":
		return "Objective-C"
	case "objectivecpp":
		return "Objective-C++"
	case "javascript":
		return "JavaScript"
	case "typescript":
		return "TypeScript"
	case "go":
		return "Go"
	case "rust":
		return "Rust"
	case "java":
		return "Java"
	case "kotlin":
		return "Kotlin"
	case "python":
		return "Python"
	case "ruby":
		return "Ruby"
	case "c":
		return "C"
	}
	return scipLang
}

// stripFileURI converts a "file://..." URI into a local path. Handles
// both POSIX form ("file:///home/u/p" → "/home/u/p") and Windows form
// ("file:///C:/p" → "C:/p", "file://C:/p" → "C:/p").
func stripFileURI(uri string) string {
	s := strings.TrimPrefix(uri, "file://")
	// Windows drive-letter paths: drop the leading "/" in "/C:/...".
	if len(s) >= 3 && s[0] == '/' && isDriveLetter(s[1]) && s[2] == ':' {
		s = s[1:]
	}
	return s
}

func isDriveLetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}
