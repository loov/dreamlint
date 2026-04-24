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
	"slices"
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
			switch classifySymbol(sym) {
			case classFunction:
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
			case classType:
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
	docRanges := make(map[*scip.Document]docDefinitions, len(docs))
	docTypeRanges := make(map[*scip.Document]docDefinitions, len(docs))
	for _, doc := range docs {
		docRanges[doc] = definitionRanges(doc, symbolToID)
		docTypeRanges[doc] = typeDefinitionRanges(doc, typeSymbolSet)
	}

	// Pass 3: bodies use the shared ranges so indexers without
	// EnclosingRange (e.g. scip-clang) still get useful function bodies.
	for _, e := range fnEntries {
		body, warn := extractBody(e.info, e.doc, e.absPath, docRanges[e.doc].ByID, src)
		e.fn.Body = body
		if warn != "" {
			warnings = append(warnings, warn)
		}
	}
	for _, e := range typeEntries {
		e.ti.Body = extractTypeBody(e.info, e.doc, e.absPath, docTypeRanges[e.doc].ByID, src)
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
		id := receiverKey(fn.Package, fn.Receiver)
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

// languageEntry describes one display language and every SCIP handle
// that maps to it. The table below feeds three lookups (Document.Language
// enum, indexer scheme, file extension) so adding a new language is a
// one-spot change.
type languageEntry struct {
	Display    string
	Schemes    []string // SCIP indexer schemes (sym.Scheme) producing this language
	SCIPNames  []string // scip.Language enum strings, lowercased
	Extensions []string // file extensions including leading dot, lowercased
}

var languageTable = []languageEntry{
	{Display: "Go", Schemes: []string{"scip-go"}, SCIPNames: []string{"go"}, Extensions: []string{".go"}},
	{Display: "Rust", Schemes: []string{"rust-analyzer"}, SCIPNames: []string{"rust"}, Extensions: []string{".rs"}},
	{Display: "Java", Schemes: []string{"scip-java", "semanticdb"}, SCIPNames: []string{"java"}, Extensions: []string{".java"}},
	{Display: "Kotlin", SCIPNames: []string{"kotlin"}, Extensions: []string{".kt", ".kts"}},
	{Display: "Python", Schemes: []string{"scip-python"}, SCIPNames: []string{"python"}, Extensions: []string{".py"}},
	{Display: "Ruby", Schemes: []string{"scip-ruby"}, SCIPNames: []string{"ruby"}, Extensions: []string{".rb"}},
	{Display: "TypeScript", Schemes: []string{"scip-typescript"}, SCIPNames: []string{"typescript"}, Extensions: []string{".ts", ".tsx"}},
	{Display: "JavaScript", SCIPNames: []string{"javascript"}, Extensions: []string{".js", ".jsx"}},
	{Display: "C", SCIPNames: []string{"c"}, Extensions: []string{".c", ".h"}},
	{Display: "C++", Schemes: []string{"scip-clang"}, SCIPNames: []string{"cpp"}, Extensions: []string{".cpp", ".cc", ".cxx", ".hpp", ".hxx"}},
	{Display: "C#", SCIPNames: []string{"csharp"}},
	{Display: "Objective-C", SCIPNames: []string{"objectivec"}, Extensions: []string{".m"}},
	{Display: "Objective-C++", SCIPNames: []string{"objectivecpp"}, Extensions: []string{".mm"}},
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
	return displayLanguage(mostCommon(counts))
}

// inferLanguageFromScheme returns the language for the first parseable
// symbol whose scheme maps to an entry in languageTable. Falls back to
// file-extension inference.
func inferLanguageFromScheme(docs []*scip.Document) string {
	for _, doc := range docs {
		for _, info := range doc.Symbols {
			sym, err := scip.ParseSymbol(info.Symbol)
			if err != nil {
				continue
			}
			if lang := schemeToLanguage(sym.Scheme); lang != "" {
				return lang
			}
		}
	}
	return inferLanguageFromExtension(docs)
}

// inferLanguageFromExtension guesses the language from the most common
// file extension across documents. Last-resort fallback when neither
// Document.Language nor the symbol scheme is informative.
func inferLanguageFromExtension(docs []*scip.Document) string {
	counts := make(map[string]int)
	for _, doc := range docs {
		ext := strings.ToLower(filepath.Ext(doc.RelativePath))
		if lang := extToLanguage(ext); lang != "" {
			counts[lang]++
		}
	}
	return mostCommon(counts)
}

// mostCommon returns the key with the highest count. Ties are broken by
// lexical order so the result is deterministic. Returns "" for empty
// input.
func mostCommon(counts map[string]int) string {
	best, bestN := "", 0
	for k, n := range counts {
		if n > bestN || (n == bestN && k < best) {
			best, bestN = k, n
		}
	}
	return best
}

func extToLanguage(ext string) string {
	for _, e := range languageTable {
		if slices.Contains(e.Extensions, ext) {
			return e.Display
		}
	}
	return ""
}

func schemeToLanguage(scheme string) string {
	for _, e := range languageTable {
		if slices.Contains(e.Schemes, scheme) {
			return e.Display
		}
	}
	return ""
}

// displayLanguage maps SCIP Language enum strings to human-readable names.
// Indexers are inconsistent about case (scip-clang emits "CPP",
// rust-analyzer emits lowercase "rust"), so normalize here. Unknown
// values pass through unchanged.
func displayLanguage(scipLang string) string {
	lower := strings.ToLower(scipLang)
	for _, e := range languageTable {
		if slices.Contains(e.SCIPNames, lower) {
			return e.Display
		}
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
