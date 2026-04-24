package scipextract

import (
	"fmt"
	"os"
	"strings"

	"github.com/scip-code/scip/bindings/go/scip"
)

// sourceCache caches file contents keyed by absolute path.
type sourceCache struct {
	root  string
	files map[string][]byte
}

func newSourceCache(root string) *sourceCache {
	return &sourceCache{root: root, files: map[string][]byte{}}
}

func (c *sourceCache) get(absPath string) []byte {
	if content, ok := c.files[absPath]; ok {
		return content
	}
	content, err := os.ReadFile(absPath)
	if err != nil {
		c.files[absPath] = nil
		return nil
	}
	c.files[absPath] = content
	return content
}

// extractBody returns the source text for a function's body and an
// optional warning message describing any fallback that was taken.
//
// docRanges is the per-document map produced by definitionRanges — it
// carries either the SCIP EnclosingRange (when the indexer emits one) or a
// heuristic span-to-next-function range. The warning is non-empty when we
// had to rely on the heuristic.
func extractBody(info *scip.SymbolInformation, doc *scip.Document, absPath string, docRanges map[string]scip.Range, cache *sourceCache) (string, string) {
	r, ok := docRanges[info.Symbol]
	if !ok {
		return "", fmt.Sprintf("no definition range for %s", info.Symbol)
	}

	defOcc := findDefinition(info.Symbol, doc)
	synthetic := defOcc == nil || len(defOcc.EnclosingRange) == 0

	body, ok := sliceRange(r, doc, absPath, cache)
	if !ok {
		return "", fmt.Sprintf("failed to read source for %s at %s", info.Symbol, absPath)
	}
	if synthetic {
		return body, fmt.Sprintf("no enclosing range for %s; extracted by span-to-next-function heuristic", info.Symbol)
	}
	return body, ""
}

// findDefinition returns the first occurrence of symbol in doc with the
// Definition role set and a well-formed Range. Occurrences with malformed
// Ranges are skipped so callers can rely on scip.NewRange(occ.Range)
// succeeding.
func findDefinition(symbol string, doc *scip.Document) *scip.Occurrence {
	for _, occ := range doc.Occurrences {
		if occ.Symbol != symbol {
			continue
		}
		if occ.SymbolRoles&int32(scip.SymbolRole_Definition) == 0 {
			continue
		}
		if _, err := scip.NewRange(occ.Range); err != nil {
			continue
		}
		return occ
	}
	return nil
}

// sliceRange extracts the text covered by r. It prefers Document.Text when
// the indexer embedded it; otherwise it reads from disk via the cache.
// Returns (text, true) on success. SCIP ranges are half-open, so an End
// position with Character == 0 excludes End.Line from the slice.
func sliceRange(r scip.Range, doc *scip.Document, absPath string, cache *sourceCache) (string, bool) {
	if doc.Text != "" {
		if s, ok := sliceLinesHalfOpen(doc.Text, r); ok {
			return s, true
		}
	}

	content := cache.get(absPath)
	if content == nil {
		return "", false
	}
	return sliceLinesHalfOpen(string(content), r)
}

// sliceLinesHalfOpen returns the substring covering r using line-level
// half-open semantics: lines [r.Start.Line .. r.End.Line] are included when
// r.End.Character > 0; otherwise r.End.Line is excluded.
func sliceLinesHalfOpen(text string, r scip.Range) (string, bool) {
	lines := strings.SplitAfter(text, "\n")
	startLine := int(r.Start.Line)
	if startLine < 0 || startLine >= len(lines) {
		return "", false
	}
	endLine := int(r.End.Line)
	if r.End.Character == 0 {
		endLine--
	}
	if endLine >= len(lines) {
		endLine = len(lines) - 1
	}
	if endLine < startLine {
		endLine = startLine
	}
	var b strings.Builder
	for i := startLine; i <= endLine; i++ {
		b.WriteString(lines[i])
	}
	return strings.TrimRight(b.String(), "\n"), true
}
