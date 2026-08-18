// SourceScan.swift — port of `src/lsp/diagnostics.py`'s string/template/comment-aware
// delimiter scanner (`_scan`/`mask_strings_and_comments`) and identifier-char predicates, so
// a `.` inside a string or comment never opens a completion-mask span. Operates over
// `[Character]` (Swift grapheme clusters) rather than Python's code-point `str` indices — a
// documented, harmless divergence for TS source, which is ASCII-dominant at every position
// this scanner inspects (operators, quotes, brackets); the residual only matters for
// multi-scalar graphemes appearing *inside* string/comment content, which this scanner masks
// out regardless of their internal structure.

import Foundation

public enum SourceScan {
    /// `[A-Za-z0-9_$]` — mirrors `_lsp/completion_mask.py`'s `_IDENT_CHAR_RE` exactly
    /// (ASCII only, not `Character.isLetter`/`isNumber`'s Unicode-wide notion of "letter").
    public static func isIdentChar(_ c: Character) -> Bool {
        guard let s = c.asciiValue else { return false }
        return (s >= 48 && s <= 57) || (s >= 65 && s <= 90) || (s >= 97 && s <= 122)
            || s == 0x5F /* _ */ || s == 0x24 /* $ */
    }

    /// `[A-Za-z_$]` — mirrors `_IDENT_START_RE`.
    public static func isIdentStart(_ c: Character) -> Bool {
        guard let s = c.asciiValue else { return false }
        return (s >= 65 && s <= 90) || (s >= 97 && s <= 122) || s == 0x5F || s == 0x24
    }

    private static let pairs: [Character: Character] = ["(": ")", "[": "]", "{": "}"]
    private static let closers: Set<Character> = [")", "]", "}"]
    private static let quotes: Set<Character> = ["'", "\"", "`"]
    private static let dangerTokens: Set<String> = ["'", "\"", "`", "/*", "//"]

    /// Blank out every character living inside a string/template literal or a comment,
    /// replacing it with a space — preserves `text.count` and every newline position, so
    /// offsets computed against the masked text stay meaningful against `text` itself. A
    /// character is masked if the open-delimiter stack was in a "danger" state either before
    /// or after it was consumed, covering both the opening and closing delimiter of a
    /// string/comment. `${...}` interpolations inside a template literal are NOT masked —
    /// they are live code, not string content.
    ///
    /// Known residual (matches the Python reference): a two-character token (`\x` escape,
    /// `${`, `*/`, the opening `//`) only advances the scan by one visited index — the
    /// second character of such a token is never independently visited and can survive
    /// unmasked as stray punctuation. Documented, not fixed (see the Python docstring).
    public static func maskStringsAndComments(_ text: String) -> String {
        let chars = Array(text)
        let n = chars.count
        var stack: [String] = []
        var danger = [Bool](repeating: false, count: n)
        var prevTop: String? = nil

        var i = 0
        while i < n {
            let c = chars[i]
            let top = stack.last
            var consumed = 1

            if top == "'" || top == "\"" {
                if c == "\\" && i + 1 < n {
                    consumed = 2
                } else if String(c) == top {
                    stack.removeLast()
                }
            } else if top == "`" {
                if c == "\\" && i + 1 < n {
                    consumed = 2
                } else if c == "`" {
                    stack.removeLast()
                } else if c == "$" && i + 1 < n && chars[i + 1] == "{" {
                    stack.append("}")
                    consumed = 2
                }
            } else if top == "/*" {
                if c == "*" && i + 1 < n && chars[i + 1] == "/" {
                    stack.removeLast()
                    consumed = 2
                }
            } else if top == "//" {
                if c == "\n" {
                    stack.removeLast()
                }
            } else {
                // "Real code" context: stack empty, or top is a bracket closer.
                if c == "/" && i + 1 < n && chars[i + 1] == "/" {
                    stack.append("//")
                    consumed = 2
                } else if c == "/" && i + 1 < n && chars[i + 1] == "*" {
                    stack.append("/*")
                    consumed = 2
                } else if quotes.contains(c) {
                    stack.append(String(c))
                } else if let close = pairs[c] {
                    stack.append(String(close))
                } else if closers.contains(c) {
                    if let last = stack.last, last == String(c) { stack.removeLast() }
                }
            }

            let newTop = stack.last
            if c != "\n" {
                let prevDanger = prevTop != nil && dangerTokens.contains(prevTop!)
                let newDanger = newTop != nil && dangerTokens.contains(newTop!)
                danger[i] = prevDanger || newDanger
            }
            prevTop = newTop
            i += consumed
        }

        var out = chars
        for idx in 0..<n where danger[idx] { out[idx] = " " }
        return String(out)
    }
}
