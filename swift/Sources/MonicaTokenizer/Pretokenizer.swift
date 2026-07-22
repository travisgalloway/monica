// Pretokenizer — splits text into pre-tokens before BPE. stdlib-only (no Foundation,
// no regex engine) so behavior is bit-identical on macOS and Linux.
//
// Scheme (o200k-style, code-oriented; see docs/plan). At each position, first match wins:
//   1. contraction:            '  + (s|t|re|ve|m|ll|d)   (case-insensitive)
//   2. optional leading space + letter run:   ` ?\p{L}+`
//   3. optional leading space + digit run, capped at `digitGroup` (≤3): ` ?\p{N}{1,3}`
//   4. optional leading space + other run:    ` ?[^\s\p{L}\p{N}]+`
//   5. whitespace run: consumed whole at end-of-string, else all-but-last (the last ws
//      char is left so a following word can attach one leading space via rule 2/3/4).
//
// \p{L}/\p{N}/whitespace come from stdlib `Unicode.Scalar.Properties.generalCategory`
// (exact, platform-invariant), with an ASCII fast path for the common (code) case.

public enum Pretokenizer {

    /// Split `text` into pre-tokens, each returned as its raw UTF-8 bytes (BPE operates
    /// on bytes). `digitGroup` caps a digit pre-token's length (3 = o200k-style).
    public static func pretokenize(_ text: String, digitGroup: Int) -> [[UInt8]] {
        let sc = Array(text.unicodeScalars)
        let n = sc.count
        var out: [[UInt8]] = []
        var i = 0
        while i < n {
            let start = i

            // 1. contraction
            if sc[i] == "'", let len = contractionLength(sc, i) {
                i += len
                out.append(bytesOf(sc, start, i)); continue
            }

            // optional single leading space (0x20) for rules 2–4
            let hasSpace = (sc[i] == " ") ? 1 : 0
            let j = i + hasSpace

            if j < n && isLetter(sc[j]) {                       // 2. letters
                i = j
                while i < n && isLetter(sc[i]) { i += 1 }
                out.append(bytesOf(sc, start, i)); continue
            }
            if j < n && isDigit(sc[j]) {                        // 3. digits (≤ digitGroup)
                i = j
                var c = 0
                while i < n && isDigit(sc[i]) && c < digitGroup { i += 1; c += 1 }
                out.append(bytesOf(sc, start, i)); continue
            }
            if j < n && isOther(sc[j]) {                        // 4. other (punctuation/symbols)
                i = j
                while i < n && isOther(sc[i]) { i += 1 }
                out.append(bytesOf(sc, start, i)); continue
            }

            // 5. whitespace run (sc[i] is whitespace here)
            var k = i
            while k < n && isWhitespace(sc[k]) { k += 1 }
            i = (k == n) ? k : max(i + 1, k - 1)                // leave last ws char if a word follows
            out.append(bytesOf(sc, start, i))
        }
        return out
    }

    // MARK: - scalar classification (ASCII fast path, else exact Unicode general category)

    @inline(__always)
    static func isLetter(_ s: Unicode.Scalar) -> Bool {
        let v = s.value
        if v < 128 { return (v | 0x20) >= 97 && (v | 0x20) <= 122 }   // a–z / A–Z
        switch s.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
             .modifierLetter, .otherLetter:
            return true
        default:
            return false
        }
    }

    @inline(__always)
    static func isDigit(_ s: Unicode.Scalar) -> Bool {
        let v = s.value
        if v < 128 { return v >= 48 && v <= 57 }                     // 0–9
        switch s.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber:
            return true
        default:
            return false
        }
    }

    @inline(__always)
    static func isWhitespace(_ s: Unicode.Scalar) -> Bool {
        let v = s.value
        if v < 128 {
            return v == 0x20 || v == 0x09 || v == 0x0A || v == 0x0D || v == 0x0B || v == 0x0C
        }
        return s.properties.isWhitespace
    }

    @inline(__always)
    static func isOther(_ s: Unicode.Scalar) -> Bool {
        return !isWhitespace(s) && !isLetter(s) && !isDigit(s)
    }

    /// If `sc[i]` (== "'") begins a contraction, return its total length (incl. the
    /// apostrophe), else nil. Case-insensitive: 's 't 're 've 'm 'll 'd.
    static func contractionLength(_ sc: [Unicode.Scalar], _ i: Int) -> Int? {
        let n = sc.count
        func low(_ k: Int) -> UInt32? {
            let p = i + k
            guard p < n else { return nil }
            let v = sc[p].value
            return v < 128 ? (v | 0x20) : v
        }
        guard let a = low(1) else { return nil }
        if let b = low(2) {
            // re / ve / ll
            if (a == 114 && b == 101) || (a == 118 && b == 101) || (a == 108 && b == 108) {
                return 3
            }
        }
        if a == 115 || a == 116 || a == 109 || a == 100 { return 2 }  // s / t / m / d
        return nil
    }

    /// Raw UTF-8 bytes of `sc[start..<end]` — no Foundation, no per-scalar String alloc.
    @inline(__always)
    static func bytesOf(_ sc: [Unicode.Scalar], _ start: Int, _ end: Int) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(end - start)
        for k in start..<end {
            UTF8.encode(sc[k]) { out.append($0) }
        }
        return out
    }
}
