// swift-tools-version:5.9
// Native code tokenizer for the Monica model (#191 / M13 #163). Pure Swift, no
// dependencies — builds on macOS (Apple Silicon) AND Linux/x86-64 (the CUDA host)
// with bit-identical output. The BPE core (Pretokenizer/BPE/Trainer) is stdlib-only;
// Foundation is used only for JSON/file I/O (portable subset, present on Linux).
import PackageDescription

let package = Package(
    name: "MonicaTokenizer",
    platforms: [.macOS(.v13)],  // Apple-platform floor only; Linux builds unconstrained.
    products: [
        .library(name: "MonicaTokenizer", targets: ["MonicaTokenizer"]),
        .executable(name: "monica-tokenize", targets: ["monica-tokenize"]),
        .executable(name: "monica-selfcheck", targets: ["monica-selfcheck"]),
        // #197 (M13 #163): the native tsserver/LSP harness + completion-mask trie. Pure
        // Foundation (Process/pipes/JSON/string scanning) — no MLX, no new dependency edge.
        // See MonicaLSP's own file headers for the port-of-what mapping.
        .library(name: "MonicaLSP", targets: ["MonicaLSP"]),
    ],
    targets: [
        .target(name: "MonicaTokenizer"),
        .executableTarget(name: "monica-tokenize", dependencies: ["MonicaTokenizer"]),
        // Dependency-free test runner. Runs on macOS (Command Line Tools — no Xcode/XCTest
        // needed) AND Linux, so cross-platform parity is verified the same way on both.
        .executableTarget(name: "monica-selfcheck", dependencies: ["MonicaTokenizer"]),
        // A new TARGET, not a new package dependency — `dependencies: []` above is
        // untouched, so `swift package show-dependencies` stays empty and swift-linux keeps
        // needing no network. No MonicaTokenizer edge either: MonicaLSP's decode/encode
        // closures are supplied by the caller, so this target has no tokenizer dependency.
        .target(name: "MonicaLSP"),
        // Dependency-free-runner CLI (mirrors monica-selfcheck): --self-test (binary-free,
        // both platforms), --emit-mask-parity, --probe-reap, --bench (macOS CI only, need
        // node_modules/typescript-language-server).
        .executableTarget(name: "monica-lsp", dependencies: ["MonicaLSP"]),
    ]
)
