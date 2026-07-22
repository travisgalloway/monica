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
    ],
    targets: [
        .target(name: "MonicaTokenizer"),
        .executableTarget(name: "monica-tokenize", dependencies: ["MonicaTokenizer"]),
        // Dependency-free test runner. Runs on macOS (Command Line Tools — no Xcode/XCTest
        // needed) AND Linux, so cross-platform parity is verified the same way on both.
        .executableTarget(name: "monica-selfcheck", dependencies: ["MonicaTokenizer"]),
    ]
)
