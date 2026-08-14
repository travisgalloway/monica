// swift-tools-version:6.0
// The Swift/MLX inference engine (#166, M13 epic #163) — Apple-only.
//
// WHY THIS IS A SEPARATE PACKAGE from `swift/Package.swift`.
// The tokenizer package (`swift/`, #191/#245) has `dependencies: []` and builds
// bit-identically on macOS AND Linux; the `swift-linux` CI job (#246) therefore needs
// no network for dependencies at all. Adding mlx-swift there — even with
// `condition: .when(platforms: [.macOS])` — would not help: platform conditions apply
// to BUILD EDGES, not to RESOLUTION, so SwiftPM would still clone mlx-swift (which
// vendors the whole `mlx` C++ tree) on Linux, and it would drag the tokenizer's
// tools-version/platform floor up with it. A sibling package keeps `swift/`'s
// zero-dependency, cross-platform property intact — the property that makes the #246
// bit-identity gate cheap and credible.
//
// `swift/engine/` is not inside any target path declared by `swift/Package.swift`, so
// `cd swift && swift build` ignores it exactly the way it already ignores `Fixtures/`.
// Resolves docs/design/14-inference-engine.md's open "Where the Swift engine lives".
//
// TOOLCHAIN. mlx-swift 0.31.6 itself declares `swift-tools-version: 6.3;(experimentalCGen)`,
// so the toolchain that RESOLVES this package must be able to parse that manifest (verified
// on Apple Swift 6.3.3). If a CI image ever ships an older Xcode and resolution fails on the
// manifest version, the fallback is to pin mlx-swift `0.30.6` (tools-version 5.12) here and
// record the reason — not to vendor or work around it.
//
// CROSS-PACKAGE DEPENDENCY DIRECTION (#167). `monica-generate` (below) needs the native code
// tokenizer, so this package gains a PATH dependency on `swift/` (`.package(path: "..")`).
// The edge points engine -> tokenizer, NEVER the reverse: `swift/Package.swift` is untouched
// and stays `dependencies: []`, so `cd swift && swift build` on Linux still resolves nothing.
// Path dependencies are not recorded in `Package.resolved`, so this does not perturb the
// `swift-engine` cache key. Only the `monica-generate` EXECUTABLE gains the tokenizer
// dependency — the `MonicaEngine` library and `monica-parity` stay exactly as coupled as
// before.
import PackageDescription

let package = Package(
    name: "MonicaEngine",
    platforms: [.macOS("14.0")],
    products: [
        .library(name: "MonicaEngine", targets: ["MonicaEngine"]),
        .executable(name: "monica-parity", targets: ["monica-parity"]),
        .executable(name: "monica-generate", targets: ["monica-generate"]),
        .executable(name: "monica-bench", targets: ["monica-bench"]),
        .executable(name: "monica-train", targets: ["monica-train"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.6")),
        .package(path: ".."),   // the zero-dependency tokenizer package (#191/#245)
    ],
    targets: [
        .target(
            name: "MonicaEngine",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                // #195: AdamW for the Swift train step. `LossScaler.swift` deliberately
                // does NOT need this — it is pure-Swift policy, importable and testable
                // even where mlx-swift cannot execute.
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Dependency-free-style runner, mirroring `monica-selfcheck`: no XCTest, no
        // `.testTarget` (the tokenizer package has none either — see ci.yml).
        .executableTarget(
            name: "monica-parity",
            dependencies: [
                "MonicaEngine",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // The generation CLI (#167). Same dependency-free-runner style as monica-parity.
        .executableTarget(
            name: "monica-generate",
            dependencies: [
                "MonicaEngine",
                .product(name: "MLX", package: "mlx-swift"),
                // `package:` is the path dependency's IDENTITY = the directory basename
                // ("swift"), NOT its Package(name:) ("MonicaTokenizer"). Verified empirically
                // (see docs/design/14-inference-engine.md and .claude/plans/issue-167.md):
                // the name form fails with "unknown package … valid packages are: 'swift'".
                .product(name: "MonicaTokenizer", package: "swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // The benchmark harness (#170). Deliberately depends on `MonicaEngine` + `MLX`
        // ONLY — no tokenizer edge, unlike `monica-generate`. The bench modes (prefill/
        // decode/memory) all operate on random token ids (see `Bench.swift`), so there
        // is nothing here for a tokenizer to do; keeping the edge off means
        // `monica-bench` never drags the `swift/` package's build graph in for a run
        // that never calls it.
        .executableTarget(
            name: "monica-bench",
            dependencies: [
                "MonicaEngine",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // The training-step + optimizer parity runner (#195). Same dependency-free-runner
        // style as monica-parity/monica-bench: no tokenizer edge (it trains on the
        // fixtures' own token batches, never text), but does need MLXOptimizers for AdamW.
        .executableTarget(
            name: "monica-train",
            dependencies: [
                "MonicaEngine",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
