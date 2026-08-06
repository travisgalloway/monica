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
import PackageDescription

let package = Package(
    name: "MonicaEngine",
    platforms: [.macOS("14.0")],
    products: [
        .library(name: "MonicaEngine", targets: ["MonicaEngine"]),
        .executable(name: "monica-parity", targets: ["monica-parity"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.6")),
    ],
    targets: [
        .target(
            name: "MonicaEngine",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
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
    ]
)
