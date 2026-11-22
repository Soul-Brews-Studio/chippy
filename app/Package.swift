// swift-tools-version:5.9
import PackageDescription

// The tokenizer is HF `tokenizers` (Rust, the library Python's fast tokenizer uses), built
// as a static library first:  cargo build --release --manifest-path tokenizer-ffi/Cargo.toml
let ffiLib = Context.packageDirectory + "/tokenizer-ffi/target/release"

let package = Package(
    name: "ANEEmbed",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "ane-embed", targets: ["ane-embed"]),
        .executable(name: "ANEEmbedBar", targets: ["ANEEmbedBar"]),
    ],
    targets: [
        .systemLibrary(name: "CTokenizerFFI", path: "tokenizer-ffi/include"),
        .target(name: "ANEEmbedCore", dependencies: ["CTokenizerFFI"],
                linkerSettings: [.unsafeFlags(["-L", ffiLib])]),
        .target(name: "ANEEmbedHTTP", dependencies: ["ANEEmbedCore"]),
        .executableTarget(name: "ane-embed", dependencies: ["ANEEmbedCore", "ANEEmbedHTTP"]),
        .executableTarget(name: "ANEEmbedBar", dependencies: ["ANEEmbedCore", "ANEEmbedHTTP"]),
        .testTarget(name: "ANEEmbedCoreTests", dependencies: ["ANEEmbedCore"]),
    ]
)
