// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "FluidAudioBridge",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "FluidAudioBridge",
            type: .static,
            targets: ["FluidAudioBridge"]
        ),
    ],
    dependencies: [
        // `traits: []` opts out of FluidAudio's default NemoTextProcessing trait, a
        // prebuilt Rust staticlib that would put a second Rust runtime (duplicate
        // `_rust_eh_personality`) into any Rust binary linking this crate.
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.17.7", traits: []),
    ],
    targets: [
        .target(
            name: "FluidAudioBridge",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "swift"
        ),
    ],
    // The bridge's blocking semaphore wrappers predate strict concurrency.
    swiftLanguageModes: [.v5]
)
