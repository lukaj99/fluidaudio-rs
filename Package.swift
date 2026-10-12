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
        // The default NemoTextProcessing trait is on, so FluidAudio's
        // TextNormalizer calls the NeMo ITN C API. The engine behind it is a
        // prebuilt Rust staticlib with its own Rust runtime, which cannot share
        // an image with the Rust binary that links us: build.rs moves it into
        // libfluidaudio_nemo_itn.dylib, and NemoITNLoader forwards the calls.
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.17.7"),
    ],
    targets: [
        .target(
            name: "FluidAudioBridge",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
                "NemoITNLoader",
            ],
            path: "swift"
        ),
        .target(
            name: "NemoITNLoader",
            path: "c/NemoITNLoader",
            publicHeadersPath: "include"
        ),
    ],
    // The bridge's blocking semaphore wrappers predate strict concurrency.
    swiftLanguageModes: [.v5]
)
