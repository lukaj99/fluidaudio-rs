use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Tell Cargo to rerun if Swift files change
    println!("cargo:rerun-if-changed=swift/");
    println!("cargo:rerun-if-changed=c/");
    println!("cargo:rerun-if-changed=Package.swift");
    println!("cargo:rerun-if-changed=nemo_itn.exports");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // Build the Swift package first to get FluidAudio dependency
    println!("cargo:warning=Building Swift package...");

    let swift_build_dir = out_dir.join("swift-build");
    std::fs::create_dir_all(&swift_build_dir).expect("Failed to create swift-build directory");

    // Build Swift package in release mode
    let status = Command::new("swift")
        .args([
            "build",
            "-c",
            "release",
            "--build-path",
            swift_build_dir.to_str().unwrap(),
        ])
        .current_dir(&manifest_dir)
        .status()
        .expect("Failed to run swift build");

    if !status.success() {
        panic!("Swift package build failed");
    }

    // Link the Swift library; FluidAudio's NeMo ITN engine goes into its own dylib
    let link_dir = split_out_nemo_itn(&swift_build_dir.join("release"), &out_dir, &manifest_dir);
    println!("cargo:rustc-link-search=native={}", link_dir.display());
    println!("cargo:rustc-link-lib=static=FluidAudioBridge");

    // Link Apple frameworks
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // Link Swift runtime
    println!("cargo:rustc-link-lib=dylib=swiftCore");

    // Link C++ standard library (needed for FastClusterWrapper.cpp in FluidAudio)
    println!("cargo:rustc-link-lib=c++");
}

/// Name of the dylib holding FluidAudio's NeMo ITN engine.
const NEMO_ITN_DYLIB: &str = "libfluidaudio_nemo_itn.dylib";

/// Prepare `OUT_DIR/link/libFluidAudioBridge.a` without FluidAudio's NeMo ITN
/// engine, build the engine as `libfluidaudio_nemo_itn.dylib`, and return the
/// link directory.
///
/// The NemoTextProcessing trait links text-processing-rs, a prebuilt Rust
/// staticlib (`libtext_processing_rs.a`) with its own Rust std, and SwiftPM
/// copies its members into `libFluidAudioBridge.a`. In the Rust binary that
/// links this crate they clash with that binary's std: duplicate globals
/// (`rust_eh_personality`, `__rust_alloc`, ...), and, even with those made
/// local, a second `rust_eh_personality`, one more personality routine than
/// compact unwind can encode once C++ and Objective-C code share the image.
/// So the members leave the bridge archive and become a dylib exporting only
/// `nemo_*` (`nemo_itn.exports`), with its own runtime and unwind tables. The
/// NemoITNLoader target defines the `nemo_*` functions FluidAudio calls and
/// forwards them to the dylib, loaded at first use (see its header for the
/// search order).
///
/// The dylib is copied to the profile directory (`target/<profile>/`), where
/// this crate's tests and examples find it at `@executable_path/../` and from
/// where an app copies it into `Contents/Frameworks`.
fn split_out_nemo_itn(swift_lib_dir: &Path, out_dir: &Path, manifest_dir: &Path) -> PathBuf {
    let engine = swift_lib_dir.join("libtext_processing_rs.a");
    assert!(
        engine.exists(),
        "{} is missing: FluidAudio's NemoTextProcessing trait must be enabled in Package.swift",
        engine.display()
    );
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "aarch64" => "arm64",
        "x86_64" => "x86_64",
        other => panic!("unsupported target architecture for NeMo ITN: {other}"),
    };
    let link_dir = out_dir.join("link");
    let _ = std::fs::remove_dir_all(&link_dir);
    std::fs::create_dir_all(&link_dir).unwrap();

    // The prebuilt macOS slice is universal; ar wants one architecture.
    let archs = capture(Command::new("xcrun").arg("lipo").arg("-archs").arg(&engine));
    let engine = if archs.split_whitespace().count() > 1 {
        let thin = out_dir.join("text_processing_rs-thin.a");
        run(Command::new("xcrun")
            .arg("lipo")
            .arg(&engine)
            .args(["-thin", arch, "-output"])
            .arg(&thin));
        thin
    } else {
        engine
    };

    // libFluidAudioBridge.a without the engine's members (their names are unique).
    let engine_members: std::collections::HashSet<String> =
        capture(Command::new("ar").arg("t").arg(&engine))
            .lines()
            .map(str::to_string)
            .collect();
    let objects = out_dir.join("bridge-objects");
    let _ = std::fs::remove_dir_all(&objects);
    std::fs::create_dir_all(&objects).unwrap();
    run(Command::new("ar")
        .arg("x")
        .arg(swift_lib_dir.join("libFluidAudioBridge.a"))
        .current_dir(&objects));
    let mut kept: Vec<String> = std::fs::read_dir(&objects)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| !name.starts_with("__.SYMDEF") && !engine_members.contains(name))
        .collect();
    kept.sort();
    let file_list = out_dir.join("bridge-objects.txt");
    std::fs::write(&file_list, kept.join("\n")).unwrap();
    run(Command::new("xcrun")
        .args(["libtool", "-static", "-no_warning_for_no_symbols", "-o"])
        .arg(link_dir.join("libFluidAudioBridge.a"))
        .arg("-filelist")
        .arg(format!("{},{}", file_list.display(), objects.display())));

    // The engine as a dylib with only its C API exported.
    let deployment_target =
        std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".to_string());
    let dylib = out_dir.join(NEMO_ITN_DYLIB);
    run(Command::new("xcrun")
        .args(["clang", "-dynamiclib", "-arch", arch])
        .arg(format!("-mmacosx-version-min={deployment_target}"))
        .arg(format!("-Wl,-install_name,@rpath/{NEMO_ITN_DYLIB}"))
        .arg(format!(
            "-Wl,-exported_symbols_list,{}",
            manifest_dir.join("nemo_itn.exports").display()
        ))
        .arg(format!("-Wl,-force_load,{}", engine.display()))
        .arg("-Wl,-dead_strip")
        .arg("-o")
        .arg(&dylib));
    // out_dir is <target>/<profile>/build/<package>-<hash>/out. Copy, then
    // rename over the old file: rewriting a signed dylib in place can get a
    // process that has it mapped killed.
    let profile_dir = out_dir.ancestors().nth(3).unwrap();
    let staged = profile_dir.join(format!("{NEMO_ITN_DYLIB}.tmp"));
    std::fs::copy(&dylib, &staged).unwrap();
    std::fs::rename(&staged, profile_dir.join(NEMO_ITN_DYLIB)).unwrap();

    link_dir
}

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|e| panic!("failed to run {command:?}: {e}"));
    assert!(status.success(), "{command:?} failed with {status}");
}

fn capture(command: &mut Command) -> String {
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("failed to run {command:?}: {e}"));
    assert!(
        output.status.success(),
        "{command:?} failed with {}",
        output.status
    );
    String::from_utf8(output.stdout).expect("non-UTF-8 tool output")
}
