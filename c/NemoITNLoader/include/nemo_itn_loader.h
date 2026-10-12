#ifndef FLUIDAUDIO_NEMO_ITN_LOADER_H
#define FLUIDAUDIO_NEMO_ITN_LOADER_H

/// Whether libfluidaudio_nemo_itn.dylib is loaded, loading it on first call.
///
/// The NeMo text-processing engine is a Rust library with its own Rust
/// runtime, so it lives in its own dylib, built by build.rs. This target
/// defines the `nemo_*` C API that FluidAudio's TextNormalizer calls and
/// forwards each call to the dylib. Search order:
///   1. $FLUIDAUDIO_NEMO_ITN_DYLIB, when set (and nothing else);
///   2. @executable_path/../Frameworks/ (app bundles);
///   3. @executable_path/ and @executable_path/../ (cargo targets).
/// Without the dylib every call returns NULL (0 for counts), which
/// TextNormalizer treats as "leave the text unchanged".
int fluidaudio_nemo_itn_available(void);

#endif
