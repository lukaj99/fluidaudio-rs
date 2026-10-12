//! Integration tests for the Swift→Rust FFI bindings.
//!
//! These tests exercise the FFI surface without requiring large model downloads
//! or network access. They verify:
//!   * Bridge lifecycle (create / drop / cleanup) does not crash
//!   * Pre-init availability checks return `false` instead of panicking
//!   * Strings allocated in Swift survive the round trip into Rust
//!   * Error paths (missing files) are reported instead of segfaulting
//!
//! Tests that require network downloads or compiled CoreML models are guarded
//! with `#[ignore]` and can be run via:
//!     cargo test --test ffi_bindings -- --ignored

use fluidaudio_rs::{FluidAudio, FluidAudioError};

/// The bridge can be created and dropped without panicking. Drop must not
/// crash even if no `init_*` method was ever called.
#[test]
fn bridge_create_and_drop_is_safe() {
    let audio = FluidAudio::new().expect("bridge creation should succeed");
    drop(audio);
}

/// Multiple bridges can coexist. Each Rust handle owns its own retained
/// instance, so creating a second bridge must not invalidate the first.
#[test]
fn multiple_bridges_can_be_created() {
    let a = FluidAudio::new().expect("first bridge");
    let b = FluidAudio::new().expect("second bridge");

    // Both should report consistent system information; that proves both
    // pointers are still valid and the FFI calls succeed on each.
    let info_a = a.system_info();
    let info_b = b.system_info();
    assert_eq!(info_a.platform, info_b.platform);
    assert_eq!(info_a.is_apple_silicon, info_b.is_apple_silicon);
}

/// Bridges created and dropped on several threads at once must not interfere.
/// The Swift side once assigned every new bridge to an unsynchronized global,
/// and concurrent assignments over-released bridges (SIGSEGV in deinit).
#[test]
fn bridges_can_be_created_and_dropped_concurrently() {
    let threads: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..200 {
                    drop(FluidAudio::new().expect("bridge creation"));
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().expect("bridge thread panicked");
    }
}

/// `cleanup` must be idempotent. The destructor calls it again on drop, so
/// calling it explicitly first must not double-free.
#[test]
fn cleanup_is_idempotent() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.cleanup();
    audio.cleanup();
    // Drop runs cleanup once more; should still be safe.
}

/// Before any `init_*` call, every `is_*_available` accessor must return
/// `false` rather than reading uninitialized state.
#[test]
fn availability_is_false_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(!audio.is_asr_available(), "ASR should be unavailable pre-init");
    assert!(
        !audio.is_streaming_asr_available(),
        "streaming ASR should be unavailable pre-init"
    );
    assert!(!audio.is_vad_available(), "VAD should be unavailable pre-init");
    assert!(
        !audio.is_diarization_available(),
        "diarization should be unavailable pre-init"
    );
}

/// `system_info()` exercises Swift→Rust string ownership: the Swift side
/// allocates with `strdup`, hands a pointer to Rust, and Rust frees it via
/// `fluidaudio_free_string`. The platform string must come back populated.
#[test]
fn system_info_round_trips_swift_strings() {
    let audio = FluidAudio::new().expect("bridge creation");
    let info = audio.system_info();

    assert!(
        info.platform == "macOS" || info.platform == "iOS",
        "unexpected platform: {}",
        info.platform
    );
    assert!(!info.chip_name.is_empty(), "chip name should be populated");
    assert!(info.memory_gb > 0.0, "memory should be reported as positive");
}

/// `is_apple_silicon()` reads `SystemInfo.isAppleSilicon` from Swift and
/// must return a value consistent with `system_info()`.
#[test]
fn is_apple_silicon_matches_system_info() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert_eq!(audio.is_apple_silicon(), audio.system_info().is_apple_silicon);
}

/// `is_intel_mac()` and `is_apple_silicon()` must be mutually exclusive: a
/// process is either arm64 or x86_64, never both, never neither (for the
/// architectures FluidAudio supports).
#[test]
fn is_intel_mac_and_is_apple_silicon_are_exclusive() {
    let audio = FluidAudio::new().expect("bridge creation");
    let arm = audio.is_apple_silicon();
    let intel = audio.is_intel_mac();
    assert!(
        arm != intel,
        "expected exactly one of (apple_silicon, intel_mac) to be true, got both={arm} and intel={intel}"
    );
}

/// File-based methods must validate the path before crossing the FFI boundary,
/// so a missing file returns `FileNotFound` rather than panicking inside Swift.
#[test]
fn missing_file_returns_file_not_found() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .transcribe_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("transcribe_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));

    let err = audio
        .transcribe_file_streaming("/this/path/definitely/does/not/exist.wav")
        .expect_err("transcribe_file_streaming with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));

    let err = audio
        .diarize_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("diarize_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));
}

/// Calling streaming ASR session methods before initializing must surface a
/// bridge error rather than panicking. We don't assert on the message because
/// it depends on the Swift error path; we just want to confirm we get back a
/// `Result::Err` cleanly across the FFI.
#[test]
fn streaming_asr_session_methods_error_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(audio.streaming_asr_start().is_err());
    assert!(audio.streaming_asr_feed(&[0.0_f32; 1600]).is_err());
    assert!(audio.streaming_asr_finish().is_err());
}

/// Same for the transcription calls: they must fail cleanly before their
/// engine is loaded. The fixture exists, so the file-based calls get past the
/// Rust-side path check and reach the Swift side.
#[test]
fn asr_transcription_errors_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hello.wav");
    assert!(audio.transcribe_samples(&[0.0_f32; 16_000]).is_err());
    assert!(audio.transcribe_file(&fixture).is_err());
    assert!(audio.transcribe_file_streaming(&fixture).is_err());
}

/// Same as above for the Nemotron streaming session methods. `reset` is
/// included because the streaming feeder calls it on teardown, so it must fail
/// cleanly on an uninitialized bridge rather than trapping.
#[test]
fn nemotron_streaming_session_methods_error_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(!audio.is_nemotron_streaming_available());
    assert!(audio.nemotron_streaming_process(&[0.0_f32; 1600]).is_err());
    assert!(audio.nemotron_streaming_get_partial().is_err());
    assert!(audio.nemotron_streaming_finish().is_err());
    assert!(audio.nemotron_streaming_reset().is_err());
}

/// VAD `process` methods must error before `init_vad` rather than crash.
#[test]
fn vad_process_errors_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(audio.vad_process_samples(&[0.0_f32; 4096]).is_err());
    assert!(audio
        .vad_process_file("/this/path/definitely/does/not/exist.wav")
        .is_err());
}

/// VAD `process_file` validates the path on the Rust side and returns
/// `FileNotFound` (not a Swift-side error) when the path doesn't exist.
#[test]
fn vad_process_file_returns_file_not_found() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .vad_process_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("vad_process_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));
}

/// ITN does not require model loading — `TextNormalizer.shared` is always
/// available. Calling `itn_normalize` on a fresh bridge should round-trip a
/// string from Swift back to Rust without crashing.
#[test]
fn itn_normalize_round_trips() {
    let audio = FluidAudio::new().expect("bridge creation");
    // Use a string that won't be modified by ITN regardless of which backend
    // is active. We only care that we got a String back, not that it matches
    // a specific normalized form.
    let result = audio.itn_normalize("hello").expect("itn_normalize");
    assert!(!result.is_empty(), "ITN result should not be empty");
}

/// Sentence-mode ITN must also work without init and accept arbitrary text.
#[test]
fn itn_normalize_sentence_round_trips() {
    let audio = FluidAudio::new().expect("bridge creation");
    let result = audio
        .itn_normalize_sentence("the quick brown fox")
        .expect("itn_normalize_sentence");
    assert!(!result.is_empty());

    let result = audio
        .itn_normalize_sentence_max_span("five plus five", 8)
        .expect("itn_normalize_sentence_max_span");
    assert!(!result.is_empty());
}

/// `itn_is_native_available()` must return a definite bool without crashing,
/// regardless of whether the native NeMo library is loaded in this process.
#[test]
fn itn_is_native_available_returns_bool() {
    let audio = FluidAudio::new().expect("bridge creation");
    let _ = audio.itn_is_native_available();
}

// ---------------------------------------------------------------------------
// Heavier tests below: gated behind `--ignored` because they download models
// from HuggingFace and warm up the Apple Neural Engine (cold start ~20s).
// ---------------------------------------------------------------------------

/// VAD initialization downloads a small CoreML model on first run.
#[test]
#[ignore = "downloads VAD model from HuggingFace on first run"]
fn vad_initializes() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_vad(0.85).expect("VAD init");
    assert!(audio.is_vad_available());
}

/// End-to-end VAD on a silence buffer. We don't assert classification
/// (which depends on the threshold) — only that we get one frame per 4096
/// samples and the per-frame fields look sane.
#[test]
#[ignore = "downloads VAD model from HuggingFace on first run"]
fn vad_processes_silence_buffer() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_vad(0.85).expect("VAD init");

    // 2 seconds of silence at 16kHz mono = 32_000 samples = ~7.8 chunks of 4096.
    // The Swift side pads the trailing partial chunk, so expect ceil(32000/4096) = 8.
    let samples = vec![0.0_f32; 16_000 * 2];
    let frames = audio.vad_process_samples(&samples).expect("vad process");

    assert_eq!(frames.len(), 8, "expected 8 chunks for 2s at 16kHz");
    for frame in &frames {
        assert!(frame.probability >= 0.0 && frame.probability <= 1.0);
        assert!(frame.processing_time >= 0.0);
    }
}

/// End-to-end ASR sanity check on a buffer of silence. We don't assert on the
/// transcript content, only that the call returns successfully, the reported
/// duration matches the input length, and confidence is on FluidAudio's TDT
/// scale (0.1 for an empty transcript, otherwise within [0.1, 1.0]).
#[test]
#[ignore = "downloads Parakeet Unified EN 0.6B (~614 MB) and triggers ANE compilation"]
fn asr_transcribes_silence_buffer() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_asr().expect("ASR init");
    assert!(audio.is_asr_available());

    // 2 seconds of silence at 16kHz mono.
    let samples = vec![0.0_f32; 16_000 * 2];
    let result = audio.transcribe_samples(&samples).expect("transcribe");
    assert!(result.duration >= 1.9 && result.duration <= 2.1);
    if result.text.trim().is_empty() {
        assert_eq!(result.confidence, 0.1);
    } else {
        assert!(
            (0.1..=1.0).contains(&result.confidence),
            "confidence {}",
            result.confidence
        );
    }
}

/// Regression test: consecutive one-shot calls must not share decoder state.
/// The bridge once kept the TDT decoder's LSTM hidden/cell tensors and
/// `lastToken` between calls, which biased the next call's predictor with the
/// previous call's end-of-utterance state and collapsed later transcripts to a
/// lone "." (the previous call's terminal punctuation). The Unified engine
/// decodes every window from a fresh RNNT state; this keeps it that way.
///
/// We transcribe a short shipped WAV fixture twice in a row and assert:
///
///   1. The first transcript is non-empty and matches the spoken phrase
///      (sanity: the model is actually running).
///   2. The second transcript equals the first byte-for-byte.
///
/// Under the bug, (1) holds but (2) fails — the second result is some
/// degenerate suffix of the first (typically just ".").
///
/// The fixture (`tests/fixtures/hello.wav`) was generated on macOS with:
///
///     say -v Samantha --file-format=WAVE --data-format=LEI16@16000 \
///         -o tests/fixtures/hello.wav "Hello world, this is a test."
///
/// Apple's TTS output isn't subject to third-party licensing; regenerating
/// it on any macOS host produces a comparable clip.
#[test]
#[ignore = "downloads Parakeet Unified EN 0.6B (~614 MB) and triggers ANE compilation"]
fn asr_transcribe_file_is_stateless_across_calls() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_asr().expect("ASR init");

    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hello.wav");
    assert!(
        fixture.exists(),
        "fixture not found at {:?}",
        fixture
    );

    let first = audio.transcribe_file(&fixture).expect("first transcribe");
    let second = audio.transcribe_file(&fixture).expect("second transcribe");

    // Sanity: the model produced something for our spoken phrase. We don't
    // pin the exact transcript (model versions and TTS voices vary) — just
    // that it ran.
    let first_lower = first.text.to_lowercase();
    assert!(
        first_lower.contains("hello") || first_lower.contains("test"),
        "first transcript should reference the spoken phrase, got {:?}",
        first.text,
    );
    assert!(
        (0.1..=1.0).contains(&first.confidence),
        "confidence {}",
        first.confidence
    );

    // The actual regression assertion.
    assert_eq!(
        first.text, second.text,
        "consecutive transcribe_file calls must yield identical output on \
         identical input; got {:?} then {:?}. Decoder state is leaking \
         between calls.",
        first.text, second.text,
    );
}

/// Streaming sessions share one loaded engine. `is_streaming_asr_available`
/// may be polled from another thread while the model loads; feeds return only
/// new, non-empty partials; and a second session over the same audio yields
/// the same final text, so no state carries over between sessions.
#[test]
#[ignore = "downloads the Parakeet Unified EN 0.6B streaming encoder (~592 MB) and triggers ANE compilation"]
fn streaming_asr_sessions_share_one_engine() {
    let audio = std::sync::Arc::new(FluidAudio::new().expect("bridge creation"));

    let poller = {
        let audio = std::sync::Arc::clone(&audio);
        std::thread::spawn(move || {
            while !audio.is_streaming_asr_available() {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        })
    };
    audio.init_streaming_asr().expect("streaming ASR init");
    poller.join().expect("poller thread");

    // hello.wav lasts 1.9 s. Trailing silence lets the engine (which decodes
    // once 2.08 s are buffered) emit a partial before finish.
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hello.wav");
    let mut samples = read_wav_samples(&fixture);
    samples.resize(samples.len() + 24_000, 0.0);

    let run_session = || {
        audio.streaming_asr_start().expect("start");
        let mut partials: Vec<String> = Vec::new();
        for chunk in samples.chunks(1_600) {
            if let Some(partial) = audio.streaming_asr_feed(chunk).expect("feed") {
                assert!(!partial.is_empty());
                assert_ne!(partials.last(), Some(&partial), "feed repeated a partial");
                partials.push(partial);
            }
        }
        (partials, audio.streaming_asr_finish().expect("finish"))
    };

    let (partials, first) = run_session();
    let (_, second) = run_session();

    assert!(
        !partials.is_empty(),
        "expected a live partial before finish"
    );
    let first_lower = first.to_lowercase();
    assert!(
        first_lower.contains("hello") || first_lower.contains("test"),
        "final transcript should reference the spoken phrase, got {first:?}",
    );
    assert_eq!(
        first, second,
        "the second session must not carry state from the first"
    );
}

/// 16-bit PCM samples from a 16kHz mono WAV, scaled to [-1, 1]. Walks the RIFF
/// chunks to `data`: the fixture carries an `FLLR` padding chunk before it.
fn read_wav_samples(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &bytes[pos..pos + 4] == b"data" {
            return bytes[pos + 8..pos + 8 + size]
                .chunks_exact(2)
                .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
                .collect();
        }
        pos += 8 + size + (size & 1);
    }
    panic!("no data chunk in {path:?}");
}
