//! # fluidaudio-rs
//!
//! Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) -
//! a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.
//!
//! ## Features
//!
//! - **ASR (Automatic Speech Recognition)** - English speech-to-text with NVIDIA
//!   Parakeet Unified EN 0.6B, one-shot or streaming with live partial transcripts
//! - **VAD (Voice Activity Detection)** - Detect speech segments in audio
//! - **Speaker Diarization** - Identify and label different speakers in audio
//!
//! ## Requirements
//!
//! - macOS 14+ or iOS 17+
//! - Apple Silicon (M1/M2/M3) recommended
//!
//! ## Example
//!
//! ```rust,no_run
//! use fluidaudio_rs::FluidAudio;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = FluidAudio::new()?;
//!
//!     // Transcribe an audio file
//!     audio.init_asr()?;
//!     let result = audio.transcribe_file("audio.wav")?;
//!     println!("Text: {}", result.text);
//!     println!("Confidence: {:.2}%", result.confidence * 100.0);
//!
//!     Ok(())
//! }
//! ```

mod ffi;

use std::path::Path;
use thiserror::Error;

// Re-export FFI types
pub use ffi::{AsrResult, DiarizationSegment, SystemInfo, VadFrame};

/// Errors that can occur when using FluidAudio
#[derive(Error, Debug)]
pub enum FluidAudioError {
    #[error("FluidAudio not initialized: {0}")]
    NotInitialized(String),

    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Audio file not found: {0}")]
    FileNotFound(String),

    #[error("Swift bridge error: {0}")]
    BridgeError(String),
}

impl From<String> for FluidAudioError {
    fn from(s: String) -> Self {
        FluidAudioError::BridgeError(s)
    }
}

/// Main FluidAudio interface for Rust
///
/// Provides access to ASR and VAD functionality.
pub struct FluidAudio {
    bridge: ffi::FluidAudioBridge,
}

impl FluidAudio {
    /// Create a new FluidAudio instance
    pub fn new() -> Result<Self, FluidAudioError> {
        let bridge = ffi::FluidAudioBridge::new()
            .ok_or_else(|| FluidAudioError::BridgeError("Failed to create bridge".to_string()))?;
        Ok(Self { bridge })
    }

    // ========== ASR Methods ==========

    /// Initialize the ASR (Automatic Speech Recognition) engine
    ///
    /// Loads NVIDIA Parakeet Unified EN 0.6B (FastConformer-RNNT, English) for
    /// one-shot transcription. The first run downloads the offline model set
    /// (~614 MB) into `~/Library/Application Support/FluidAudio/Models/parakeet-unified-en-0.6b`,
    /// pinned to a fixed Hugging Face revision, and compiles it for the Neural
    /// Engine; later runs load from that cache. Blocks until loaded.
    pub fn init_asr(&self) -> Result<(), FluidAudioError> {
        self.bridge.initialize_asr().map_err(FluidAudioError::from)
    }

    /// Transcribe an audio file
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    pub fn transcribe_file<P: AsRef<Path>>(&self, path: P) -> Result<AsrResult, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .transcribe_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Transcribe audio samples directly
    ///
    /// This method accepts raw 16kHz mono audio samples, making it ideal for
    /// real-time audio applications where audio is captured from a microphone
    /// or other streaming source.
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_asr()?;
    ///
    ///     // Simulated audio buffer (16kHz mono)
    ///     let samples: Vec<f32> = vec![0.0; 16000]; // 1 second of silence
    ///
    ///     let result = audio.transcribe_samples(&samples)?;
    ///     println!("Text: {}", result.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<AsrResult, FluidAudioError> {
        self.bridge
            .transcribe_samples(samples)
            .map_err(FluidAudioError::from)
    }

    /// Check if ASR is initialized and ready
    pub fn is_asr_available(&self) -> bool {
        self.bridge.is_asr_available()
    }

    // ========== Vocabulary Boosting ==========

    /// Boost recognition of `terms` (names, jargon, drug names) in one-shot
    /// transcription: [`transcribe_samples`](Self::transcribe_samples) and
    /// [`transcribe_file`](Self::transcribe_file). Streaming is unaffected.
    ///
    /// The first call downloads and loads Parakeet CTC 110M (~97.5 MB) into
    /// `~/Library/Application Support/FluidAudio/Models/parakeet-ctc-110m-coreml`;
    /// the bridge keeps it loaded. After each transcription a CTC pass over
    /// the same audio scores the terms, and a term replaces the words it was
    /// misheard as only when the audio supports it better than those words.
    /// Terms are matched case-insensitively and written out as given. Terms
    /// carry no weights: FluidAudio sets the boost from the vocabulary size.
    ///
    /// Replaces any earlier vocabulary. Blank terms are ignored, and a list
    /// with no terms clears the vocabulary. Safe to call while other threads
    /// transcribe: a transcription uses the vocabulary set when it started,
    /// and the last `set_vocabulary`/`clear_vocabulary` call wins. Blocks until
    /// the CTC model is loaded.
    pub fn set_vocabulary(&self, terms: &[String]) -> Result<(), FluidAudioError> {
        self.bridge
            .set_vocabulary(terms)
            .map_err(FluidAudioError::from)
    }

    /// Stop boosting vocabulary in one-shot transcription.
    pub fn clear_vocabulary(&self) {
        self.bridge.clear_vocabulary()
    }

    // ========== Streaming ASR Methods ==========

    /// Initialize Streaming ASR for live partial transcripts
    ///
    /// Loads the streaming Parakeet Unified EN 0.6B engine (2.08 s latency). The
    /// first run downloads ~610 MB (the int8 streaming encoder plus its own
    /// decoder, joint, vocabulary and metadata) into
    /// `~/Library/Application Support/FluidAudio/Models/parakeet-unified-preview/parakeet-unified-en-0.6b`,
    /// pinned to the same Hugging Face revision as [`init_asr`](Self::init_asr).
    /// The folder is separate from `init_asr`'s because FluidAudio deletes a
    /// model folder after a non-network load failure: a failed preview load
    /// cannot remove the one-shot model, nor the reverse.
    ///
    /// Blocks until loaded, so it can run on a background thread while other
    /// threads call `init_asr`, transcribe one-shot or poll
    /// [`is_streaming_asr_available`](Self::is_streaming_asr_available); none of
    /// those wait for it. Concurrent `init_streaming_asr` calls run one at a
    /// time, and a call after a successful load returns at once.
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///
    ///     // Initialize streaming ASR
    ///     audio.init_streaming_asr()?;
    ///
    ///     // Use session-based API for real-time streaming
    ///     audio.streaming_asr_start()?;
    ///
    ///     // Feed audio chunks as they become available
    ///     let chunk1: Vec<f32> = vec![0.0; 16000]; // 1 second
    ///     audio.streaming_asr_feed(&chunk1)?;
    ///
    ///     let chunk2: Vec<f32> = vec![0.0; 16000]; // another second
    ///     audio.streaming_asr_feed(&chunk2)?;
    ///
    ///     // Get final transcription
    ///     let text = audio.streaming_asr_finish()?;
    ///     println!("Transcription: {}", text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn init_streaming_asr(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_streaming_asr()
            .map_err(FluidAudioError::from)
    }

    /// Start a streaming ASR session
    ///
    /// Call this before feeding audio chunks. Use `streaming_asr_feed()` to process
    /// audio chunks, then `streaming_asr_finish()` to get the final result. Any
    /// previous session's audio and text are discarded; the model stays loaded.
    pub fn streaming_asr_start(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .streaming_asr_start()
            .map_err(FluidAudioError::from)
    }

    /// Feed audio samples to the streaming ASR session
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    ///
    /// Call this multiple times to process audio in chunks. The transcription engine
    /// will process the audio incrementally.
    ///
    /// # Returns
    /// * `Some(String)` — the session's transcript so far, suitable for a live
    ///   preview; returned only when it differs from the previous `Some`
    /// * `None` — nothing new has been decoded since then
    ///
    /// The engine decodes once the first 2.08 s are buffered and then every
    /// 1.04 s, so most calls return `None`. Use
    /// [`streaming_asr_finish`](Self::streaming_asr_finish) for the final transcript.
    pub fn streaming_asr_feed(&self, samples: &[f32]) -> Result<Option<String>, FluidAudioError> {
        self.bridge
            .streaming_asr_feed(samples)
            .map_err(FluidAudioError::from)
    }

    /// Finish the streaming ASR session and get the transcription result
    ///
    /// # Returns
    /// * `String` containing the complete transcribed text
    ///
    /// This flushes the buffered audio and returns the full transcription, then
    /// clears the session. Call `streaming_asr_start()` again to start a new one.
    pub fn streaming_asr_finish(&self) -> Result<String, FluidAudioError> {
        self.bridge
            .streaming_asr_finish()
            .map_err(FluidAudioError::from)
    }

    /// Transcribe an audio file through the streaming engine
    ///
    /// This is a convenience method that runs the whole file as one streaming
    /// session. It shares the engine with the session API, so do not call it
    /// while a session is open.
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata; `confidence`
    ///   is not computed on this path and is always 0.0
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_streaming_asr()?;
    ///
    ///     let result = audio.transcribe_file_streaming("long_audio.wav")?;
    ///     println!("Text: {}", result.text);
    ///     println!("RTFx: {:.2}x", result.rtfx);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn transcribe_file_streaming<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<AsrResult, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .transcribe_file_streaming(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Check if streaming ASR is initialized and ready
    ///
    /// Safe to poll from any thread, including while
    /// [`init_streaming_asr`](Self::init_streaming_asr) is still loading.
    pub fn is_streaming_asr_available(&self) -> bool {
        self.bridge.is_streaming_asr_available()
    }

    // ========== Nemotron Streaming ASR Methods ==========

    /// Initialize the Nemotron Speech Streaming 0.6B pipeline
    ///
    /// Models are downloaded from Hugging Face on first use and cached under
    /// Application Support. Nemotron has no separate one-shot model — the
    /// streaming pipeline serves both the live and batch paths.
    pub fn init_nemotron_streaming(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_nemotron_streaming()
            .map_err(FluidAudioError::from)
    }

    /// Feed audio samples to the Nemotron streaming session
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    ///
    /// # Returns
    /// * `String` containing the text decoded from this chunk (empty when the
    ///   chunk produced no new tokens)
    pub fn nemotron_streaming_process(&self, samples: &[f32]) -> Result<String, FluidAudioError> {
        self.bridge
            .nemotron_streaming_process(samples)
            .map_err(FluidAudioError::from)
    }

    /// Read the accumulated hypothesis without advancing the session
    ///
    /// Intended for live preview between [`nemotron_streaming_process`](Self::nemotron_streaming_process)
    /// calls. Returns an empty string before anything has been decoded.
    pub fn nemotron_streaming_get_partial(&self) -> Result<String, FluidAudioError> {
        self.bridge
            .nemotron_streaming_get_partial()
            .map_err(FluidAudioError::from)
    }

    /// Flush buffered audio and return the final transcript
    ///
    /// Call [`nemotron_streaming_reset`](Self::nemotron_streaming_reset) afterwards
    /// before starting another session.
    pub fn nemotron_streaming_finish(&self) -> Result<String, FluidAudioError> {
        self.bridge
            .nemotron_streaming_finish()
            .map_err(FluidAudioError::from)
    }

    /// Clear encoder/decoder cache state for a new session
    ///
    /// The models stay loaded. A session left un-reset carries its state into
    /// the next one, so call this between recordings.
    pub fn nemotron_streaming_reset(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .nemotron_streaming_reset()
            .map_err(FluidAudioError::from)
    }

    /// Check if Nemotron streaming ASR is initialized and ready
    pub fn is_nemotron_streaming_available(&self) -> bool {
        self.bridge.is_nemotron_streaming_available()
    }

    // ========== VAD Methods ==========

    /// Initialize the VAD (Voice Activity Detection) engine
    ///
    /// # Arguments
    /// * `threshold` - Detection threshold (0.0-1.0, default 0.85)
    pub fn init_vad(&self, threshold: f32) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_vad(threshold)
            .map_err(FluidAudioError::from)
    }

    /// Check if VAD is initialized and ready
    pub fn is_vad_available(&self) -> bool {
        self.bridge.is_vad_available()
    }

    /// Run VAD over an audio file.
    ///
    /// The audio is automatically resampled to 16 kHz mono Float32 and processed
    /// in 4096-sample (256 ms) chunks. One [`VadFrame`] is returned per chunk.
    ///
    /// # Arguments
    /// * `path` - Path to an audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_vad(0.5)?;
    ///     let frames = audio.vad_process_file("speech.wav")?;
    ///     let voiced = frames.iter().filter(|f| f.is_voice_active).count();
    ///     println!("{} / {} chunks classified as voice", voiced, frames.len());
    ///     Ok(())
    /// }
    /// ```
    pub fn vad_process_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Vec<VadFrame>, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .vad_process_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Run VAD over raw 16 kHz mono Float32 samples.
    ///
    /// One [`VadFrame`] is returned for every 4096-sample (256 ms) chunk; the
    /// trailing partial chunk (if any) is padded internally.
    pub fn vad_process_samples(&self, samples: &[f32]) -> Result<Vec<VadFrame>, FluidAudioError> {
        self.bridge
            .vad_process_samples(samples)
            .map_err(FluidAudioError::from)
    }

    // ========== Diarization Methods ==========

    /// Initialize the speaker diarization engine
    ///
    /// This downloads and loads the diarization models. First run may take
    /// some time as models are compiled for the Neural Engine.
    ///
    /// # Arguments
    /// * `threshold` - Clustering threshold (0.0-1.0, default 0.6). Lower values
    ///   produce more speakers, higher values merge speakers more aggressively.
    pub fn init_diarization(&self, threshold: f64) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_diarization(threshold)
            .map_err(FluidAudioError::from)
    }

    /// Diarize an audio file using a pre-staged Sortformer model (`.mlpackage`
    /// path), with **no** network download. The caller is responsible for
    /// provisioning and verifying the model. Config is `.balancedV2` (matches
    /// the `SortformerNvidiaLow_v2.mlpackage`). Unlike [`Self::diarize_file`],
    /// this never touches HuggingFace and needs no prior `init_diarization`.
    ///
    /// The compiled model is cached in a writable per-user directory (never next
    /// to `model_path`, so a read-only / air-gapped model location works) and the
    /// loaded handle is retained in-memory, so repeated calls in the same process
    /// reuse it with no reload.
    ///
    /// # Arguments
    /// * `audio` - Path to the audio file (WAV, M4A, MP3, etc.)
    /// * `model_path` - Path to the Sortformer `.mlpackage`
    pub fn diarize_file_with_models<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        audio: P,
        model_path: Q,
    ) -> Result<Vec<DiarizationSegment>, FluidAudioError> {
        let audio_str = audio.as_ref().to_string_lossy();
        if !audio.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(audio_str.to_string()));
        }
        let model_str = model_path.as_ref().to_string_lossy();
        if !model_path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(model_str.to_string()));
        }
        self.bridge
            .diarize_file_with_models(&audio_str, &model_str)
            .map_err(FluidAudioError::from)
    }

    /// Pre-compile the Sortformer `.mlpackage` and warm it, so the first real
    /// diarization is fast. The compiled `.mlmodelc` is written to a writable
    /// per-user cache directory (keyed by a fingerprint of the model — its path,
    /// total size, and newest mtime — never written next
    /// to `model_path`, so a read-only / air-gapped model location works), paying
    /// the one-time ~100s ANE compile up front (e.g. at install time). The loaded
    /// model is also retained in-memory, so within the **same process** subsequent
    /// [`Self::diarize_file_with_models`] calls reuse it with no reload. Across
    /// **separate processes** only the on-disk compiled cache carries over, so the
    /// first diarize in a new process still pays the ~4s warm `MLModel` load (vs
    /// ~100s cold). No audio is processed.
    ///
    /// # Arguments
    /// * `model_path` - Path to the Sortformer `.mlpackage`
    pub fn compile_diarization_model<Q: AsRef<Path>>(
        &self,
        model_path: Q,
    ) -> Result<(), FluidAudioError> {
        let model_str = model_path.as_ref().to_string_lossy();
        if !model_path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(model_str.to_string()));
        }
        self.bridge
            .compile_diarization_model(&model_str)
            .map_err(FluidAudioError::from)
    }

    /// Diarize an audio file to identify speaker segments
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `Vec<DiarizationSegment>` containing speaker-labeled time segments
    pub fn diarize_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Vec<DiarizationSegment>, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .diarize_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Check if diarization is initialized and ready
    pub fn is_diarization_available(&self) -> bool {
        self.bridge.is_diarization_available()
    }

    // ========== TTS (Kokoro) Methods ==========

    /// Initialize the Kokoro TTS engine with a default voice and language.
    ///
    /// `lang` selects the KokoroAne variant in the Swift bridge (`zh` → Mandarin,
    /// everything else → English). Downloads the variant's model on first run
    /// (FluidAudio-managed cache).
    pub fn init_kokoro(&self, default_voice: &str, lang: &str) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_kokoro(default_voice, lang)
            .map_err(FluidAudioError::from)
    }

    /// Synthesize text via Kokoro TTS in the language the engine was initialized
    /// with (`init_kokoro`'s `lang`): English by default, or Mandarin when
    /// `lang` was `zh` (the `.mandarin` KokoroAne variant).
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `voice` - Kokoro voice id (e.g. `am_michael`)
    /// * `speed` - Speech rate (0.5-2.0; clamped by FluidAudio)
    ///
    /// # Returns
    /// * `Vec<u8>` - a complete WAV (24 kHz mono, 16-bit PCM)
    pub fn synthesize_kokoro(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
    ) -> Result<Vec<u8>, FluidAudioError> {
        self.bridge
            .kokoro_synthesize(text, voice, speed)
            .map_err(FluidAudioError::from)
    }

    /// Check if Kokoro TTS is initialized and ready.
    pub fn is_kokoro_available(&self) -> bool {
        self.bridge.is_kokoro_available()
    }

    // ========== System Info ==========

    /// Get system information
    pub fn system_info(&self) -> SystemInfo {
        self.bridge.system_info()
    }

    /// Check if running on Apple Silicon
    pub fn is_apple_silicon(&self) -> bool {
        self.bridge.is_apple_silicon()
    }

    /// Check if running on an Intel Mac (x86_64).
    ///
    /// Local Apple-Silicon-only models (Parakeet, CoreML TTS) will fail
    /// to load on Intel Macs. Apps can use this to gate UI selection of those
    /// models and steer Intel users to cloud transcription instead.
    pub fn is_intel_mac(&self) -> bool {
        self.bridge.is_intel_mac()
    }

    // ========== ITN (Inverse Text Normalization) ==========

    /// Normalize a short spoken-form expression to written form.
    ///
    /// Examples:
    /// - `"two hundred thirty two"` → `"232"`
    /// - `"five dollars and fifty cents"` → `"$5.50"`
    /// - `"period"` → `"."`
    ///
    /// Intended for short fragments. For full sentences with mixed punctuation
    /// commands and ordinary prose, prefer [`itn_normalize_sentence`](Self::itn_normalize_sentence).
    pub fn itn_normalize(&self, text: &str) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize(text)
            .map_err(FluidAudioError::from)
    }

    /// Sentence-mode normalization with sliding-window span matching.
    ///
    /// Uses Apple's NaturalLanguage framework to disambiguate words like
    /// `"period"` between punctuation commands and ordinary nouns/verbs.
    pub fn itn_normalize_sentence(&self, text: &str) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize_sentence(text)
            .map_err(FluidAudioError::from)
    }

    /// Sentence-mode normalization with a caller-controlled maximum span size
    /// (in tokens). Larger spans catch longer multi-word numbers/dates at the
    /// cost of more work per sentence.
    pub fn itn_normalize_sentence_max_span(
        &self,
        text: &str,
        max_span_tokens: u32,
    ) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize_sentence_max_span(text, max_span_tokens)
            .map_err(FluidAudioError::from)
    }

    /// Whether the NeMo ITN engine (text-processing-rs) is loaded.
    ///
    /// The engine has its own Rust runtime, so it ships as a separate
    /// `libfluidaudio_nemo_itn.dylib`, which the build places in
    /// `target/<profile>/`. It is loaded on first use from
    /// `$FLUIDAUDIO_NEMO_ITN_DYLIB` if set, else from the executable's
    /// `../Frameworks/` (app bundles), its own directory, or its parent.
    /// When it is not found this returns `false` and every ITN call returns
    /// its input unchanged.
    pub fn itn_is_native_available(&self) -> bool {
        self.bridge.itn_is_native_available()
    }

    // ========== Cleanup ==========

    /// Release all resources
    pub fn cleanup(&self) {
        self.bridge.cleanup()
    }
}

impl Drop for FluidAudio {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_instance() {
        // Note: This test will fail until Swift bridge is properly linked
        // For now, just test the types exist
        let _ = FluidAudioError::NotInitialized("test".to_string());
    }
}
