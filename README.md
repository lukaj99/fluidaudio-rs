# fluidaudio-rs

Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) - a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.

## Features

- **ASR (Automatic Speech Recognition)**
  - **Parakeet Unified EN 0.6B** - NVIDIA's English speech-to-text model
    ([`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b),
    FastConformer encoder with an RNN-T decoder; punctuation and capitalisation)
    - One-shot transcription of files and raw samples
    - Streaming sessions with live partial transcripts (2.08 s latency)
- **VAD (Voice Activity Detection)** - Detect speech segments in audio
- **Speaker Diarization** - Identify and label different speakers in audio

## Requirements

- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3) recommended
- Rust 1.70+
- Swift 6.2+ toolchain (Xcode 26+); the package manifest uses tools version 6.2

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fluidaudio-rs = "0.1"
```

## Usage

### Speech-to-Text (ASR)

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;

    // Check system info
    let info = audio.system_info();
    println!("Running on: {} ({})", info.chip_name, info.platform);
    println!("Apple Silicon: {}", audio.is_apple_silicon());

    // Initialize ASR (downloads models on first run)
    audio.init_asr()?;

    // Transcribe an audio file
    let result = audio.transcribe_file("audio.wav")?;
    println!("Text: {}", result.text);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Processing speed: {:.1}x realtime", result.rtfx);

    Ok(())
}
```

#### Real-Time Audio (Samples)

For real-time audio applications, you can transcribe raw audio samples directly without file I/O:

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;
    audio.init_asr()?;

    // Audio samples from microphone or streaming source
    // (16kHz mono, normalized to -1.0 to 1.0)
    let samples: Vec<f32> = capture_audio_from_mic();

    // Transcribe samples directly
    let result = audio.transcribe_samples(&samples)?;
    println!("Text: {}", result.text);

    Ok(())
}
```

This is ideal for meeting transcription apps, voice assistants, and other real-time scenarios where writing to temporary files adds unnecessary overhead.

### Streaming ASR (Live Partials)

Streaming ASR runs the streaming export of Parakeet Unified EN 0.6B: each step
re-encodes 5.6 s of left context, a 1.04 s chunk and 1.04 s of look-ahead. A
step runs once 2.08 s of audio is buffered and then every 1.04 s, so a word
reaches the partial transcript about 2.08 s after it is spoken.
`init_streaming_asr` downloads the streaming encoder (~592 MB) on first use;
the rest of the model is shared with `init_asr`.

#### Session-Based API

For real-time streaming with live partials:

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;
    audio.init_streaming_asr()?;

    // Start streaming session (clears any previous one; the model stays loaded)
    audio.streaming_asr_start()?;

    // Feed audio chunks as they arrive (16kHz mono)
    loop {
        let samples: Vec<f32> = capture_audio_chunk(); // From mic, network, etc.
        if let Some(partial) = audio.streaming_asr_feed(&samples)? {
            println!("Partial: {}", partial); // only when the text changed
        }

        if done {
            break;
        }
    }

    // Get final transcription
    let text = audio.streaming_asr_finish()?;
    println!("Transcription: {}", text);

    Ok(())
}
```

`init_streaming_asr` blocks until the model is loaded, so it can run on a
background thread while other threads transcribe one-shot or poll
`is_streaming_asr_available()`.

#### Simple File Wrapper

Runs a whole file through the streaming engine as one session (no confidence
score on this path):

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;
    audio.init_streaming_asr()?;

    let result = audio.transcribe_file_streaming("meeting.wav")?;
    println!("Text: {}", result.text);
    println!("Speed: {:.1}x realtime", result.rtfx);

    Ok(())
}
```

**When to use Streaming ASR:**
- Live captions or dictation previews while audio is still arriving

**When to use Regular ASR:**
- Complete utterances, recordings and files: the full-attention offline encoder
  sees each 15 s window at once and is the more accurate export in FluidAudio's
  LibriSpeech test-clean runs

### Voice Activity Detection (VAD)

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;

    // Initialize VAD with threshold (0.0-1.0)
    audio.init_vad(0.85)?;

    println!("VAD available: {}", audio.is_vad_available());

    Ok(())
}
```

### Speaker Diarization

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;

    // Initialize diarization with clustering threshold (0.0-1.0)
    // Lower = more speakers, higher = fewer speakers
    audio.init_diarization(0.6)?;

    // Diarize an audio file
    let segments = audio.diarize_file("meeting.wav")?;
    for seg in &segments {
        println!(
            "[{:.2}s - {:.2}s] {}",
            seg.start_time, seg.end_time, seg.speaker_id
        );
    }

    Ok(())
}
```

## Model Loading

Models download from Hugging Face on first use into
`~/Library/Application Support/FluidAudio/Models/`. Parakeet Unified EN 0.6B
comes from [`FluidInference/parakeet-unified-en-0.6b-coreml`](https://huggingface.co/FluidInference/parakeet-unified-en-0.6b-coreml),
pinned to commit `d32e972dd4315f1dc3f6be28fb2aab0ab3e80358` so new uploads
cannot change what an existing build loads, into the `parakeet-unified-en-0.6b`
folder:

| Call | Downloads |
|------|-----------|
| `init_asr` | int8 offline encoder, decoder, joint, `vocab.json`, `metadata.json` (~614 MB) |
| `init_streaming_asr` | int8 streaming encoder (~592 MB), plus the shared decoder, joint and vocab files if `init_asr` has not fetched them |

On an M5 Pro (macOS 27.2) the first load after the download compiles the
models for the Neural Engine: about 8.5 s for `init_asr` and 6.5 s for
`init_streaming_asr`. Later loads by the same executable take about 0.1 s.
The compiled cache is tied to the executable: a new or relocated binary
compiled again in our measurements even with the model files cached.

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | Full support |
| macOS (Intel) | Limited (no ASR) |
| iOS | Full support |
| Linux/Windows | Not supported |

## How it Works

This crate uses a C FFI bridge to communicate between Rust and Swift:

1. The Swift layer (`FluidAudioBridge`) wraps the FluidAudio library
2. C-compatible functions are exported using `@_cdecl`
3. Rust calls these functions through `extern "C"` declarations
4. The build.rs script compiles the Swift package and links it

## License

This crate is MIT. [FluidAudio](https://github.com/FluidInference/FluidAudio) is
Apache-2.0. Model weights carry their own licences: Parakeet Unified EN 0.6B is
released by NVIDIA under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
which requires anyone who redistributes the model to include a copy of the
Agreement and a Notice ("Licensed by NVIDIA Corporation under the NVIDIA Open
Model License").
