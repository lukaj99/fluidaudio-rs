//! Example: Streaming ASR (live partial transcripts)
//!
//! This example demonstrates two ways to use Streaming ASR:
//! 1. Session-based API for real-time streaming (start/feed/finish): the file
//!    is fed in 100 ms chunks, as a microphone would deliver it, and every new
//!    partial transcript is printed as it arrives
//! 2. Simple file wrapper that runs a whole file as one streaming session
//!
//! The session API needs a 16kHz mono 16-bit PCM WAV file.
//!
//! Usage: cargo run --example streaming_asr -- path/to/audio.wav [--session]

use fluidaudio_rs::FluidAudio;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let audio_path = args.get(1).ok_or("Usage: streaming_asr <audio_file> [--session]")?;
    let use_session_api = args.get(2).map(|s| s == "--session").unwrap_or(false);

    println!("FluidAudio Streaming ASR Example");
    println!("=================================\n");

    // Create FluidAudio instance
    let audio = FluidAudio::new()?;

    // Print system info
    let info = audio.system_info();
    println!("System: {} ({})", info.chip_name, info.platform);
    println!("Memory: {:.1} GB", info.memory_gb);
    println!("Apple Silicon: {}\n", audio.is_apple_silicon());

    // Initialize Streaming ASR
    println!("Initializing Streaming ASR (downloads the streaming encoder on first run)...");
    let start = Instant::now();
    audio.init_streaming_asr()?;
    println!(
        "Streaming ASR initialized in {:.2}s\n",
        start.elapsed().as_secs_f64()
    );

    if use_session_api {
        println!("Using session-based API (start/feed/finish)");
        println!("===========================================\n");
        session_based_transcription(&audio, audio_path)?;
    } else {
        println!("Using simple file wrapper API");
        println!("==============================\n");
        simple_file_transcription(&audio, audio_path)?;
    }

    Ok(())
}

/// Demonstrate the session-based API for real-time streaming
fn session_based_transcription(audio: &FluidAudio, audio_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let samples = read_wav_samples(audio_path)?;
    println!(
        "Feeding {:.2}s of audio from {} in 100 ms chunks\n",
        samples.len() as f64 / 16_000.0,
        audio_path
    );

    let start = Instant::now();
    audio.streaming_asr_start()?;

    // A feed returns Some only when the transcript changed, so most return None.
    let mut fed = 0;
    for chunk in samples.chunks(1_600) {
        fed += chunk.len();
        if let Some(partial) = audio.streaming_asr_feed(chunk)? {
            println!(
                "[{:5.2}s audio | {:6.3}s wall] {}",
                fed as f64 / 16_000.0,
                start.elapsed().as_secs_f64(),
                partial
            );
        }
    }

    // Finish session and get result
    let text = audio.streaming_asr_finish()?;

    println!("\n--- Session-Based Results ---");
    println!("Text: {}", text);
    println!("Processing time: {:.3}s", start.elapsed().as_secs_f64());

    Ok(())
}

/// Demonstrate the simple file wrapper (easier to use)
fn simple_file_transcription(audio: &FluidAudio, audio_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Transcribing: {}", audio_path);

    let result = audio.transcribe_file_streaming(audio_path)?;

    println!("\n--- File Wrapper Results ---");
    println!("Text: {}", result.text);
    println!("Duration: {:.2}s", result.duration);
    println!("Processing time: {:.2}s", result.processing_time);
    println!("Speed: {:.1}x realtime", result.rtfx);

    Ok(())
}

/// 16-bit PCM samples from a 16kHz mono WAV, scaled to [-1, 1]. Walks the RIFF
/// chunks to `data`, since many WAV files carry `LIST` or padding chunks first.
fn read_wav_samples(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut is_16k_mono_pcm16 = false;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = bytes
            .get(pos + 8..pos + 8 + size)
            .ok_or("truncated WAV chunk")?;
        match &bytes[pos..pos + 4] {
            b"fmt " if body.len() >= 16 => {
                let format_tag = u16::from_le_bytes([body[0], body[1]]);
                let channels = u16::from_le_bytes([body[2], body[3]]);
                let sample_rate = u32::from_le_bytes(body[4..8].try_into()?);
                let bits = u16::from_le_bytes([body[14], body[15]]);
                is_16k_mono_pcm16 =
                    format_tag == 1 && channels == 1 && sample_rate == 16_000 && bits == 16;
            }
            b"data" if is_16k_mono_pcm16 => {
                return Ok(body
                    .chunks_exact(2)
                    .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
                    .collect());
            }
            b"data" => return Err("the session API needs a 16kHz mono 16-bit PCM WAV".into()),
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }
    Err("no data chunk in WAV file".into())
}
