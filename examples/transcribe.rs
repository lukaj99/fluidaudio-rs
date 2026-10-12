//! Example: Transcribe an audio file
//!
//! Usage: cargo run --example transcribe -- path/to/audio.wav

use fluidaudio_rs::FluidAudio;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get audio file path from command line
    let args: Vec<String> = env::args().collect();
    let audio_path = args.get(1).ok_or("Usage: transcribe <audio_file>")?;

    println!("FluidAudio Transcription Example");
    println!("================================\n");

    // Create FluidAudio instance
    let audio = FluidAudio::new()?;

    // Print system info
    let info = audio.system_info();
    println!("System: {} ({})", info.chip_name, info.platform);
    println!("Memory: {:.1} GB", info.memory_gb);
    println!("Apple Silicon: {}\n", audio.is_apple_silicon());

    // Initialize ASR
    println!("Initializing ASR (this may take a moment on first run)...");
    let start = Instant::now();
    audio.init_asr()?;
    println!("ASR initialized in {:.2}s\n", start.elapsed().as_secs_f64());

    // Transcribe
    println!("Transcribing: {}", audio_path);
    let result = audio.transcribe_file(audio_path)?;

    println!("\n--- Results ---");
    println!("Text: {}", result.text);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Duration: {:.2}s", result.duration);
    println!("Processing time: {:.2}s", result.processing_time);
    println!("Speed: {:.1}x realtime", result.rtfx);

    Ok(())
}
