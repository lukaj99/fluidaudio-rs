import Foundation
import AVFoundation
import CoreML
import CryptoKit
import FluidAudio
import NemoITNLoader
import Darwin

// MARK: - Bridge Class

/// Internal bridge class that wraps FluidAudio
/// Internal diarization segment used within the bridge.
struct BridgeDiarizationSegment {
    var speakerId: String
    var startTime: Float
    var endTime: Float
    var qualityScore: Float
}

class FluidAudioBridgeInternal {
    private var asrManager: UnifiedAsrManager?
    // Recorded at load: the actor's own `encoderPrecision` needs an await.
    private var asrPrecision: UnifiedEncoderPrecision?
    private var vadManager: VadManager?
    private var diarizerManager: OfflineDiarizerManager?
    private var nemotronStreamingManager: StreamingNemotronAsrManager?
    private var kokoroManager: KokoroAneManager?
    // Model-path diarization: retain the loaded MLModel (the ~4s-to-load part) keyed by
    // model path so repeated in-process diarize calls skip the reload. Guarded by a lock
    // — the Rust bridge is Send+Sync and may be entered concurrently from multiple threads.
    private let sortformerCacheLock = NSLock()
    private var sortformerModelCache: [String: MLModel] = [:]
    // Streaming ASR can still be loading on one thread while another polls
    // `isStreamingAsrAvailable()`, so the manager is published under a lock, and
    // only once it has loaded. The last partial handed out shares the lock.
    private let streamingAsrLock = NSLock()
    private var streamingAsrManager: StreamingUnifiedAsrManager?
    private var streamingAsrPrecision: UnifiedEncoderPrecision?
    private var lastStreamingPartial = ""
    // The one-shot and preview engines download into separate folders (see
    // `previewModelsDirectory`). Loads into the same folder, from any bridge in
    // the process, would write the same `.partial` files, so each folder has
    // its own process-wide lock: one-shot loads queue behind one-shot loads,
    // preview loads behind preview loads, and neither waits for the other.
    private static let oneShotLoadLock = NSLock()
    private static let previewLoadLock = NSLock()

    // Vocabulary boosting for one-shot transcription. `vocabularyLock` guards
    // only the published session and its generation, so a transcription never
    // waits for a CTC model load; `vocabularyLoadLock` serializes the loads.
    // Each set/clear bumps the generation, and a load publishes its session
    // only if no later set/clear came in meanwhile: the last call wins.
    private let vocabularyLock = NSLock()
    private var vocabularyBoosting: VocabularyBoostingSession?
    private var vocabularyGeneration = 0
    private let vocabularyLoadLock = NSLock()
    private var ctcModels: CtcModels?

    /// Base directory for the live-preview (streaming) engine:
    /// `~/Library/Application Support/FluidAudio/Models/parakeet-unified-preview`,
    /// so its files land in `parakeet-unified-preview/parakeet-unified-en-0.6b`
    /// (~610 MB: the int8 streaming encoder plus its own copies of the decoder,
    /// joint, vocab and metadata).
    ///
    /// It is kept apart from the one-shot folder because FluidAudio's
    /// `ModelHub.loadWithRecovery` deletes the whole repo folder after a load
    /// failure that is not a network error or a cancellation. In a shared
    /// folder a failed preview load would delete the one-shot encoder, and a
    /// failed one-shot load the streaming encoder.
    private static let previewModelsDirectory = MLModelConfigurationUtils
        .defaultModelsDirectory()
        .appendingPathComponent("parakeet-unified-preview", isDirectory: true)

    /// Base directories for the fp16 encoder sets the two ASR paths fall back
    /// to when Core ML cannot load their int8 encoder, so the files land in
    /// `parakeet-unified-fp16/parakeet-unified-en-0.6b` (one-shot) and
    /// `parakeet-unified-preview-fp16/parakeet-unified-en-0.6b` (preview).
    ///
    /// FluidAudio's default would put each fp16 encoder into its path's int8
    /// folder. Separate folders keep `loadWithRecovery`'s purge contained: a
    /// failed fp16 load cannot delete an int8 set, and the int8 attempt on
    /// the next launch cannot delete the fp16 set it will fall back to.
    private static let oneShotFp16ModelsDirectory = MLModelConfigurationUtils
        .defaultModelsDirectory()
        .appendingPathComponent("parakeet-unified-fp16", isDirectory: true)
    private static let previewFp16ModelsDirectory = MLModelConfigurationUtils
        .defaultModelsDirectory()
        .appendingPathComponent("parakeet-unified-preview-fp16", isDirectory: true)

    /// Hugging Face commit of the Core ML Parakeet Unified weights both ASR
    /// paths download. FluidAudio applies it by repo, whatever the target
    /// folder. Unpinned, FluidAudio follows the repo's `main` branch.
    private static let unifiedWeightsRepo = "FluidInference/parakeet-unified-en-0.6b-coreml"
    private static let unifiedWeightsRevision = "d32e972dd4315f1dc3f6be28fb2aab0ab3e80358"

    /// Hugging Face commit of the Parakeet CTC 110M weights vocabulary
    /// boosting downloads.
    private static let ctcWeightsRepo = "FluidInference/parakeet-ctc-110m-coreml"
    private static let ctcWeightsRevision = "accdafd8cf8a2ff1cabe3c11e54416b405d409aa"

    /// Registers the pins once per process. `revisionOverrides` is process-wide,
    /// so this merges keys instead of replacing the dictionary.
    private static let pinUnifiedWeights: Void = {
        ModelRegistry.revisionOverrides[unifiedWeightsRepo] = unifiedWeightsRevision
        ModelRegistry.revisionOverrides[ctcWeightsRepo] = ctcWeightsRevision
    }()

    init() {
        // Before this bridge can start any download.
        _ = Self.pinUnifiedWeights
    }

    /// Load NVIDIA Parakeet Unified EN 0.6B for one-shot transcription.
    ///
    /// The first call downloads the offline file set (~614 MB: int8 15 s
    /// encoder, decoder, joint, vocab, metadata) into
    /// `Application Support/FluidAudio/Models/parakeet-unified-en-0.6b`.
    /// If Core ML cannot load the int8 encoder, the fp16 set is loaded
    /// instead, from `oneShotFp16ModelsDirectory`.
    func initializeAsr() throws {
        Self.oneShotLoadLock.lock()
        defer { Self.oneShotLoadLock.unlock() }

        let (manager, precision) = try Self.loadPreferringInt8(
            path: "one-shot", int8Directory: nil, fp16Directory: Self.oneShotFp16ModelsDirectory
        ) { precision, directory in
            let manager = UnifiedAsrManager(encoderPrecision: precision)
            try await manager.loadModels(to: directory)
            return manager
        }
        asrManager = manager
        asrPrecision = precision
    }

    /// Load a Parakeet Unified engine with the int8 encoder, or with the fp16
    /// one if Core ML rejects int8.
    ///
    /// The int8 encoder cannot build an execution plan on some chips, even
    /// from an intact download (FluidAudio issue #828). `load` runs through
    /// FluidAudio's `loadWithRecovery`, which already purges and re-downloads
    /// the folder once after a load failure, so a Core ML error that still
    /// comes out of it is not a damaged download; one fp16 attempt follows.
    /// Network errors and cancellations are rethrown: fp16 would not fare
    /// better.
    private static func loadPreferringInt8<Manager>(
        path: String,
        int8Directory: URL?,
        fp16Directory: URL,
        load: @escaping (UnifiedEncoderPrecision, URL?) async throws -> Manager
    ) throws -> (Manager, UnifiedEncoderPrecision) {
        do {
            return (try blocking { try await load(.int8, int8Directory) }, .int8)
        } catch let error where isCoreMLError(error) {
            print(
                "FluidAudioBridge: the \(path) int8 encoder failed to load (\(error)); retrying with the fp16 encoder in \(fp16Directory.path)"
            )
            return (try blocking { try await load(.fp16, fp16Directory) }, .fp16)
        }
    }

    /// Whether `error`, or an error it wraps, comes from Core ML.
    static func isCoreMLError(_ error: Error) -> Bool {
        var next: NSError? = error as NSError
        // Bounded: an underlying-error chain may loop.
        for _ in 0..<8 {
            guard let current = next else { return false }
            if current.domain == MLModelErrorDomain { return true }
            next = current.userInfo[NSUnderlyingErrorKey] as? NSError
        }
        return false
    }

    func asrEncoderPrecision() -> UnifiedEncoderPrecision? {
        return asrManager == nil ? nil : asrPrecision
    }

    func transcribeFile(_ path: String) throws -> (String, Float, Double, Double, Float) {
        guard asrManager != nil else {
            throw BridgeError.notInitialized
        }
        let samples = try AudioConverter().resampleAudioFile(path: path)
        return try transcribeSamples(samples)
    }

    /// Trailing zeros that bring `n` samples to `ceil(n / 1280) * 1280 + 160`,
    /// i.e. a valid mel frame count of 8k + 1 (ALIGN).
    ///
    /// FluidAudio 0.17.7's Unified mel extractor declares `floor(n / 160)`
    /// valid frames (UnifiedMelExtractor.swift:61-68). When that count mod 8
    /// is 0 or 7, short clips decode to "" (every jfk prefix up to 1.04 s did).
    /// Padding to 8k + 1 frames matches the framing FluidAudio's v2 path uses.
    /// On real Speech Commands audio through VoiceFlow's app path (A/B G4c,
    /// APP condition) it cut empty transcripts from 19.5% to 0.35% and raised
    /// exact-match accuracy from 77.2% to 96.7%.
    static func alignPadding(sampleCount n: Int) -> Int {
        let target = (n + 1279) / 1280 * 1280 + 160
        return target - n
    }

    func transcribeSamples(_ samples: [Float]) throws -> (String, Float, Double, Double, Float) {
        // Nothing carries over between calls: the manager decodes every 15 s
        // window from a fresh RNNT decoder state.
        guard let manager = asrManager else {
            throw BridgeError.notInitialized
        }
        let padded = samples + [Float](repeating: 0, count: Self.alignPadding(sampleCount: samples.count))
        let boosting = currentVocabularyBoosting()

        let semaphore = DispatchSemaphore(value: 0)
        var result: UnifiedAsrManager.TranscriptionWithTimings?
        var transcribeError: Error?
        var processingTime: Double = 0

        Task {
            let start = DispatchTime.now()
            do {
                let decoded = try await manager.transcribeWithTimings(padded)
                result = decoded
                // The rescoring UnifiedAsrManager applies when boosting is
                // configured on it, over the same padded audio and timings.
                if let boosting,
                    let rescored = await boosting.rescore(
                        text: decoded.text, tokenTimings: decoded.tokenTimings, audioSamples: padded)
                {
                    result = .init(text: rescored.text, tokenTimings: decoded.tokenTimings)
                }
            } catch {
                transcribeError = error
            }
            processingTime = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
            semaphore.signal()
        }

        semaphore.wait()

        if let error = transcribeError {
            throw error
        }

        guard let r = result else {
            throw BridgeError.noResult
        }

        let duration = Double(samples.count) / 16000.0
        let rtfx = processingTime > 0 ? Float(duration / processingTime) : 0
        let confidence = Self.utteranceConfidence(text: r.text, tokenTimings: r.tokenTimings)
        return (r.text, confidence, duration, processingTime, rtfx)
    }

    /// Utterance confidence on the scale of FluidAudio's TDT `ASRResult.confidence`
    /// (`AsrManager.calculateConfidence`), which callers gate on: the mean
    /// per-token softmax probability clamped to [0.1, 1.0], 0.1 for an empty
    /// transcript, and 0.5 when text arrives without token scores.
    private static func utteranceConfidence(text: String, tokenTimings: [TokenTiming]) -> Float {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return 0.1
        }
        guard !tokenTimings.isEmpty else {
            return 0.5
        }
        let mean = tokenTimings.map(\.confidence).reduce(0, +) / Float(tokenTimings.count)
        return max(0.1, min(1.0, mean))
    }

    func isAsrAvailable() -> Bool {
        return asrManager != nil
    }

    // MARK: - Vocabulary Boosting

    /// Boost `terms` in one-shot transcription (`transcribeSamples`,
    /// `transcribeFile`); streaming is unaffected.
    ///
    /// The first call downloads and loads Parakeet CTC 110M (~97.5 MB, into
    /// `Application Support/FluidAudio/Models/parakeet-ctc-110m-coreml`),
    /// which this bridge then keeps. Each term is tokenized with the CTC
    /// tokenizer; a term that encodes to nothing is dropped. After every
    /// transcription a CTC pass over the same audio scores the terms, and a
    /// term replaces the words it was misheard as only when its acoustic
    /// score beats theirs. That is `UnifiedAsrManager`'s own boosting, with
    /// its default `VocabularyBoostingSession.itnDefaultConfig` (the #702
    /// spotter-rescue similarity floors that this engine's ITN output needs).
    /// The session lives on the bridge rather than on the manager because
    /// `UnifiedAsrManager` has no way to unset it.
    ///
    /// Blank terms are ignored; no terms at all clears the vocabulary.
    func setVocabulary(_ terms: [String]) throws {
        let words = terms
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let generation = startVocabularyChange()
        guard !words.isEmpty else {
            publishVocabularyBoosting(nil, generation: generation)
            return
        }

        vocabularyLoadLock.lock()
        defer { vocabularyLoadLock.unlock() }
        let models = try ctcModels ?? Self.blocking { try await CtcModels.downloadAndLoad(variant: .ctc110m) }
        ctcModels = models
        let session = try Self.blocking {
            try await VocabularyBoostingSession(
                vocabulary: CustomVocabularyContext(terms: words.map { CustomVocabularyTerm(text: $0) }),
                ctcModels: models,
                config: VocabularyBoostingSession.itnDefaultConfig
            )
        }
        publishVocabularyBoosting(session.vocabulary.terms.isEmpty ? nil : session, generation: generation)
    }

    func clearVocabulary() {
        publishVocabularyBoosting(nil, generation: startVocabularyChange())
    }

    private func startVocabularyChange() -> Int {
        vocabularyLock.lock()
        defer { vocabularyLock.unlock() }
        vocabularyGeneration += 1
        return vocabularyGeneration
    }

    private func publishVocabularyBoosting(_ session: VocabularyBoostingSession?, generation: Int) {
        vocabularyLock.lock()
        defer { vocabularyLock.unlock() }
        if generation == vocabularyGeneration {
            vocabularyBoosting = session
        }
    }

    private func currentVocabularyBoosting() -> VocabularyBoostingSession? {
        vocabularyLock.lock()
        defer { vocabularyLock.unlock() }
        return vocabularyBoosting
    }

    /// Run `body` on the Swift concurrency pool and block the calling (FFI)
    /// thread until it finishes.
    private static func blocking<T>(_ body: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<T, Error>?
        Task {
            do {
                result = .success(try await body())
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try result!.get()
    }

    /// Map a kesha/espeak-style language tag to a KokoroAne variant. FluidAudio
    /// 0.14.8 ships exactly two KokoroAne variants — `.english` and `.mandarin`
    /// — so `zh` selects Mandarin (its own tone-aware G2P) and everything else
    /// (en plus the Latin-script es/fr/it/pt, which synthesize acceptably through
    /// the English G2P) falls back to `.english`.
    private static func kokoroVariant(for lang: String) -> KokoroAneVariant {
        let base = lang.lowercased().split(separator: "-").first.map(String.init) ?? ""
        switch base {
        case "zh": return .mandarin
        default: return .english
        }
    }

    func initializeKokoro(defaultVoice: String, lang: String) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                // `KokoroAneManager` feeds `speed` as a real model input tensor,
                // so `--rate` applies correctly (unlike the prior `voiceSpeed:` path).
                let variant = Self.kokoroVariant(for: lang)
                let manager = KokoroAneManager(variant: variant, defaultVoice: defaultVoice)
                try await manager.initialize(preloadVoices: [defaultVoice])
                self.kokoroManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    /// Synthesize `text` and return a complete WAV byte buffer (24 kHz mono),
    /// exactly what `KokoroAneManager.synthesize` produces. `speed` (1.0 =
    /// normal) is fed to the model as a real input tensor, so it genuinely
    /// applies — unlike the removed `KokoroTtsManager.synthesize(voiceSpeed:)`.
    func synthesizeKokoro(text: String, voice: String, speed: Float) throws -> Data {
        guard let manager = kokoroManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: Data?
        var synthError: Error?

        Task {
            do {
                result = try await manager.synthesize(text: text, voice: voice, speed: speed)
            } catch {
                synthError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = synthError {
            throw error
        }

        guard let data = result else {
            throw BridgeError.noResult
        }

        return data
    }

    func isKokoroAvailable() -> Bool {
        return kokoroManager != nil
    }

    func initializeVad(_ threshold: Float) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let config = VadConfig(defaultThreshold: threshold)
                let manager = try await VadManager(config: config)
                self.vadManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    func isVadAvailable() -> Bool {
        return vadManager != nil
    }

    // MARK: - Diarization

    func initializeDiarization(_ threshold: Double) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                var config = OfflineDiarizerConfig()
                config.clustering.threshold = threshold
                let manager = OfflineDiarizerManager(config: config)
                try await manager.prepareModels()
                self.diarizerManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    func diarizeFile(_ path: String) throws -> [BridgeDiarizationSegment] {
        guard let manager = diarizerManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: DiarizationResult?
        var diarizeError: Error?

        Task {
            do {
                let url = URL(fileURLWithPath: path)
                result = try await manager.process(url)
            } catch {
                diarizeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = diarizeError {
            throw error
        }

        guard let r = result else {
            throw BridgeError.noResult
        }

        return r.segments.map { segment in
            BridgeDiarizationSegment(
                speakerId: segment.speakerId,
                startTime: segment.startTimeSeconds,
                endTime: segment.endTimeSeconds,
                qualityScore: segment.qualityScore
            )
        }
    }

    /// Resolve the pre-staged `.mlpackage` at `modelPath` to a compiled `.mlmodelc` in a
    /// WRITABLE per-user cache — never next to the model, which may live in a read-only /
    /// air-gapped location. `MLModel.compileModel` recompiles to a throwaway temp on every
    /// call; the ~100s cost is the CoreML ANE program compile inside `MLModel(contentsOf:,
    /// .all)`, which Apple caches in `com.apple.e5rt.e5bundlecache` keyed to the compiled
    /// model's path — so a *stable* path makes the 2nd process onward ~4s (measured: cold
    /// 104.7s, warm 3.9s on M3 Pro). The compiled name fingerprints the package (path +
    /// size + mtime), so swapping a different model in at the same path recompiles.
    private static func compiledModelURL(forModelPath modelPath: String) async throws -> URL {
        let cacheDir = try compiledCacheDir()
        let key = cacheKey(forModelPath: modelPath)
        let stableURL = cacheDir.appendingPathComponent(key + ".mlmodelc")
        let fm = FileManager.default

        if !fm.fileExists(atPath: stableURL.path) {
            let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
            defer { try? fm.removeItem(at: compiled) }
            // Publish atomically: stage into a unique sibling of the final path (same
            // volume, so the rename below is atomic), then rename into place — a reader
            // never sees a half-written bundle. copyItem, not moveItem, from `compiled`:
            // it lives in the system temp dir, often a different volume (EXDEV).
            let staging = cacheDir.appendingPathComponent(key + ".\(UUID().uuidString).tmp")
            do {
                try fm.copyItem(at: compiled, to: staging)
                try fm.moveItem(at: staging, to: stableURL)
            } catch {
                try? fm.removeItem(at: staging)
                // Another process may have published it first; tolerate that.
                if !fm.fileExists(atPath: stableURL.path) { throw error }
            }
        }
        return stableURL
    }

    /// Writable cache dir for compiled Sortformer models: Application Support (not
    /// `.cachesDirectory`, which the OS may reclaim — recompiling costs ~100s).
    private static func compiledCacheDir() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent("fluidaudio-rs/SortformerCompiled", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// SHA-256 of `<path>|<recursive size>|<newest mtime>`. Size+mtime invalidate the
    /// compiled artifact if a different model is staged at the same path; `.mlpackage` is a
    /// directory bundle, so the fingerprint walks its contents rather than stat-ing the dir.
    private static func cacheKey(forModelPath modelPath: String) -> String {
        let url = URL(fileURLWithPath: modelPath).standardizedFileURL
        var totalSize: Int64 = 0
        var newestMtime: TimeInterval = 0
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]
        if let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: Array(keys)) {
            for case let file as URL in en {
                guard let v = try? file.resourceValues(forKeys: keys) else { continue }
                totalSize += Int64(v.fileSize ?? 0)
                if let m = v.contentModificationDate?.timeIntervalSince1970, m > newestMtime {
                    newestMtime = m
                }
            }
        }
        let material = "\(url.path)|\(totalSize)|\(Int(newestMtime))"
        let digest = SHA256.hash(data: Data(material.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Return a loaded `MLModel` for `modelPath`, retained on the bridge so repeated
    /// in-process diarizations skip the ~4s `MLModel(contentsOf:)` load. Double-checked
    /// under `sortformerCacheLock`; the lock is never held across the `await` compile/load.
    /// We cache the `MLModel` (thread-safe, expensive to load) and build a fresh
    /// `SortformerModels` per call — `SortformerModels` owns mutable scratch buffers that
    /// must not be shared across concurrent diarizations.
    private func cachedSortformerModel(modelPath: String) async throws -> MLModel {
        // Key by raw path: a hit returns with zero filesystem I/O. The content
        // fingerprint (which requires walking the bundle) is computed only on the miss
        // path, inside `compiledModelURL`, to name the on-disk compiled artifact — that
        // disk cache is where content invalidation matters (across processes/versions).
        sortformerCacheLock.lock()
        if let cached = sortformerModelCache[modelPath] {
            sortformerCacheLock.unlock()
            return cached
        }
        sortformerCacheLock.unlock()

        let url = try await Self.compiledModelURL(forModelPath: modelPath)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let model = try MLModel(contentsOf: url, configuration: cfg)

        sortformerCacheLock.lock()
        defer { sortformerCacheLock.unlock() }
        if let existing = sortformerModelCache[modelPath] { return existing }
        sortformerModelCache[modelPath] = model
        return model
    }

    /// Diarize using a pre-staged Sortformer `.mlpackage` at `modelPath`. The compiled
    /// model is cached in a writable per-user dir and retained in-memory
    /// (`cachedSortformerModel`), so repeated in-process calls skip the reload; it never
    /// downloads from HuggingFace — unlike `diarizeFile`, which uses the auto-downloading
    /// OfflineDiarizerManager. Config is `.balancedV2` to match the shipped
    /// `SortformerNvidiaLow_v2.mlpackage` (fifoLen=188); a mismatched config is a hard
    /// CoreML tensor-shape error at runtime.
    func diarizeFileWithModels(audioPath: String, modelPath: String) throws -> [BridgeDiarizationSegment] {
        let semaphore = DispatchSemaphore(value: 0)
        var timeline: DiarizerTimeline?
        var diarizeError: Error?

        Task {
            do {
                let diarizer = SortformerDiarizer(
                    config: SortformerConfig.balancedV2,
                    timelineConfig: DiarizerTimelineConfig.sortformerDefault
                )
                let model = try await self.cachedSortformerModel(modelPath: modelPath)
                let models = try SortformerModels(config: SortformerConfig.balancedV2, main: model)
                diarizer.initialize(models: models)
                timeline = try diarizer.processComplete(
                    audioFileURL: URL(fileURLWithPath: audioPath),
                    keepingEnrolledSpeakers: nil,
                    finalizeOnCompletion: true,
                    progressCallback: nil
                )
            } catch {
                diarizeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = diarizeError {
            throw error
        }

        guard let tl = timeline else {
            throw BridgeError.noResult
        }

        // DiarizerTimeline.speakers: [Int: DiarizerSpeaker]; flatten finalized
        // segments across speakers and sort by start time.
        let allSegments = tl.speakers.values.flatMap { $0.finalizedSegments }
        return allSegments
            .sorted { $0.startTime < $1.startTime }
            .map { seg in
                BridgeDiarizationSegment(
                    speakerId: String(format: "SPEAKER_%02d", max(0, seg.speakerIndex)),
                    startTime: seg.startTime,
                    endTime: seg.endTime,
                    qualityScore: 1.0
                )
            }
    }

    /// Warm the diarization model: compile the `.mlpackage` into the writable per-user
    /// cache (one-time ~100s ANE compile, populates the e5rt cache) and load it once,
    /// retaining the handle in-memory. Callers warm this at install time so the first real
    /// diarize is fast — ~4s cross-process, or no reload at all in-process. No audio is
    /// processed; the caller's model directory is never written to.
    func compileDiarizationModel(modelPath: String) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var compileError: Error?

        Task {
            do {
                _ = try await self.cachedSortformerModel(modelPath: modelPath)
            } catch {
                compileError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = compileError {
            throw error
        }
    }

    func isDiarizationAvailable() -> Bool {
        return diarizerManager != nil
    }

    // MARK: - Streaming ASR

    /// Load the streaming Parakeet Unified EN 0.6B engine for live preview.
    ///
    /// One `StreamingUnifiedAsrManager` at FluidAudio's default [70, 13, 13]
    /// encoder context (5.6 s left, 1.04 s chunk, 1.04 s right: 2.08 s latency)
    /// serves every session. The first call downloads ~610 MB (the streaming
    /// encoder plus its own decoder, joint, vocab and metadata) into
    /// `previewModelsDirectory`, apart from the one-shot set, so a failed load
    /// here never deletes the one-shot model.
    ///
    /// Blocks until the model is downloaded and loaded. Other threads may poll
    /// `isStreamingAsrAvailable()` meanwhile; it turns true only once the
    /// manager has loaded. One-shot loads and transcriptions do not wait for
    /// it. Calling this again after a successful load is a no-op.
    func initializeStreamingAsr() throws {
        Self.previewLoadLock.lock()
        defer { Self.previewLoadLock.unlock() }
        if isStreamingAsrAvailable() {
            return
        }

        let (manager, precision) = try Self.loadPreferringInt8(
            path: "preview", int8Directory: Self.previewModelsDirectory,
            fp16Directory: Self.previewFp16ModelsDirectory
        ) { precision, directory in
            let manager = StreamingUnifiedAsrManager(encoderPrecision: precision)
            try await manager.loadModels(to: directory)
            return manager
        }

        streamingAsrLock.lock()
        streamingAsrManager = manager
        streamingAsrPrecision = precision
        streamingAsrLock.unlock()
    }

    /// Begin a streaming session on the loaded manager.
    ///
    /// `reset()` clears the previous session's audio, decoder state and
    /// transcript; the model stays loaded.
    func streamingAsrStart() throws {
        guard let manager = loadedStreamingAsrManager() else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var startError: Error?

        Task {
            do {
                try await manager.reset()
            } catch {
                startError = error
            }
            semaphore.signal()
        }

        semaphore.wait()
        clearLastStreamingPartial()

        if let error = startError {
            throw error
        }
    }

    /// Feed a chunk of 16kHz mono audio and return the hypothesis if it changed.
    ///
    /// The engine decodes once the first 2.08 s are buffered and then every
    /// 1.04 s, so most feeds add no text: `nil` means nothing new since the
    /// last returned partial. Callers use this for a live preview only;
    /// `streamingAsrFinish()` returns the session's final transcript.
    func streamingAsrFeed(_ samples: [Float]) throws -> String? {
        guard let manager = loadedStreamingAsrManager() else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var partial = ""
        var feedError: Error?

        Task {
            do {
                if !samples.isEmpty {
                    try await manager.appendAudio(Self.monoBuffer(samples))
                }
                try await manager.processBufferedAudio()
                partial = await manager.getPartialTranscript()
            } catch {
                feedError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = feedError {
            throw error
        }

        streamingAsrLock.lock()
        defer { streamingAsrLock.unlock() }
        guard !partial.isEmpty, partial != lastStreamingPartial else {
            return nil
        }
        lastStreamingPartial = partial
        return partial
    }

    /// Flush the session and return its final transcript. The manager is reset
    /// afterwards, so the next `streamingAsrStart()` begins clean.
    func streamingAsrFinish() throws -> String {
        guard let manager = loadedStreamingAsrManager() else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result = ""
        var finishError: Error?

        Task {
            do {
                result = try await manager.finish()
            } catch {
                finishError = error
            }
            // Reset even after a failed finish. A failing reset is not lost:
            // streamingAsrStart() resets again and reports it.
            try? await manager.reset()
            semaphore.signal()
        }

        semaphore.wait()
        clearLastStreamingPartial()

        if let error = finishError {
            throw error
        }

        return result
    }

    /// Transcribe a whole file as one streaming session. It shares the manager
    /// with the session API, so it must not run while a session is open.
    func transcribeFileStreaming(_ path: String) throws -> (String, Float, Double, Double, Float) {
        guard let manager = loadedStreamingAsrManager() else {
            throw BridgeError.notInitialized
        }
        let samples = try AudioConverter().resampleAudioFile(path: path)

        let semaphore = DispatchSemaphore(value: 0)
        var text = ""
        var transcribeError: Error?
        var processingTime: Double = 0

        Task {
            let start = DispatchTime.now()
            do {
                try await manager.reset()
                if !samples.isEmpty {
                    try await manager.appendAudio(Self.monoBuffer(samples))
                }
                text = try await manager.finish()
            } catch {
                transcribeError = error
            }
            processingTime = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
            try? await manager.reset()
            semaphore.signal()
        }

        semaphore.wait()

        if let error = transcribeError {
            throw error
        }

        let duration = Double(samples.count) / 16000.0
        let rtfx = processingTime > 0 ? Float(duration / processingTime) : 0
        return (text, 0.0, duration, processingTime, rtfx)
    }

    func isStreamingAsrAvailable() -> Bool {
        return loadedStreamingAsrManager() != nil
    }

    func streamingAsrEncoderPrecision() -> UnifiedEncoderPrecision? {
        streamingAsrLock.lock()
        defer { streamingAsrLock.unlock() }
        return streamingAsrManager == nil ? nil : streamingAsrPrecision
    }

    private func loadedStreamingAsrManager() -> StreamingUnifiedAsrManager? {
        streamingAsrLock.lock()
        defer { streamingAsrLock.unlock() }
        return streamingAsrManager
    }

    private func clearLastStreamingPartial() {
        streamingAsrLock.lock()
        lastStreamingPartial = ""
        streamingAsrLock.unlock()
    }

    /// Wrap 16kHz mono samples in the Float32 buffer FluidAudio's streaming API takes.
    private static func monoBuffer(_ samples: [Float]) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        return buffer
    }

    // MARK: - Nemotron Streaming ASR

    /// Load the Nemotron Speech Streaming 0.6B pipeline.
    ///
    /// Models are downloaded from Hugging Face on first use and cached under
    /// Application Support. Unlike Parakeet there is no separate one-shot
    /// model — the streaming pipeline serves both paths.
    func initializeNemotronStreaming() throws {
        // FluidAudio's default tier became 2240 ms in 0.15.1; keep the 1120 ms
        // tier (and its existing `nemotron-streaming/1120ms` cache).
        let chunkSize = NemotronChunkSize.ms1120
        Self.removeStaleNemotronCache(
            at: MLModelConfigurationUtils.defaultModelsDirectory(for: chunkSize.repo))

        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let manager = StreamingNemotronAsrManager(requestedChunkSize: chunkSize)
                try await manager.loadModels()
                self.nemotronStreamingManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    /// Feed a chunk of 16kHz mono audio, returning the text decoded from it.
    /// Delete a Nemotron chunk folder (e.g. `Models/nemotron-streaming/1120ms`)
    /// if any compiled model in it lacks `model.mil`, so FluidAudio downloads
    /// the folder fresh.
    ///
    /// Caches downloaded under FluidAudio 0.14.x can hold a
    /// `decoder_joint.mlmodelc` without `model.mil`. FluidAudio 0.17.7's cache
    /// check only looks for `coremldata.bin`, so it loads the bundle, and
    /// Core ML crashes the process (EXC_BAD_ACCESS) instead of throwing, which
    /// means FluidAudio's purge-and-redownload never runs.
    static func removeStaleNemotronCache(at chunkFolder: URL) {
        let fileManager = FileManager.default
        guard let walker = fileManager.enumerator(at: chunkFolder, includingPropertiesForKeys: nil) else {
            return
        }
        var broken: [String] = []
        for case let url as URL in walker where url.pathExtension == "mlmodelc" {
            walker.skipDescendants()
            if !fileManager.fileExists(atPath: url.appendingPathComponent("model.mil").path) {
                broken.append(url.lastPathComponent)
            }
        }
        guard !broken.isEmpty else { return }
        do {
            try fileManager.removeItem(at: chunkFolder)
            print("FluidAudioBridge: removed stale Nemotron cache \(chunkFolder.path): no model.mil in \(broken.joined(separator: ", "))")
        } catch {
            print("FluidAudioBridge: could not remove stale Nemotron cache \(chunkFolder.path): \(error)")
        }
    }

    func nemotronStreamingProcess(_ samples: [Float]) throws -> String {
        guard let manager = nemotronStreamingManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: String?
        var processError: Error?

        Task {
            do {
                let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(samples.count))!
                buffer.frameLength = UInt32(samples.count)

                let channelData = buffer.floatChannelData![0]
                for (i, sample) in samples.enumerated() {
                    channelData[i] = sample
                }

                result = try await manager.process(audioBuffer: buffer)
            } catch {
                processError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = processError {
            throw error
        }

        return result ?? ""
    }

    /// Current accumulated hypothesis, for live preview between chunks.
    func nemotronStreamingGetPartial() throws -> String {
        guard let manager = nemotronStreamingManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: String?

        Task {
            result = await manager.getPartialTranscript()
            semaphore.signal()
        }

        semaphore.wait()

        return result ?? ""
    }

    /// Flush any buffered audio and return the final transcript.
    func nemotronStreamingFinish() throws -> String {
        guard let manager = nemotronStreamingManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: String?
        var finishError: Error?

        Task {
            do {
                result = try await manager.finish()
            } catch {
                finishError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = finishError {
            throw error
        }

        return result ?? ""
    }

    /// Clear encoder/decoder cache state for a new session.
    ///
    /// The manager is reused across recordings and accumulates state, so this
    /// must run between sessions. It keeps the models loaded — unlike
    /// `cleanup()`, which releases them.
    func nemotronStreamingReset() throws {
        guard let manager = nemotronStreamingManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)

        Task {
            await manager.reset()
            semaphore.signal()
        }

        semaphore.wait()
    }

    func isNemotronStreamingAvailable() -> Bool {
        return nemotronStreamingManager != nil
    }

    // MARK: - VAD processing

    /// Returned per-chunk VAD frame data, suitable for flat C arrays.
    struct BridgeVadFrame {
        var probability: Float
        var isVoiceActive: Bool
        var processingTime: Double
    }

    func vadProcessFile(_ path: String) throws -> [BridgeVadFrame] {
        guard let manager = vadManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var frames: [BridgeVadFrame] = []
        var processError: Error?

        Task {
            do {
                let url = URL(fileURLWithPath: path)
                let results = try await manager.process(url)
                frames = results.map {
                    BridgeVadFrame(
                        probability: $0.probability,
                        isVoiceActive: $0.isVoiceActive,
                        processingTime: $0.processingTime
                    )
                }
            } catch {
                processError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = processError {
            throw error
        }

        return frames
    }

    func vadProcessSamples(_ samples: [Float]) throws -> [BridgeVadFrame] {
        guard let manager = vadManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var frames: [BridgeVadFrame] = []
        var processError: Error?

        Task {
            do {
                let results = try await manager.process(samples)
                frames = results.map {
                    BridgeVadFrame(
                        probability: $0.probability,
                        isVoiceActive: $0.isVoiceActive,
                        processingTime: $0.processingTime
                    )
                }
            } catch {
                processError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = processError {
            throw error
        }

        return frames
    }

    // MARK: - ITN (Inverse Text Normalization)

    func itnNormalize(_ input: String) -> String {
        return TextNormalizer.shared.normalize(input)
    }

    func itnNormalizeSentence(_ input: String) -> String {
        return TextNormalizer.shared.normalizeSentence(input)
    }

    func itnNormalizeSentenceMaxSpan(_ input: String, maxSpanTokens: UInt32) -> String {
        return TextNormalizer.shared.normalizeSentence(input, maxSpanTokens: maxSpanTokens)
    }

    /// Whether the NeMo engine dylib loaded. `TextNormalizer.isNativeAvailable`
    /// is a compile-time `true` here, since the trait is on, whether or not
    /// NemoITNLoader found the dylib.
    func itnIsNativeAvailable() -> Bool {
        return fluidaudio_nemo_itn_available() != 0
    }

    func cleanup() {
        clearVocabulary()
        vocabularyLoadLock.lock()
        ctcModels = nil
        vocabularyLoadLock.unlock()
        asrManager = nil
        vadManager = nil
        diarizerManager = nil
        sortformerCacheLock.lock()
        sortformerModelCache.removeAll()
        sortformerCacheLock.unlock()
        streamingAsrLock.lock()
        streamingAsrManager = nil
        lastStreamingPartial = ""
        streamingAsrLock.unlock()
        nemotronStreamingManager = nil
        kokoroManager = nil
    }
}

enum BridgeError: Error {
    case notInitialized
    case noResult
}

// MARK: - C FFI Functions

// Each handle owns its retained bridge. Keep no global reference to it:
// unsynchronized writes from concurrent creates over-release bridges.
@_cdecl("fluidaudio_bridge_create")
public func fluidaudio_bridge_create() -> UnsafeMutableRawPointer? {
    return Unmanaged.passRetained(FluidAudioBridgeInternal()).toOpaque()
}

@_cdecl("fluidaudio_bridge_destroy")
public func fluidaudio_bridge_destroy(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeRetainedValue().cleanup()
}

@_cdecl("fluidaudio_initialize_asr")
public func fluidaudio_initialize_asr(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeAsr()
        return 0
    } catch {
        print("ASR init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_transcribe_file")
public func fluidaudio_transcribe_file(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let (text, confidence, duration, processingTime, rtfx) = try bridge.transcribeFile(pathString)

        // Allocate and copy text
        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        outConfidence?.pointee = confidence
        outDuration?.pointee = duration
        outProcessingTime?.pointee = processingTime
        outRtfx?.pointee = rtfx

        return 0
    } catch {
        print("Transcribe error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_transcribe_samples")
public func fluidaudio_transcribe_samples(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ sampleCount: UInt32,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(sampleCount)))

    do {
        let (text, confidence, duration, processingTime, rtfx) = try bridge.transcribeSamples(samplesArray)

        // Allocate and copy text
        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        outConfidence?.pointee = confidence
        outDuration?.pointee = duration
        outProcessingTime?.pointee = processingTime
        outRtfx?.pointee = rtfx

        return 0
    } catch {
        print("Transcribe samples error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_asr_available")
public func fluidaudio_is_asr_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isAsrAvailable() ? 1 : 0
}

/// 0 when no engine is loaded, 1 for an int8 encoder, 2 for fp16.
private func precisionCode(_ precision: UnifiedEncoderPrecision?) -> Int32 {
    switch precision {
    case nil: return 0
    case .int8: return 1
    case .fp16: return 2
    }
}

/// Encoder precision of the loaded one-shot engine: 0 none, 1 int8, 2 fp16.
@_cdecl("fluidaudio_asr_encoder_precision")
public func fluidaudio_asr_encoder_precision(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return precisionCode(bridge.asrEncoderPrecision())
}

/// Encoder precision of the loaded live-preview engine: 0 none, 1 int8, 2 fp16.
@_cdecl("fluidaudio_streaming_asr_encoder_precision")
public func fluidaudio_streaming_asr_encoder_precision(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return precisionCode(bridge.streamingAsrEncoderPrecision())
}

/// Hand `error`'s description to the caller through `outError` (free it with
/// `fluidaudio_free_string`) and return the failure status, -1.
private func reportFailure(
    _ error: Error, _ outError: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    outError?.pointee = strdup(String(describing: error))
    return -1
}

// MARK: - Vocabulary Boosting FFI

/// Boost `count` NUL-terminated `terms` in one-shot transcription. On failure
/// returns -1 and sets `outError`.
@_cdecl("fluidaudio_set_vocabulary")
public func fluidaudio_set_vocabulary(
    _ ptr: UnsafeMutableRawPointer?,
    _ terms: UnsafePointer<UnsafePointer<CChar>?>?,
    _ count: UInt32,
    _ outError: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let words = (0..<Int(count)).compactMap { i in terms?[i].map { String(cString: $0) } }
    do {
        try bridge.setVocabulary(words)
        return 0
    } catch {
        return reportFailure(error, outError)
    }
}

@_cdecl("fluidaudio_clear_vocabulary")
public func fluidaudio_clear_vocabulary(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue().clearVocabulary()
}

// MARK: - Streaming ASR FFI

@_cdecl("fluidaudio_initialize_streaming_asr")
public func fluidaudio_initialize_streaming_asr(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeStreamingAsr()
        return 0
    } catch {
        print("Streaming ASR init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_streaming_asr_start")
public func fluidaudio_streaming_asr_start(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.streamingAsrStart()
        return 0
    } catch {
        print("Streaming ASR start error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_streaming_asr_feed")
public func fluidaudio_streaming_asr_feed(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: UInt32,
    _ outPartialText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(count)))

    do {
        let partialText = try bridge.streamingAsrFeed(samplesArray)

        if let outPartialText = outPartialText {
            if let text = partialText {
                outPartialText.pointee = strdup(text)
            } else {
                outPartialText.pointee = nil
            }
        }

        return 0
    } catch {
        print("Streaming ASR feed error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_streaming_asr_finish")
public func fluidaudio_streaming_asr_finish(
    _ ptr: UnsafeMutableRawPointer?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    do {
        let text = try bridge.streamingAsrFinish()

        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        return 0
    } catch {
        print("Streaming ASR finish error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_transcribe_file_streaming")
public func fluidaudio_transcribe_file_streaming(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let (text, confidence, duration, processingTime, rtfx) = try bridge.transcribeFileStreaming(pathString)

        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        outConfidence?.pointee = confidence
        outDuration?.pointee = duration
        outProcessingTime?.pointee = processingTime
        outRtfx?.pointee = rtfx

        return 0
    } catch {
        print("Streaming transcribe file error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_streaming_asr_available")
public func fluidaudio_is_streaming_asr_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isStreamingAsrAvailable() ? 1 : 0
}

// MARK: - Nemotron Streaming FFI

@_cdecl("fluidaudio_initialize_nemotron_streaming")
public func fluidaudio_initialize_nemotron_streaming(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeNemotronStreaming()
        return 0
    } catch {
        print("Nemotron streaming init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_nemotron_streaming_process")
public func fluidaudio_nemotron_streaming_process(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: UInt32,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(count)))

    do {
        let text = try bridge.nemotronStreamingProcess(samplesArray)

        if let outText = outText {
            outText.pointee = strdup(text)
        }

        return 0
    } catch {
        print("Nemotron streaming process error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_nemotron_streaming_get_partial")
public func fluidaudio_nemotron_streaming_get_partial(
    _ ptr: UnsafeMutableRawPointer?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    do {
        let text = try bridge.nemotronStreamingGetPartial()

        if let outText = outText {
            outText.pointee = strdup(text)
        }

        return 0
    } catch {
        print("Nemotron streaming get_partial error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_nemotron_streaming_finish")
public func fluidaudio_nemotron_streaming_finish(
    _ ptr: UnsafeMutableRawPointer?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    do {
        let text = try bridge.nemotronStreamingFinish()

        if let outText = outText {
            outText.pointee = strdup(text)
        }

        return 0
    } catch {
        print("Nemotron streaming finish error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_nemotron_streaming_reset")
public func fluidaudio_nemotron_streaming_reset(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.nemotronStreamingReset()
        return 0
    } catch {
        print("Nemotron streaming reset error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_nemotron_streaming_available")
public func fluidaudio_is_nemotron_streaming_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isNemotronStreamingAvailable() ? 1 : 0
}

// MARK: - VAD FFI

@_cdecl("fluidaudio_initialize_vad")
public func fluidaudio_initialize_vad(_ ptr: UnsafeMutableRawPointer?, _ threshold: Float) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeVad(threshold)
        return 0
    } catch {
        print("VAD init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_vad_available")
public func fluidaudio_is_vad_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isVadAvailable() ? 1 : 0
}

// MARK: - Diarization FFI

@_cdecl("fluidaudio_initialize_diarization")
public func fluidaudio_initialize_diarization(_ ptr: UnsafeMutableRawPointer?, _ threshold: Double) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeDiarization(threshold)
        return 0
    } catch {
        print("Diarization init error: \(error)")
        return -1
    }
}

/// Diarize a file. Returns segment count via outCount.
/// Each segment is 4 consecutive values: speakerId (char*), startTime (float), endTime (float), qualityScore (float).
/// The flat arrays outSpeakerIds, outStartTimes, outEndTimes, outQualityScores must be freed by the caller.
@_cdecl("fluidaudio_diarize_file")
public func fluidaudio_diarize_file(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outSpeakerIds: UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?>?,
    _ outStartTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outEndTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outQualityScores: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let segments = try bridge.diarizeFile(pathString)
        emitDiarizationSegments(
            segments,
            outSpeakerIds: outSpeakerIds,
            outStartTimes: outStartTimes,
            outEndTimes: outEndTimes,
            outQualityScores: outQualityScores,
            outCount: outCount
        )
        return 0
    } catch {
        print("Diarize error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_diarization_available")
public func fluidaudio_is_diarization_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isDiarizationAvailable() ? 1 : 0
}

@_cdecl("fluidaudio_free_diarization_result")
public func fluidaudio_free_diarization_result(
    _ speakerIds: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ startTimes: UnsafeMutablePointer<Float>?,
    _ endTimes: UnsafeMutablePointer<Float>?,
    _ qualityScores: UnsafeMutablePointer<Float>?,
    _ count: UInt32
) {
    if let ids = speakerIds {
        for i in 0..<Int(count) {
            free(ids[i])
        }
        ids.deallocate()
    }
    startTimes?.deallocate()
    endTimes?.deallocate()
    qualityScores?.deallocate()
}

// MARK: - System Info FFI

@_cdecl("fluidaudio_get_platform")
public func fluidaudio_get_platform(_ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) {
    #if os(macOS)
    let platform = "macOS"
    #elseif os(iOS)
    let platform = "iOS"
    #else
    let platform = "unknown"
    #endif

    out?.pointee = strdup(platform)
}

@_cdecl("fluidaudio_get_chip_name")
public func fluidaudio_get_chip_name(_ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) {
    var size: size_t = 0
    var chipName = "Unknown"

    if sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 {
        var buffer = [CChar](repeating: 0, count: Int(size))
        if sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 {
            chipName = String(cString: buffer)
        }
    }

    out?.pointee = strdup(chipName)
}

@_cdecl("fluidaudio_get_memory_gb")
public func fluidaudio_get_memory_gb() -> Double {
    return Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
}

@_cdecl("fluidaudio_is_apple_silicon")
public func fluidaudio_is_apple_silicon() -> Int32 {
    return SystemInfo.isAppleSilicon ? 1 : 0
}

@_cdecl("fluidaudio_cleanup")
public func fluidaudio_cleanup(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    bridge.cleanup()
}

@_cdecl("fluidaudio_free_string")
public func fluidaudio_free_string(_ s: UnsafeMutablePointer<CChar>?) {
    free(s)
}

// MARK: - System Info (extended)

@_cdecl("fluidaudio_is_intel_mac")
public func fluidaudio_is_intel_mac() -> Int32 {
    return SystemInfo.isIntelMac ? 1 : 0
}

// MARK: - VAD processing FFI

/// Process an audio file through VAD. Returns the number of frames via outCount.
/// Each frame contributes one entry to outProbabilities (Float), outIsVoiceActive (UInt8 0/1),
/// and outProcessingTimes (Double). Caller must free with fluidaudio_free_vad_result.
@_cdecl("fluidaudio_vad_process_file")
public func fluidaudio_vad_process_file(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outProbabilities: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outIsVoiceActive: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    _ outProcessingTimes: UnsafeMutablePointer<UnsafeMutablePointer<Double>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let frames = try bridge.vadProcessFile(pathString)
        emitVadFrames(
            frames,
            outProbabilities: outProbabilities,
            outIsVoiceActive: outIsVoiceActive,
            outProcessingTimes: outProcessingTimes,
            outCount: outCount
        )
        return 0
    } catch {
        print("VAD process file error: \(error)")
        return -1
    }
}

/// Process raw 16kHz mono Float32 samples through VAD. See fluidaudio_vad_process_file
/// for output array semantics.
@_cdecl("fluidaudio_vad_process_samples")
public func fluidaudio_vad_process_samples(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: UInt32,
    _ outProbabilities: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outIsVoiceActive: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    _ outProcessingTimes: UnsafeMutablePointer<UnsafeMutablePointer<Double>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(count)))

    do {
        let frames = try bridge.vadProcessSamples(samplesArray)
        emitVadFrames(
            frames,
            outProbabilities: outProbabilities,
            outIsVoiceActive: outIsVoiceActive,
            outProcessingTimes: outProcessingTimes,
            outCount: outCount
        )
        return 0
    } catch {
        print("VAD process samples error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_free_vad_result")
public func fluidaudio_free_vad_result(
    _ probabilities: UnsafeMutablePointer<Float>?,
    _ isVoiceActive: UnsafeMutablePointer<UInt8>?,
    _ processingTimes: UnsafeMutablePointer<Double>?,
    _ count: UInt32
) {
    _ = count // Reserved for future per-element cleanup (none currently needed).
    probabilities?.deallocate()
    isVoiceActive?.deallocate()
    processingTimes?.deallocate()
}

private func emitVadFrames(
    _ frames: [FluidAudioBridgeInternal.BridgeVadFrame],
    outProbabilities: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    outIsVoiceActive: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    outProcessingTimes: UnsafeMutablePointer<UnsafeMutablePointer<Double>?>?,
    outCount: UnsafeMutablePointer<UInt32>?
) {
    let count = frames.count
    outCount?.pointee = UInt32(count)

    if count == 0 {
        outProbabilities?.pointee = nil
        outIsVoiceActive?.pointee = nil
        outProcessingTimes?.pointee = nil
        return
    }

    let probs = UnsafeMutablePointer<Float>.allocate(capacity: count)
    let voice = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
    let times = UnsafeMutablePointer<Double>.allocate(capacity: count)

    for (i, frame) in frames.enumerated() {
        probs[i] = frame.probability
        voice[i] = frame.isVoiceActive ? 1 : 0
        times[i] = frame.processingTime
    }

    outProbabilities?.pointee = probs
    outIsVoiceActive?.pointee = voice
    outProcessingTimes?.pointee = times
}

/// Marshal diarization segments into the four caller-owned C out-arrays (freed by
/// `fluidaudio_free_diarization_result`). Shared by the download and model-path FFI
/// entry points (the latter lives in `Diarize_ffi.swift`). `count == 0` yields NULL arrays.
func emitDiarizationSegments(
    _ segments: [BridgeDiarizationSegment],
    outSpeakerIds: UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?>?,
    outStartTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    outEndTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    outQualityScores: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    outCount: UnsafeMutablePointer<UInt32>?
) {
    let count = segments.count
    outCount?.pointee = UInt32(count)

    if count == 0 {
        outSpeakerIds?.pointee = nil
        outStartTimes?.pointee = nil
        outEndTimes?.pointee = nil
        outQualityScores?.pointee = nil
        return
    }

    let ids = UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>.allocate(capacity: count)
    let starts = UnsafeMutablePointer<Float>.allocate(capacity: count)
    let ends = UnsafeMutablePointer<Float>.allocate(capacity: count)
    let scores = UnsafeMutablePointer<Float>.allocate(capacity: count)

    for (i, seg) in segments.enumerated() {
        ids[i] = strdup(seg.speakerId)
        starts[i] = seg.startTime
        ends[i] = seg.endTime
        scores[i] = seg.qualityScore
    }

    outSpeakerIds?.pointee = ids
    outStartTimes?.pointee = starts
    outEndTimes?.pointee = ends
    outQualityScores?.pointee = scores
}

// MARK: - ITN (Inverse Text Normalization) FFI

/// Normalize a short ASR expression (e.g. "two hundred thirty two" -> "232").
/// The returned string must be freed via fluidaudio_free_string.
@_cdecl("fluidaudio_itn_normalize")
public func fluidaudio_itn_normalize(
    _ ptr: UnsafeMutableRawPointer?,
    _ text: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let text = text else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let normalized = bridge.itnNormalize(String(cString: text))
    outText?.pointee = strdup(normalized)
    return 0
}

@_cdecl("fluidaudio_itn_normalize_sentence")
public func fluidaudio_itn_normalize_sentence(
    _ ptr: UnsafeMutableRawPointer?,
    _ text: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let text = text else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let normalized = bridge.itnNormalizeSentence(String(cString: text))
    outText?.pointee = strdup(normalized)
    return 0
}

@_cdecl("fluidaudio_itn_normalize_sentence_max_span")
public func fluidaudio_itn_normalize_sentence_max_span(
    _ ptr: UnsafeMutableRawPointer?,
    _ text: UnsafePointer<CChar>?,
    _ maxSpanTokens: UInt32,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let text = text else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let normalized = bridge.itnNormalizeSentenceMaxSpan(String(cString: text), maxSpanTokens: maxSpanTokens)
    outText?.pointee = strdup(normalized)
    return 0
}

@_cdecl("fluidaudio_itn_is_native_available")
public func fluidaudio_itn_is_native_available(
    _ ptr: UnsafeMutableRawPointer?
) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.itnIsNativeAvailable() ? 1 : 0
}
