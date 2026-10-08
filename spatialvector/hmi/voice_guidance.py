"""
M13 — SpatialVector Voice Guidance Engine

Production-quality, low-latency, non-blocking Text-to-Speech subsystem for
the SpatialVector-HMI navigation assistant.

Design goals
────────────
1. Always uses the Windows *default* audio output device automatically
   (speakers, Bluetooth earbuds, wired earphones — whichever is selected in
   Windows Sound settings). Zero code changes required on device switch.

2. High-quality natural voice via Windows System.Speech.Synthesis
   (SAPI 5 / .NET) which produces human-like speech significantly better than
   eSpeak or pyttsx3's basic SAPI binding.  Rate, Volume and Voice are tuned
   for clarity at a demonstration distance.

3. Dedicated background worker thread — speech generation NEVER blocks the
   main perception pipeline.

4. Intelligent priority queue — safety-critical messages (STOP / WARNING) always
   jump to the front of the queue and flush stale lower-priority items.

5. Cooldown logic — identical phrases are not repeated within their cooldown
   window, preventing auditory fatigue while still re-alarming for sustained
   hazards.

6. Confidence filtering — caller may supply a confidence score; instructions
   below the threshold are suppressed to avoid spurious alerts from unstable
   detections.

7. Structured console + dashboard log — every spoken phrase is emitted to the
   Python logger so it appears in both the console and any log-file handler.

Architecture
────────────
  VoiceGuidanceEngine
    ├─ _worker_thread  (daemon, always running)
    │    └─ pulls (priority, sequence_number, phrase) from _pq (PriorityQueue)
    │         └─ calls _speak_blocking(phrase) via PowerShell / SAPI
    └─ speak(phrase, priority, confidence)
         └─ inserts into _pq after cooldown + confidence checks

Priority levels (lower number = higher priority)
  PRIORITY_CRITICAL = 0   e.g. "Stop", "Warning. Obstacle approaching."
  PRIORITY_WARNING  = 1   e.g. "Obstacle ahead. Move left."
  PRIORITY_INFO     = 2   e.g. "Move forward."

Phrase catalogue — canonical human-readable phrases keyed by symbolic constant.
Callers should use the module-level speak_*() helpers or the engine's
.speak() method with one of the PHRASE_* constants.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from typing import List, Optional

logger = logging.getLogger("SpatialVector.VoiceGuidance")

# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------
PRIORITY_CRITICAL: int = 0
PRIORITY_WARNING: int = 1
PRIORITY_INFO: int = 2

# ---------------------------------------------------------------------------
# Canonical phrase catalogue
# ---------------------------------------------------------------------------
PHRASE_STOP                  = "Stop."
PHRASE_WARNING_APPROACHING   = "Warning. Obstacle approaching."
PHRASE_OBSTACLE_MOVE_LEFT    = "Obstacle ahead. Move left."
PHRASE_OBSTACLE_MOVE_RIGHT   = "Obstacle ahead. Move right."
PHRASE_MOVE_FORWARD          = "Move forward."
PHRASE_PERSON_FROM_LEFT      = "Person approaching from your left."
PHRASE_PERSON_FROM_RIGHT     = "Person approaching from your right."
PHRASE_CAUTION               = "Caution. Slow down."
PHRASE_ALL_CLEAR             = "Path is clear."

# Map phrase -> its natural priority level
_PHRASE_DEFAULT_PRIORITY: dict = {
    PHRASE_STOP:                PRIORITY_CRITICAL,
    PHRASE_WARNING_APPROACHING: PRIORITY_CRITICAL,
    PHRASE_OBSTACLE_MOVE_LEFT:  PRIORITY_WARNING,
    PHRASE_OBSTACLE_MOVE_RIGHT: PRIORITY_WARNING,
    PHRASE_PERSON_FROM_LEFT:    PRIORITY_WARNING,
    PHRASE_PERSON_FROM_RIGHT:   PRIORITY_WARNING,
    PHRASE_CAUTION:             PRIORITY_WARNING,
    PHRASE_MOVE_FORWARD:        PRIORITY_INFO,
    PHRASE_ALL_CLEAR:           PRIORITY_INFO,
}

# Per-phrase cooldown in seconds (minimum interval before the same phrase repeats)
_PHRASE_COOLDOWN_S: dict = {
    PHRASE_STOP:                1.2,
    PHRASE_WARNING_APPROACHING: 1.8,
    PHRASE_OBSTACLE_MOVE_LEFT:  2.5,
    PHRASE_OBSTACLE_MOVE_RIGHT: 2.5,
    PHRASE_PERSON_FROM_LEFT:    3.0,
    PHRASE_PERSON_FROM_RIGHT:   3.0,
    PHRASE_CAUTION:             3.5,
    PHRASE_MOVE_FORWARD:        4.0,
    PHRASE_ALL_CLEAR:           5.0,
}

# Global minimum gap between ANY two speech events (avoids phrase overlap)
_MIN_INTER_SPEECH_GAP_S: float = 1.0

# Default confidence threshold — detections below this are suppressed
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40


class VoiceGuidanceEngine:
    """Non-blocking, priority-aware Text-to-Speech engine for SpatialVector.

    Usage::

        engine = VoiceGuidanceEngine()
        engine.start()

        # Inside perception loop:
        engine.speak(PHRASE_OBSTACLE_MOVE_LEFT, confidence=0.87)
        engine.speak(PHRASE_STOP, confidence=0.95)   # jumps the queue

        engine.stop()

    Thread safety: all public methods are thread-safe.
    """

    def __init__(
        self,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        speech_rate: int = 1,
        speech_volume: int = 100,
        max_queue_size: int = 6,
        voice_name: Optional[str] = None,
    ):
        """
        Args:
            confidence_threshold: detections below this confidence are ignored.
            speech_rate:  SAPI rate, -2 (slow) to +3 (fast). Default 1 = natural.
            speech_volume: SAPI volume 0-100.
            max_queue_size: maximum pending phrases before old INFO items drop.
            voice_name: partial name of SAPI voice to prefer (e.g. "Zira", "David").
                        None = use OS default voice.
        """
        self.confidence_threshold = confidence_threshold
        self.speech_rate = speech_rate
        self.speech_volume = speech_volume
        self.voice_name = voice_name

        # Priority queue entries: (priority, sequence_number, phrase)
        # sequence_number breaks priority ties so FIFO order is preserved within a tier
        self._pq: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._seq_counter: int = 0
        self._seq_lock = threading.Lock()

        # Cooldown tracking: phrase -> last-spoken monotonic timestamp
        self._last_spoken: dict = {}
        self._last_spoken_lock = threading.Lock()

        # Global gap tracking
        self._last_any_speech_t: float = 0.0
        self._last_any_lock = threading.Lock()

        # Most recently dispatched phrase (for dashboard display)
        self._last_dispatched_phrase: str = ""
        self._last_dispatched_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the background speech worker thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="SV-VoiceGuidance",
        )
        self._worker_thread.start()
        logger.info(
            "[M13] Voice guidance engine started — "
            "using Windows default audio output device (speakers / Bluetooth / wired)"
        )

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        # Unblock the worker's blocking queue.get()
        try:
            self._pq.put_nowait((PRIORITY_INFO, 0, "__STOP__"))
        except queue.Full:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("[M13] Voice guidance engine stopped")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def speak(
        self,
        phrase: str,
        priority: Optional[int] = None,
        confidence: float = 1.0,
        force: bool = False,
    ) -> bool:
        """Request speech output.

        Args:
            phrase:     Text to speak. Use PHRASE_* constants from this module.
            priority:   Override default priority. Lower number = more urgent.
                        If None, uses the catalogue default or PRIORITY_INFO.
            confidence: Detection confidence (0..1). Suppressed below threshold.
            force:      Bypass cooldown check (use for one-time critical alerts).

        Returns:
            True  if the phrase was enqueued for speech.
            False if suppressed (confidence, cooldown, or queue full).
        """
        if not phrase or phrase == "__STOP__":
            return False

        # 1. Confidence gate
        if not force and confidence < self.confidence_threshold:
            logger.debug(
                f"[M13] Suppressed (conf={confidence:.2f} < {self.confidence_threshold:.2f}): "
                f"'{phrase}'"
            )
            return False

        # 2. Resolve priority
        if priority is None:
            priority = _PHRASE_DEFAULT_PRIORITY.get(phrase, PRIORITY_INFO)

        # 3. Cooldown gate
        if not force:
            cooldown = _PHRASE_COOLDOWN_S.get(phrase, 3.0)
            now = time.monotonic()
            with self._last_spoken_lock:
                last_t = self._last_spoken.get(phrase, 0.0)
                if (now - last_t) < cooldown:
                    logger.debug(f"[M13] Cooldown active ({cooldown:.1f}s): '{phrase}'")
                    return False
                self._last_spoken[phrase] = now

        # 4. For CRITICAL / WARNING: flush stale INFO items to make room
        if priority <= PRIORITY_WARNING:
            self._flush_low_priority()

        # 5. Enqueue
        with self._seq_lock:
            self._seq_counter += 1
            seq = self._seq_counter

        try:
            self._pq.put_nowait((priority, seq, phrase))
            logger.info(f"[M13] Queued    [P{priority}]: \"{phrase}\"")
            return True
        except queue.Full:
            if priority == PRIORITY_CRITICAL:
                # Forcibly make room for critical phrases
                try:
                    self._pq.get_nowait()
                    self._pq.put_nowait((priority, seq, phrase))
                    logger.info(f"[M13] Force-enqueued CRITICAL: \"{phrase}\"")
                    return True
                except Exception:
                    pass
            logger.debug(f"[M13] Queue full — dropped: '{phrase}'")
            return False

    def get_last_phrase(self) -> str:
        """Return the most recently spoken phrase (thread-safe, for dashboard)."""
        with self._last_dispatched_lock:
            return self._last_dispatched_phrase

    # -----------------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------------

    def announce_stop(self, confidence: float = 1.0) -> bool:
        """All corridors blocked — immediate stop."""
        return self.speak(PHRASE_STOP, PRIORITY_CRITICAL, confidence)

    def announce_warning_approaching(self, confidence: float = 1.0) -> bool:
        """Fast-approaching obstacle detected."""
        return self.speak(PHRASE_WARNING_APPROACHING, PRIORITY_CRITICAL, confidence)

    def announce_obstacle_move_left(self, confidence: float = 1.0) -> bool:
        """Obstacle ahead — left corridor is safer."""
        return self.speak(PHRASE_OBSTACLE_MOVE_LEFT, PRIORITY_WARNING, confidence)

    def announce_obstacle_move_right(self, confidence: float = 1.0) -> bool:
        """Obstacle ahead — right corridor is safer."""
        return self.speak(PHRASE_OBSTACLE_MOVE_RIGHT, PRIORITY_WARNING, confidence)

    def announce_move_forward(self, confidence: float = 1.0) -> bool:
        """Safe path directly ahead."""
        return self.speak(PHRASE_MOVE_FORWARD, PRIORITY_INFO, confidence)

    def announce_person_from_left(self, confidence: float = 1.0) -> bool:
        """Crossing pedestrian approaching from user's left."""
        return self.speak(PHRASE_PERSON_FROM_LEFT, PRIORITY_WARNING, confidence)

    def announce_person_from_right(self, confidence: float = 1.0) -> bool:
        """Crossing pedestrian approaching from user's right."""
        return self.speak(PHRASE_PERSON_FROM_RIGHT, PRIORITY_WARNING, confidence)

    def announce_caution(self, confidence: float = 1.0) -> bool:
        """Risk is elevated but corridor is not yet fully blocked."""
        return self.speak(PHRASE_CAUTION, PRIORITY_WARNING, confidence)

    def announce_all_clear(self, confidence: float = 1.0) -> bool:
        """No obstacles detected — path is fully clear."""
        return self.speak(PHRASE_ALL_CLEAR, PRIORITY_INFO, confidence)

    # -----------------------------------------------------------------------
    # High-level decision -> voice translation
    # -----------------------------------------------------------------------

    def announce_from_risk_state(
        self,
        risk_state_str: str,
        direction: str,
        global_risk: float,
        tracks: List,
        predictions: List,
        confidence: float = 1.0,
    ) -> bool:
        """Translate M08/M09 outputs to the best voice instruction.

        Decision order (highest priority first):
          1. Fast-approaching obstacle (high expansion rate or very low TTC)
          2. All-corridors blocked (CRITICAL + STOP or risk >= 0.80)
          3. Crossing pedestrian (lateral velocity dominant)
          4. Directional obstacle warning (LEFT / RIGHT corridor redirect)
          5. Generic caution (WARNING/CAUTION risk states)
          6. Move forward (SAFE, low risk)

        Args:
            risk_state_str: RiskState.state string
            direction:      HapticCommand.direction ("LEFT"|"RIGHT"|"CENTER"|"STOP")
            global_risk:    RiskState.global_risk (0..1)
            tracks:         List[Track] for pedestrian crossing detection
            predictions:    List[Prediction] for fast-approach detection
            confidence:     Overall pipeline confidence to gate the announcement

        Returns:
            True if any announcement was enqueued.
        """
        # 1. Fast-approaching obstacle
        for pred in predictions:
            exp_rate = getattr(pred, "expansion_rate", 0.0)
            ttc = getattr(pred, "ttc_s", None)
            p_conf = getattr(pred, "prediction_confidence", confidence)
            if exp_rate > 0.25 or (ttc is not None and ttc < 1.5):
                return self.announce_warning_approaching(
                    confidence=min(confidence, p_conf)
                )

        # 2. Stop — all corridors blocked
        if (risk_state_str == "CRITICAL" and direction == "STOP") or global_risk >= 0.80:
            return self.announce_stop(confidence=confidence)

        # 3. Crossing pedestrian detection
        for track in tracks:
            cls = track.class_name.lower()
            if cls in ("person", "man", "woman", "child", "pedestrian"):
                vx, vy = getattr(track, "estimated_image_velocity", (0.0, 0.0))
                track_conf = getattr(track, "track_confidence", confidence)
                # Lateral velocity significantly exceeds vertical -> crossing behaviour
                if abs(vx) > abs(vy) * 1.5 and abs(vx) > 8.0:
                    effective_conf = min(confidence, track_conf)
                    if vx > 0:
                        # Moving rightward in image -> approaching from user's LEFT
                        return self.announce_person_from_left(confidence=effective_conf)
                    else:
                        # Moving leftward in image -> approaching from user's RIGHT
                        return self.announce_person_from_right(confidence=effective_conf)

        # 4. Directional redirect
        if direction == "LEFT" and global_risk > 0.40:
            return self.announce_obstacle_move_left(confidence=confidence)
        if direction == "RIGHT" and global_risk > 0.40:
            return self.announce_obstacle_move_right(confidence=confidence)

        # 5. Generic caution
        if risk_state_str in ("WARNING", "CAUTION") and global_risk > 0.30:
            return self.announce_caution(confidence=confidence)

        # 6. Safe path — move forward
        if risk_state_str == "SAFE" and global_risk < 0.25:
            return self.announce_move_forward(confidence=confidence)

        return False

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _flush_low_priority(self) -> None:
        """Drain INFO-priority items from the queue to make room for urgent ones."""
        kept: list = []
        flushed_count: int = 0

        while True:
            try:
                item = self._pq.get_nowait()
                if item[0] <= PRIORITY_WARNING:
                    kept.append(item)
                else:
                    flushed_count += 1
            except queue.Empty:
                break

        for item in kept:
            try:
                self._pq.put_nowait(item)
            except queue.Full:
                break

        if flushed_count > 0:
            logger.debug(f"[M13] Flushed {flushed_count} INFO-priority item(s) from queue")

    def _worker_loop(self) -> None:
        """Background thread: consumes the priority queue and drives Windows SAPI."""
        while not self._stop_event.is_set():
            try:
                item = self._pq.get(timeout=0.25)
            except queue.Empty:
                continue

            _priority, _seq, phrase = item

            if phrase == "__STOP__":
                break

            # Enforce minimum inter-speech gap to avoid overlapping audio
            with self._last_any_lock:
                now = time.monotonic()
                gap = now - self._last_any_speech_t
                if gap < _MIN_INTER_SPEECH_GAP_S:
                    time.sleep(_MIN_INTER_SPEECH_GAP_S - gap)
                self._last_any_speech_t = time.monotonic()

            # Update dashboard-visible last phrase
            with self._last_dispatched_lock:
                self._last_dispatched_phrase = phrase

            logger.info(f"[M13] Speaking  [P{_priority}]: \"{phrase}\"")
            print(f"[VOICE] {phrase}")   # visible in console / dashboard log
            self._speak_blocking(phrase)

    def _speak_blocking(self, phrase: str) -> None:
        """Blocking Windows SAPI call via PowerShell.

        System.Speech.Synthesis automatically routes audio to the currently
        selected Windows default output device — laptop speakers, Bluetooth
        earbuds, wired earphones, USB headset, HDMI, etc.  No code changes are
        needed when the user switches the default audio device in Windows
        Sound settings.

        Speech quality is significantly better than eSpeak/festival because
        SAPI 5 ships with Microsoft's own neural or HQ concatenative voices.
        """
        # Sanitise the phrase for safe embedding in a PowerShell string
        clean = (
            phrase
            .replace("'", " ")
            .replace('"', " ")
            .replace(";", " ")
            .replace("&", "and")
            .strip()
        )
        if not clean:
            return

        # Optionally select a preferred installed voice
        voice_selection = ""
        if self.voice_name:
            voice_selection = (
                f"$voices = $synth.GetInstalledVoices(); "
                f"$pref = $voices | "
                f"Where-Object {{ $_.VoiceInfo.Name -like '*{self.voice_name}*' }} | "
                f"Select-Object -First 1; "
                f"if ($pref) {{ $synth.SelectVoice($pref.VoiceInfo.Name); }}"
            )

        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = {self.speech_rate}; "
            f"$synth.Volume = {self.speech_volume}; "
            f"{voice_selection} "
            f"$synth.Speak('{clean}');"
        )

        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=8.0,
            )
        except subprocess.TimeoutExpired:
            logger.warning("[M13] Speech timed out — skipping phrase")
        except FileNotFoundError:
            logger.error(
                "[M13] PowerShell not found — voice guidance unavailable on this platform"
            )
        except Exception as exc:
            logger.debug(f"[M13] Speech error: {exc}")


# ---------------------------------------------------------------------------
# Module-level singleton convenience API
# ---------------------------------------------------------------------------

_global_engine: Optional[VoiceGuidanceEngine] = None
_global_lock = threading.Lock()


def get_engine(
    *,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    speech_rate: int = 1,
    speech_volume: int = 100,
    voice_name: Optional[str] = None,
    auto_start: bool = True,
) -> VoiceGuidanceEngine:
    """Return the global shared VoiceGuidanceEngine, creating it on first call.

    This is a convenience factory for callers that do not manage the engine
    lifecycle explicitly.

    Args:
        auto_start: if True, calls engine.start() on creation.

    Returns:
        The single shared VoiceGuidanceEngine instance.
    """
    global _global_engine
    with _global_lock:
        if _global_engine is None:
            _global_engine = VoiceGuidanceEngine(
                confidence_threshold=confidence_threshold,
                speech_rate=speech_rate,
                speech_volume=speech_volume,
                voice_name=voice_name,
            )
            if auto_start:
                _global_engine.start()
    return _global_engine


def shutdown_engine() -> None:
    """Stop and discard the global engine if it was started."""
    global _global_engine
    with _global_lock:
        if _global_engine is not None:
            _global_engine.stop()
            _global_engine = None
