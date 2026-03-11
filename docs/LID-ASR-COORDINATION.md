# LID → ASR Language Coordination

Date: 2026-03-10

## Context

Audio-exchange (AX) performs real-time language identification (LID) on phone calls and emits lock/switch events to voice-platform (VP). VP relays the detected language to Moonshine ASR via WebSocket `{"action": "config", "language": "zh"}`. ASR uses this to route transcription to the correct model (`transcribe_en` or `transcribe_zh`).

This document analyzes the timing coordination between the three components and identifies gaps.

## Architecture

```
AX (audio-exchange)          VP (voice-platform)           ASR (moonshine)
─────────────────            ──────────────────            ──────────────
AudioSocket TCP              WebSocket /sts                WebSocket /ws/transcribe
8kHz → 16kHz resample        Forwards 16kHz PCM            Accumulates in buffer
VAD (TEN)                    Forwards vad/lid events        Transcribes on flush
LID (AmberNet ONNX)          Calls asr.setLanguage()       Routes to en/zh model
                             Calls asr.flush() on vad-end
```

## How Language Reaches ASR

1. **On WS connect**: VP sends `{"action":"config", "language": <default>}` — currently `config.asrDefaultLanguage` (typically `"en"`)
2. **On `lid_locked`**: VP calls `asr.setLanguage("zh")` → sends `{"action":"config", "language":"zh"}` over existing WS
3. **On `lid_switch`**: Same as above with new language
4. **Tentative `lid` events**: VP ignores them (debug log only). ASR never sees them.

## ASR Buffer Behavior (WS_PARTIAL=false)

In buffer-only mode (`WS_PARTIAL=false`), ASR does NOT transcribe incrementally:
- Binary audio frames → `audio_buffer.extend(incoming)` (server.py line 700)
- No transcription happens until `{"action":"flush"}` is received
- On flush: `audio_window.extend(audio_buffer)` → `_transcribe_with_context(audio_window, lang_code=lang_code)` → clear
- `lang_code` is a local variable in the WS handler, updated by `config` action (server.py lines 608-613)
- **Config does NOT clear the buffer, retranscribe, or trigger any side effects — it's a pure variable assignment**

> **Note:** ASR has no built-in VAD or LID. All language detection and voice activity detection is expected to happen upstream (e.g., in audio-exchange). ASR purely routes to EN/ZH model based on the `lang_code` set by the client.

## Timeline: Typical Call (Chinese Speaker, > 2s utterance)

```
T+0.0s  AX connects → VP creates AsrClient(language=undefined → "en")
        ASR WS open → config: lang_code = "en"

T+0.1s  AX VAD onset confirmed (100ms)
        AX sends 16kHz PCM → VP → ASR buffer
        ASR: audio_buffer growing, lang_code = "en", no transcription

T+1.0s  AX LID check #1: zh=0.95, gap=0.90 → candidate #1 (need 2 consecutive)
        AX sends tentative "lid" event → VP debug log only
        ASR: still buffering, lang_code still "en"

T+2.0s  AX LID check #2: zh=0.98, gap=0.96 → agrees → LOCKED "zh"
        AX sends "lid_locked" → VP calls asr.setLanguage("zh")
        ASR receives config → lang_code = "zh"
        ASR buffer: ~2s of Chinese audio, NOT yet transcribed

T+3-5s  Speaker pauses → AX VAD speech-end (hangover expires)
        AX sends vad:speech=false → VP calls asr.flush()
        ASR receives flush → transcribes buffer with lang_code="zh"
        → transcribe_zh() ← CORRECT
```

**Result: Works correctly.** LID locks (T+2s) before first flush (T+3-5s). Audio buffer transcribed with correct language.

## Timeline: Short Utterance (< 2s, e.g. "Wei?")

```
T+0.0s  ASR: lang_code = "en"

T+0.1s  Speaker: "Wei?" (Chinese greeting, ~0.5s)
        AX sends audio → VP → ASR buffer

T+1.3s  AX VAD speech-end (0.5s speech + 0.8s hangover)
        AX sends vad:speech=false → VP calls asr.flush()
        ASR: transcribes 0.5s buffer with lang_code="en" ← WRONG
        LID hasn't locked yet (needs 2s minimum: 1s audio + 2 checks)

T+2.0s  LID check would happen here, but speech already ended and buffer flushed
```

**Result: WRONG LANGUAGE.** First utterance transcribed as English because flush happened before LID lock.

## Timeline: Language Switch Mid-Call

```
T+0s    Locked on "en", speaker switches to Chinese

T+1s    AX LID check: zh=0.99 → switch vote #1 (need 3 votes)
T+2s    AX LID check: zh=0.98 → switch vote #2
T+3s    AX LID check: zh=1.00 → switch vote #3 → SWITCH en→zh
        VP: asr.setLanguage("zh")

T+4s    Speaker pauses → flush
        ASR: transcribes with lang_code="zh" ← CORRECT (but 3s delay)
```

**Result: Correct, but with ~3s delay.** Audio during T+0-3s (the transition) is in the buffer. It's a mix of the tail of English and start of Chinese. Transcribed with "zh" model which may garble the English portion.

## What ASR Config Action Does (and doesn't do)

**Does:**
- Sets `lang_code` variable for the next transcription

**Does NOT:**
- Clear `audio_buffer` or `audio_window`
- Retranscribe any buffered audio
- Send acknowledgment that VP could use to coordinate
- Trigger any transcription

ASR is completely passive. It has no awareness of:
- Whether VP is waiting for LID to lock
- Whether the buffer contains audio in a different language than `lang_code`
- Whether a language switch happened mid-buffer

## Gap Analysis

| Scenario | LID Lock Time | First Flush Time | Language at Flush | Correct? |
|---|---|---|---|---|
| Long greeting (> 2s) + pause | ~2s | ~3-5s | Locked language | YES |
| Short greeting (< 2s) + pause | Never (insufficient audio) | ~1.3-2s | Default "en" | MAYBE |
| "Hello" then Chinese | ~2s (locks "en") | ~2-3s | "en" | YES (English portion) |
| Continuous speech > 5s | ~2s | When speaker pauses | Locked language | YES |
| Mid-call language switch | +3s from switch start | Next pause | New language | YES (with delay) |
| Rapid code-switching (< 3s per language) | May not switch | Various | May be stale | NO |

## Key Insight

**Buffer-only mode is actually an advantage here.** Because ASR never transcribes until flush, and flush only happens on VAD speech-end, LID almost always has enough time to lock before the first transcription. The typical call flow (greeting > 2s, then pause) naturally aligns with LID's lock timing.

The problematic case is ultra-short first utterances (< 2s). But these are rare in real phone conversations — callers typically say at least "Hello?" or "Wei?" followed by a pause, giving LID 2-3s of audio before flush.

## Recommendations for ASR

1. **No changes needed for normal flow** — buffer-only mode + upstream VAD-driven flush naturally coordinates with LID's 2s lock time
2. **Consider**: If `config` arrives while buffer is non-empty, ASR could log a warning — "language changed mid-buffer, N bytes may be in wrong language"
3. **Consider**: A new action `{"action": "config_and_reset"}` that atomically sets language AND clears the buffer — useful if VP wants to discard pre-lock audio
4. **Do NOT auto-clear buffer on config** — in the normal case (LID locks mid-utterance, same language throughout), clearing would lose valid audio

> **Architecture note (2026-03-09):** ASR no longer contains any VAD or LID components — these were stripped in commit `34150b8`. All voice activity detection and language identification happens upstream in audio-exchange (AX). ASR is a pure transcription service that routes to EN/ZH based on client-provided language config.
