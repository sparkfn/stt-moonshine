"""Server-side VAD integration tests.

Verifies that the server's StreamingVAD actually detects speech boundaries
and emits is_final events WITHOUT the client calling flush.

These tests exist because a prior bug (StreamingVAD created before model
loading) made server VAD 100% non-functional for weeks without any test
catching it. Every other WebSocket test uses flush(), which bypasses VAD.

Run:
    pytest E2Etest/test_vad.py -v
    pytest E2Etest/ -m vad -v
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from utils.client import ASRWebSocketClient
from utils.audio import generate_silence, convert_to_int16


SAMPLE_RATE = 16000
# Server's WS_BUFFER_SIZE default is 6400 bytes = 3200 samples = 200ms
CHUNK_SAMPLES = 3200
CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16 = 2 bytes per sample


def _load_real_speech(audio_dir: Path, lang: str = "english", idx: int = 1) -> np.ndarray:
    """Load a real speech WAV file and return int16 PCM at 16kHz mono."""
    wav = audio_dir / "real" / f"{lang}_{idx:02d}.wav"
    if not wav.exists():
        pytest.skip(
            f"Real speech audio not found: {wav}\n"
            "Run: python E2Etest/download_test_audio.py"
        )
    audio, sr = sf.read(str(wav), dtype="int16")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        new_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.int16)
    return audio


def _chunk_audio(audio_int16: np.ndarray, chunk_samples: int = CHUNK_SAMPLES) -> list[bytes]:
    """Split int16 audio into chunk-sized byte buffers."""
    return [
        audio_int16[i:i + chunk_samples].tobytes()
        for i in range(0, len(audio_int16), chunk_samples)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.websocket
@pytest.mark.vad
class TestServerVAD:
    """Tests that server-side VAD actually works end-to-end."""

    @pytest.mark.asyncio
    async def test_handshake_reports_vad_enabled(
        self, ws_url: str, ensure_server,
    ):
        """Server handshake accurately reports use_server_vad status."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            assert "use_server_vad" in info, (
                "Handshake missing use_server_vad field"
            )
            # If VAD model loaded successfully, it should be True
            # If it failed, it should be False (not a lie)
            assert isinstance(info["use_server_vad"], bool)

    @pytest.mark.asyncio
    async def test_vad_emits_speech_end_without_flush(
        self, ws_url: str, ensure_server, audio_dir: Path,
    ):
        """Server VAD detects speech end and sends is_final WITHOUT client flush.

        This is THE critical test. It would have caught the init-order bug
        where StreamingVAD was created before models loaded (self._vad = None).
        """
        speech = _load_real_speech(audio_dir, "english")
        # Use first 3 seconds of speech (enough to satisfy min_speech_ms=250)
        speech_3s = speech[:SAMPLE_RATE * 3]
        speech_chunks = _chunk_audio(speech_3s)

        # 1.5 seconds of silence to trigger speech_end (redemption_ms=400)
        silence = np.zeros(SAMPLE_RATE * 3 // 2, dtype=np.int16)
        silence_chunks = _chunk_audio(silence)

        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            if not info.get("use_server_vad"):
                pytest.skip("Server VAD is disabled or unavailable")

            # Stream speech with real-time pacing
            for chunk in speech_chunks:
                await client.send_audio(chunk)
                await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)  # 200ms

            # Stream silence — should trigger VAD speech_end
            for chunk in silence_chunks:
                await client.send_audio(chunk)
                await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)

            # Wait for VAD-triggered is_final — DO NOT FLUSH
            vad_final = None
            deadline = asyncio.get_event_loop().time() + 5.0
            while asyncio.get_event_loop().time() < deadline:
                try:
                    msg = await asyncio.wait_for(client.receive(), timeout=1.0)
                    if msg.get("is_final") is True:
                        vad_final = msg
                        break
                except asyncio.TimeoutError:
                    continue

            assert vad_final is not None, (
                "Server VAD never sent is_final. speech_end did not fire. "
                "Check that StreamingVAD._vad is not None and that silence "
                "frames are being processed."
            )
            # VAD final should have actual transcription text
            assert vad_final.get("text", "").strip(), (
                "VAD is_final arrived but text is empty — speech was not transcribed"
            )

    @pytest.mark.asyncio
    async def test_vad_does_not_fire_on_silence_only(
        self, ws_url: str, ensure_server,
    ):
        """Sending only silence should NOT trigger a VAD speech_end event.

        speech_end requires prior speech_start (min_speech_ms worth of speech).
        Pure silence should produce no is_final.
        """
        silence = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)
        silence_chunks = _chunk_audio(silence)

        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            if not info.get("use_server_vad"):
                pytest.skip("Server VAD is disabled or unavailable")

            for chunk in silence_chunks:
                await client.send_audio(chunk)
                await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)

            # Wait briefly — no is_final should arrive
            got_final = False
            try:
                for _ in range(3):
                    msg = await asyncio.wait_for(client.receive(), timeout=0.5)
                    if msg.get("is_final") is True:
                        got_final = True
                        break
            except asyncio.TimeoutError:
                pass

            assert not got_final, (
                "Server sent is_final on silence-only audio — "
                "VAD speech_end should require prior speech"
            )

    @pytest.mark.asyncio
    async def test_vad_speech_then_timeout_emits_final(
        self, ws_url: str, ensure_server, audio_dir: Path,
    ):
        """When client stops sending after speech, server timeout forces is_final.

        Tests the _VAD_RECV_TIMEOUT fallback: if client stops sending audio
        while VAD speech is active, the server should force speech_end after
        ~600ms rather than waiting for the client's flush watchdog.
        """
        speech = _load_real_speech(audio_dir, "english")
        # Send 2 seconds of speech then STOP sending entirely
        speech_2s = speech[:SAMPLE_RATE * 2]
        speech_chunks = _chunk_audio(speech_2s)

        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            if not info.get("use_server_vad"):
                pytest.skip("Server VAD is disabled or unavailable")

            for chunk in speech_chunks:
                await client.send_audio(chunk)
                await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)

            # Stop sending. Server should timeout after _VAD_RECV_TIMEOUT (600ms)
            # and force is_final.
            vad_final = None
            try:
                # Allow up to 3s — but expect it within ~1s (600ms timeout + inference)
                msg = await asyncio.wait_for(client.receive(), timeout=3.0)
                while True:
                    if msg.get("is_final") is True:
                        vad_final = msg
                        break
                    msg = await asyncio.wait_for(client.receive(), timeout=3.0)
            except asyncio.TimeoutError:
                pass

            assert vad_final is not None, (
                "Server did not emit is_final after client stopped sending. "
                "The _VAD_RECV_TIMEOUT fallback did not fire."
            )

    @pytest.mark.asyncio
    async def test_vad_multiple_utterances(
        self, ws_url: str, ensure_server, audio_dir: Path,
    ):
        """VAD can detect multiple speech segments in one session.

        After speech_end, VAD resets and can detect a new utterance.
        """
        speech = _load_real_speech(audio_dir, "english")
        # First utterance: 2s speech
        utt1 = speech[:SAMPLE_RATE * 2]
        # Second utterance: next 2s speech
        utt2_start = min(SAMPLE_RATE * 4, len(speech) - SAMPLE_RATE * 2)
        utt2 = speech[utt2_start:utt2_start + SAMPLE_RATE * 2]

        silence_1s = np.zeros(SAMPLE_RATE * 1, dtype=np.int16)

        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            if not info.get("use_server_vad"):
                pytest.skip("Server VAD is disabled or unavailable")

            finals_received = 0

            async def send_and_wait_final(audio_int16: np.ndarray) -> dict | None:
                """Send audio + silence, wait for VAD final."""
                for chunk in _chunk_audio(audio_int16):
                    await client.send_audio(chunk)
                    await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)
                # Send silence to trigger speech_end
                for chunk in _chunk_audio(silence_1s):
                    await client.send_audio(chunk)
                    await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)
                # Collect final
                deadline = asyncio.get_event_loop().time() + 5.0
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        msg = await asyncio.wait_for(client.receive(), timeout=1.0)
                        if msg.get("is_final") is True:
                            return msg
                    except asyncio.TimeoutError:
                        continue
                return None

            final1 = await send_and_wait_final(utt1)
            if final1:
                finals_received += 1

            # Small gap between utterances
            await asyncio.sleep(0.3)

            final2 = await send_and_wait_final(utt2)
            if final2:
                finals_received += 1

            assert finals_received >= 1, (
                "VAD did not emit any is_final for two utterances. "
                "Server VAD may not be functional."
            )

    @pytest.mark.asyncio
    async def test_vad_config_toggle(
        self, ws_url: str, ensure_server,
    ):
        """Can disable and re-enable server VAD via config command."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            original_vad = info.get("use_server_vad", False)

            # Disable VAD
            response = await client.config("auto")
            # Send config with use_server_vad=false
            await client.websocket.send(json.dumps({
                "action": "config",
                "use_server_vad": False,
            }))
            msg = await asyncio.wait_for(client.receive(), timeout=5)
            assert msg.get("status") == "configured"
            assert msg.get("use_server_vad") is False

            # Re-enable VAD
            await client.websocket.send(json.dumps({
                "action": "config",
                "use_server_vad": True,
            }))
            msg = await asyncio.wait_for(client.receive(), timeout=5)
            assert msg.get("status") == "configured"
            assert msg.get("use_server_vad") is True
