# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import numpy as np
import pybase64 as base64
import pytest
import websockets

from tests.entrypoints.speech_to_text.conftest import add_attention_backend
from tests.utils import ROCM_ENV_OVERRIDES, ROCM_EXTRA_ARGS, RemoteOpenAIServer
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

# Increase engine iteration timeout for ROCm where first-use JIT compilation
# can exceed the default 60s, causing a silent deadlock in feed_tokens.
REALTIME_ENV_OVERRIDES = {
    **ROCM_ENV_OVERRIDES,
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "600",
}

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
] + ROCM_EXTRA_ARGS

MODEL_NAME = "mistralai/Voxtral-Mini-4B-Realtime-2602"

DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"
FAILED_EVENT = "conversation.item.input_audio_transcription.failed"


def _get_websocket_url(server: RemoteOpenAIServer) -> str:
    """Convert HTTP URL to WebSocket URL for realtime endpoint."""
    http_url = server.url_root
    ws_url = http_url.replace("http://", "ws://")
    return f"{ws_url}/v1/realtime"


def _session_update(model: str) -> dict:
    """OpenAI Realtime transcription session config for 16 kHz PCM."""
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "transcription": {"model": model},
                }
            },
        },
    }


async def receive_event(ws, timeout: float = 60.0) -> dict:
    """Receive and parse JSON event from WebSocket."""
    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(message)


async def send_event(ws, event: dict) -> None:
    """Send JSON event to WebSocket."""
    await ws.send(json.dumps(event))


async def configure_session(ws, model: str) -> None:
    """Open a session: consume session.created, send session.update,
    wait for session.updated."""
    event = await receive_event(ws, timeout=30.0)
    assert event["type"] == "session.created"

    await send_event(ws, _session_update(model))

    event = await receive_event(ws, timeout=10.0)
    assert event["type"] == "session.updated", event


@pytest.fixture
def mary_had_lamb_audio_chunks() -> list[str]:
    """Audio split into ~1 second chunks for streaming."""
    path = AudioAsset("mary_had_lamb").get_local_path()
    audio, _ = load_audio(str(path), sr=16000, mono=True)

    # Split into ~0.1 second chunks (1600 samples at 16kHz)
    chunk_size = 1600
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        chunk_int16 = (chunk * 32767).astype(np.int16)
        chunk_bytes = chunk_int16.tobytes()
        chunks.append(base64.b64encode(chunk_bytes).decode("utf-8"))

    return chunks


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_multi_chunk_streaming(
    model_name, mary_had_lamb_audio_chunks, rocm_aiter_fa_attention
):
    """Test streaming multiple audio chunks before committing."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            await configure_session(ws, model_name)

            # (ROCm) Warm-up: transcribe a small utterance to trigger
            # aiter compilation on first use.
            await send_event(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": mary_had_lamb_audio_chunks[0],
                },
            )
            await send_event(ws, {"type": "input_audio_buffer.commit"})

            # (ROCm) Drain all warm-up responses with generous timeout for
            # JIT compilation
            while True:
                event = await receive_event(ws, timeout=600.0)
                if event["type"] in (COMPLETED_EVENT, FAILED_EVENT, "error"):
                    break

            # Now send the real test audio
            for chunk in mary_had_lamb_audio_chunks:
                await send_event(
                    ws, {"type": "input_audio_buffer.append", "audio": chunk}
                )

            # Commit to end the utterance
            await send_event(ws, {"type": "input_audio_buffer.commit"})

            # Collect committed / created / deltas / completed
            full_text = ""
            committed_item_id = None
            done_received = False

            while not done_received:
                event = await receive_event(ws, timeout=60.0)

                if event["type"] == "input_audio_buffer.committed":
                    committed_item_id = event["item_id"]
                elif event["type"] == "conversation.item.created":
                    assert event["item"]["id"] == committed_item_id
                elif event["type"] == DELTA_EVENT:
                    full_text += event["delta"]
                elif event["type"] == COMPLETED_EVENT:
                    done_received = True
                    assert "transcript" in event
                elif event["type"] in (FAILED_EVENT, "error"):
                    pytest.fail(f"Received error: {event}")

            # Verify transcription contains expected content
            assert event["type"] == COMPLETED_EVENT
            assert event["item_id"] == committed_item_id
            assert event["transcript"] == full_text
            assert event["usage"]["type"] == "tokens"
            assert full_text == (
                " First words I spoke in the original phonograph."
                " A little piece of practical poetry. Mary had a little lamb,"
                " it sleeps with quite a flow, and everywhere that Mary went,"
                " the lamb was sure to go."
            ) or full_text == (
                " First words I spoke in the original phonograph."
                " A little piece of practical poetry. Mary had a little lamb,"
                " it squeaked with quite a flow, and everywhere that Mary went,"
                " the lamb was sure to go."
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_empty_commit_does_not_crash_engine(
    model_name, mary_had_lamb_audio_chunks, rocm_aiter_fa_attention
):
    """Test that committing without audio does not crash the engine.

    Regression test for https://github.com/vllm-project/vllm/issues/34532.
    An empty commit (no prior input_audio_buffer.append) used to trigger
    ``AssertionError: For realtime you must provide a multimodal_embedding
    at every step`` which killed the entire engine process, disconnecting
    every connected client. It is now rejected before reaching the engine.
    """
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)

        # --- First connection: empty commit (no audio appended) ----------
        async with websockets.connect(ws_url) as ws:
            await configure_session(ws, model_name)

            # Commit without sending any audio
            await send_event(ws, {"type": "input_audio_buffer.commit"})

            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "error"
            assert event["error"]["code"] == "invalid_state"

        # --- Second connection: normal transcription ---------------------
        # Verifies the engine is still alive after the empty commit above.
        async with websockets.connect(ws_url) as ws:
            await configure_session(ws, model_name)

            for chunk in mary_had_lamb_audio_chunks:
                await send_event(
                    ws, {"type": "input_audio_buffer.append", "audio": chunk}
                )

            await send_event(ws, {"type": "input_audio_buffer.commit"})

            done_received = False
            while not done_received:
                # (ROCm) Generous timeout for first-request JIT compilation
                event = await receive_event(ws, timeout=600.0)
                if event["type"] == COMPLETED_EVENT:
                    done_received = True
                elif event["type"] in (FAILED_EVENT, "error"):
                    pytest.fail(f"Engine error after empty commit: {event}")
            assert done_received


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_session_update_invalid_model_returns_error(
    model_name, rocm_aiter_fa_attention
):
    """Test that session.update with an invalid model returns an error."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send session.update with a model that doesn't exist
            await send_event(ws, _session_update("nonexistent-model"))

            event = await receive_event(ws, timeout=10.0)
            assert event["type"] == "error"
            assert event["error"]["code"] == "model_not_found"
            assert "nonexistent-model" in event["error"]["message"]
            assert event["error"]["param"] == "session.audio.input.transcription.model"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_commit_without_session_update_returns_error(
    model_name, rocm_aiter_fa_attention
):
    """Test that committing before configuring the session returns an
    error and does not fall through to processing."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            # Send commit without sending session.update first
            await send_event(ws, {"type": "input_audio_buffer.commit"})

            event = await receive_event(ws, timeout=10.0)
            assert event["type"] == "error"
            assert event["error"]["code"] == "invalid_state"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_turn_detection_rejected(model_name, rocm_aiter_fa_attention):
    """Test that requesting server-side VAD returns a structured error."""
    server_args = ["--enforce-eager", "--max-model-len", "2048"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name, server_args, env_dict=REALTIME_ENV_OVERRIDES
    ) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            update = _session_update(model_name)
            update["session"]["audio"]["input"]["turn_detection"] = {
                "type": "server_vad"
            }
            await send_event(ws, update)

            event = await receive_event(ws, timeout=10.0)
            assert event["type"] == "error"
            assert event["error"]["code"] == "not_supported"
            assert event["error"]["param"] == "session.audio.input.turn_detection"
