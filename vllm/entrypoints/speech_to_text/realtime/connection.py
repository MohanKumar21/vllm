# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WebSocket connection for OpenAI Realtime transcription sessions.

Pre-commit deltas reference the reserved current item id that the
subsequent ``input_audio_buffer.committed`` and
``conversation.item.created`` events announce. This deviates from
OpenAI's commit-only delta emission so vLLM's streaming realtime models
(which transcribe while audio is still arriving) keep their latency
advantage; sglang's realtime endpoint documents the same deviation.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

import numpy as np
import pybase64 as base64
from fastapi import WebSocket
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    RealtimeErrorEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (  # noqa: E501
    ConversationItemInputAudioTranscriptionCompletedEvent,
    UsageTranscriptTextUsageTokens,
)
from openai.types.realtime.conversation_item_input_audio_transcription_delta_event import (  # noqa: E501
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (  # noqa: E501
    ConversationItemInputAudioTranscriptionFailedEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (  # noqa: E501
    Error as TranscriptionFailedError,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content as InputAudioContent,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_error import RealtimeError
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.logger import init_logger
from vllm.multimodal.audio import resample_audio_scipy
from vllm.utils import random_uuid

from .protocol import (
    DEFAULT_INPUT_SAMPLE_RATE,
    MODEL_SAMPLE_RATE,
    SUPPORTED_INPUT_SAMPLE_RATES,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
    TranscriptionSessionAudioInput,
    TranscriptionSessionConfig,
)
from .serving import OpenAIServingRealtime

logger = init_logger(__name__)

# PCM16: 16-bit samples, 2 bytes each.
_SAMPLE_WIDTH = 2

_CLIENT_EVENT_TYPES: dict[str, type[BaseModel]] = {
    "session.update": SessionUpdateEvent,
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "input_audio_buffer.clear": InputAudioBufferClearEvent,
}


def _event_id() -> str:
    return f"event_{random_uuid()}"


class _ItemRun:
    """Generation state for one conversation item (utterance).

    The item id is reserved at construction and only announced to the
    client by ``input_audio_buffer.committed``; pre-commit deltas
    reference it so the client can correlate them after the commit.
    """

    def __init__(self):
        self.item_id = f"item_{random_uuid()}"
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.text = ""
        self.has_audio = False
        self.committed = False
        self.total_samples = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


class RealtimeConnection:
    """One realtime transcription session.

    Drives the WS receive loop, dispatches typed client events, and maps
    vLLM's streaming generation onto the OpenAI Realtime transcription
    event flow: audio appends feed a long-lived ``engine.generate()``
    request per item; a commit ends the item's audio and finalizes it.
    """

    def __init__(self, websocket: WebSocket, serving: OpenAIServingRealtime):
        self.websocket = websocket
        self.serving = serving
        self.session_id = f"sess_{random_uuid()}"

        self._is_connected = False
        self._configured = False
        self._input_sample_rate = DEFAULT_INPUT_SAMPLE_RATE
        self._client_model: str | None = None
        self._current_client_event_id: str | None = None
        self._max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

        self.run = _ItemRun()
        self.previous_item_id: str | None = None
        self._tasks: set[asyncio.Task] = set()

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket connection accepted: %s", self.session_id)
        self._is_connected = True

        await self._send(
            SessionCreatedEvent(
                event_id=_event_id(),
                type="session.created",
                session=self._build_session_info(),
            )
        )

        try:
            while True:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    self._is_connected = False
                    logger.debug("WebSocket disconnected: %s", self.session_id)
                    return

                text = message.get("text")
                if not text:
                    if message.get("bytes") is not None:
                        # OpenAI Realtime is base64 PCM in JSON text
                        # frames; binary frames aren't part of the spec.
                        await self._send_error(
                            "invalid_payload",
                            "Binary frames are not supported on /v1/realtime;"
                            " use input_audio_buffer.append with base64 audio.",
                        )
                    continue

                await self._handle_message(text)
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.session_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self.cleanup()

    async def _handle_message(self, text: str):
        self._current_client_event_id = None
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            await self._send_error("invalid_payload", "Invalid JSON")
            return
        if not isinstance(raw, dict):
            await self._send_error(
                "invalid_payload", "Top-level event must be a JSON object"
            )
            return

        event_id = raw.get("event_id")
        self._current_client_event_id = event_id if isinstance(event_id, str) else None

        event_cls = _CLIENT_EVENT_TYPES.get(raw.get("type"))
        if event_cls is None:
            await self._send_error(
                "unknown_event", f"Unknown event type: {raw.get('type')!r}"
            )
            return

        try:
            event = event_cls.model_validate(raw)
        except ValidationError as e:
            # Report first error only; matches OpenAI server behavior.
            err = e.errors()[0]
            loc = ".".join(str(x) for x in err["loc"])
            await self._send_error(
                "invalid_value",
                err.get("msg") or "Invalid payload",
                param=loc or None,
            )
            return

        try:
            await self._dispatch(event)
        except Exception as e:
            logger.exception("Error handling event: %s", e)
            await self._send_error(
                "processing_error",
                sanitize_message(str(e)),
                error_type="server_error",
            )

    async def _dispatch(self, event: BaseModel):
        if isinstance(event, InputAudioBufferAppendEvent):
            await self._on_append(event)
        elif isinstance(event, SessionUpdateEvent):
            await self._on_session_update(event)
        elif isinstance(event, InputAudioBufferCommitEvent):
            await self._on_commit()
        elif isinstance(event, InputAudioBufferClearEvent):
            await self._on_clear()

    async def _on_session_update(self, event: SessionUpdateEvent):
        cfg = event.session
        audio_input = (
            cfg.audio.input if cfg.audio else None
        ) or TranscriptionSessionAudioInput()
        transcription = audio_input.transcription

        # Validate everything first; mutate config only once the whole
        # update is accepted.
        if audio_input.turn_detection is not None:
            await self._send_error(
                "not_supported",
                "Server-side VAD is not implemented; set"
                " audio.input.turn_detection: null and commit explicitly.",
                param="session.audio.input.turn_detection",
            )
            return
        if audio_input.noise_reduction is not None:
            await self._send_error(
                "not_supported",
                "audio.input.noise_reduction is not supported; set to null.",
                param="session.audio.input.noise_reduction",
            )
            return
        if transcription is not None and transcription.prompt is not None:
            await self._send_error(
                "not_supported",
                "audio.input.transcription.prompt is not supported.",
                param="session.audio.input.transcription.prompt",
            )
            return
        if transcription is not None and transcription.language is not None:
            await self._send_error(
                "not_supported",
                "audio.input.transcription.language is not supported;"
                " realtime models detect the language automatically.",
                param="session.audio.input.transcription.language",
            )
            return
        if (
            transcription is not None
            and transcription.model is not None
            and not self.serving._is_model_supported(transcription.model)
        ):
            await self._send_error(
                "model_not_found",
                f"The model `{transcription.model}` does not exist.",
                param="session.audio.input.transcription.model",
            )
            return

        new_rate = self._input_sample_rate
        fmt = audio_input.format
        if fmt is not None:
            if fmt.rate is not None and fmt.rate not in SUPPORTED_INPUT_SAMPLE_RATES:
                await self._send_error(
                    "invalid_value",
                    f"audio.input.format.rate must be one of"
                    f" {SUPPORTED_INPUT_SAMPLE_RATES}, got {fmt.rate}",
                    param="session.audio.input.format.rate",
                )
                return
            new_rate = fmt.rate or DEFAULT_INPUT_SAMPLE_RATE
            # Changing the rate mid-item would mix audio at two rates in
            # one utterance; require a commit or clear first.
            if new_rate != self._input_sample_rate and self.run.has_audio:
                await self._send_error(
                    "invalid_state",
                    "Cannot change audio.input.format.rate while audio is"
                    " buffered; commit or clear the current item first.",
                    param="session.audio.input.format.rate",
                )
                return

        self._input_sample_rate = new_rate
        if transcription is not None and transcription.model is not None:
            self._client_model = transcription.model
        self._configured = True

        await self._send(
            SessionUpdatedEvent(
                event_id=_event_id(),
                type="session.updated",
                session=self._build_session_info(),
            )
        )

    async def _on_append(self, event: InputAudioBufferAppendEvent):
        if not self._configured:
            await self._send_error(
                "invalid_state", "Send session.update before audio frames"
            )
            return
        # Empty audio is a no-op (heartbeat frames).
        if not event.audio:
            return

        try:
            audio_bytes = base64.b64decode(event.audio, validate=True)
        except (ValueError, TypeError):
            await self._send_error(
                "invalid_audio",
                "audio field is not valid base64",
                param="audio",
            )
            return
        if len(audio_bytes) % _SAMPLE_WIDTH != 0:
            await self._send_error(
                "invalid_audio_format",
                f"PCM16 frame length must be a multiple of {_SAMPLE_WIDTH} bytes",
            )
            return

        audio_array = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        if len(audio_array) == 0:
            return

        self.run.total_samples += len(audio_array)
        if self.run.total_samples / 1024**2 > self._max_audio_filesize_mb:
            await self._send_error(
                "invalid_audio",
                "Maximum audio size per item exceeded; commit or clear.",
            )
            return

        if self._input_sample_rate != MODEL_SAMPLE_RATE:
            # ponytail: per-chunk polyphase resample; a stateful streaming
            # resampler would avoid chunk-edge artifacts if quality matters.
            audio_array = resample_audio_scipy(
                audio_array,
                orig_sr=self._input_sample_rate,
                target_sr=MODEL_SAMPLE_RATE,
            ).astype(np.float32)

        if self.run.task is None:
            self._start_generation(self.run)
        self.run.audio_queue.put_nowait(audio_array)
        self.run.has_audio = True

    async def _on_commit(self):
        if not self._configured:
            await self._send_error("invalid_state", "Send session.update before commit")
            return
        run = self.run
        if not run.has_audio:
            await self._send_error(
                "invalid_state", "Cannot commit an empty audio buffer"
            )
            return

        run.committed = True
        await self._send(
            InputAudioBufferCommittedEvent(
                event_id=_event_id(),
                type="input_audio_buffer.committed",
                item_id=run.item_id,
                previous_item_id=self.previous_item_id,
            )
        )
        await self._send(
            ConversationItemCreatedEvent(
                event_id=_event_id(),
                type="conversation.item.created",
                previous_item_id=self.previous_item_id,
                item=RealtimeConversationItemUserMessage(
                    id=run.item_id,
                    type="message",
                    role="user",
                    status="completed",
                    content=[
                        InputAudioContent(type="input_audio", transcript=run.text)
                    ],
                ),
            )
        )

        # End of this item's audio; the generation task drains the queue
        # and emits transcription.completed for it.
        run.audio_queue.put_nowait(None)
        self.previous_item_id = run.item_id
        self.run = _ItemRun()

    async def _on_clear(self):
        # The generation may already have consumed queued audio, so the
        # only faithful "clear" is to cancel it without emitting item
        # events. A fresh item id is reserved so post-clear deltas don't
        # share an id with deltas from the abandoned audio;
        # previous_item_id is untouched (the item was never committed).
        if self.run.task is not None:
            self.run.task.cancel()
        self.run = _ItemRun()
        await self._send(
            InputAudioBufferClearedEvent(
                event_id=_event_id(), type="input_audio_buffer.cleared"
            )
        )

    def _start_generation(self, run: _ItemRun):
        """Start the streaming generation task for one item."""

        async def audio_stream() -> AsyncGenerator[np.ndarray, None]:
            while True:
                chunk = await run.audio_queue.get()
                if chunk is None:
                    break
                yield chunk

        input_stream = asyncio.Queue[list[int]]()
        streaming_input_gen = self.serving.transcribe_realtime(
            audio_stream(), input_stream
        )

        run.task = asyncio.create_task(
            self._run_generation(run, streaming_input_gen, input_stream)
        )
        self._tasks.add(run.task)
        run.task.add_done_callback(self._tasks.discard)

    async def _run_generation(
        self,
        run: _ItemRun,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        """Stream one item's transcription back to the client.

        Deltas are emitted as text is generated; when the item's audio
        stream ends (commit), the final transcript and usage are sent as
        conversation.item.input_audio_transcription.completed.
        """
        request_id = f"rt-{self.session_id}-{run.item_id}"

        try:
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=self.serving.model_cls.realtime_max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    if not run.prompt_tokens and output.prompt_token_ids:
                        run.prompt_tokens = len(output.prompt_token_ids)

                    delta = output.outputs[0].text
                    run.text += delta

                    # Feed output back as context for the next window.
                    input_stream.put_nowait(list(output.outputs[0].token_ids))
                    run.completion_tokens += len(output.outputs[0].token_ids)

                    if delta:
                        await self._send(
                            ConversationItemInputAudioTranscriptionDeltaEvent(
                                event_id=_event_id(),
                                type="conversation.item"
                                ".input_audio_transcription.delta",
                                item_id=run.item_id,
                                content_index=0,
                                delta=delta,
                            )
                        )

                if not self._is_connected:
                    # finish because websocket connection was killed
                    break

            if not self._is_connected:
                return

            await self._send(
                ConversationItemInputAudioTranscriptionCompletedEvent(
                    event_id=_event_id(),
                    type="conversation.item.input_audio_transcription.completed",
                    item_id=run.item_id,
                    content_index=0,
                    transcript=run.text,
                    usage=UsageTranscriptTextUsageTokens(
                        type="tokens",
                        input_tokens=run.prompt_tokens,
                        output_tokens=run.completion_tokens,
                        total_tokens=run.prompt_tokens + run.completion_tokens,
                    ),
                )
            )

            # Engine can finalize before a commit (e.g. per-window token
            # limit). Roll to a fresh item so later appends start clean.
            if self.run is run:
                self.run = _ItemRun()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            if not self._is_connected:
                return
            if run.committed:
                # committed + created were already emitted, so the item
                # exists client-side and transcription.failed can
                # reference it.
                await self._send(
                    ConversationItemInputAudioTranscriptionFailedEvent(
                        event_id=_event_id(),
                        type="conversation.item.input_audio_transcription.failed",
                        item_id=run.item_id,
                        content_index=0,
                        error=TranscriptionFailedError(
                            type="server_error",
                            code="inference_failed",
                            message=sanitize_message(str(e)),
                        ),
                    )
                )
            else:
                await self._send_error(
                    "inference_failed",
                    sanitize_message(str(e)),
                    error_type="server_error",
                )
                if self.run is run:
                    self.run = _ItemRun()

    def _build_session_info(self) -> TranscriptionSessionConfig:
        # id / object aren't SDK fields; round-trip via extra='allow' so
        # dumps emit them like the real server.
        return TranscriptionSessionConfig.model_validate(
            {
                "type": "transcription",
                "id": self.session_id,
                "object": "realtime.transcription_session",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self._input_sample_rate,
                        },
                        "transcription": {"model": self._client_model},
                        "noise_reduction": None,
                        "turn_detection": None,
                    }
                },
            }
        )

    async def _send(self, event: BaseModel):
        """Send event to client."""
        await self.websocket.send_text(event.model_dump_json())

    async def _send_error(
        self,
        code: str,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
    ):
        """Send a structured error event to the client."""
        envelope = RealtimeErrorEvent(
            event_id=_event_id(),
            type="error",
            error=RealtimeError(
                type=error_type,
                code=code,
                message=message,
                param=param,
                event_id=self._current_client_event_id,
            ),
        )
        await self.websocket.send_text(envelope.model_dump_json())

    async def cleanup(self):
        """Cleanup resources."""
        # Unblock any generation waiting on audio, then cancel.
        self.run.audio_queue.put_nowait(None)
        for task in list(self._tasks):
            if not task.done():
                task.cancel()

        logger.debug("Connection cleanup complete: %s", self.session_id)
