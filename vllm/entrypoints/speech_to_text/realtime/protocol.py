# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire schema for the /v1/realtime WebSocket.

Implements the transcription subset of the OpenAI Realtime API using the
official ``openai.types.realtime`` models, subclassed only where the SDK
pins fields to OpenAI-hosted values (audio rate, model names).
"""

from typing import Literal

from openai.types.realtime import SessionCreatedEvent as _SessionCreatedEvent
from openai.types.realtime import SessionUpdatedEvent as _SessionUpdatedEvent
from openai.types.realtime import SessionUpdateEvent as _SessionUpdateEvent
from openai.types.realtime.realtime_audio_formats import AudioPCM as _AudioPCM
from openai.types.realtime.realtime_transcription_session_audio import (
    RealtimeTranscriptionSessionAudio as _TranscriptionSessionAudio,
)
from openai.types.realtime.realtime_transcription_session_audio_input import (
    RealtimeTranscriptionSessionAudioInput as _TranscriptionSessionAudioInput,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest as _TranscriptionSessionCreateRequest,
)

# Sample rate the realtime models consume; appended audio is resampled
# to this rate before it reaches the engine.
MODEL_SAMPLE_RATE = 16000

# Fallback when the client omits `audio.input.format.rate`. Matches the
# OpenAI Realtime default (the SDK pins `AudioPCM.rate` to Literal[24000]).
DEFAULT_INPUT_SAMPLE_RATE = 24000

# Wire rates accepted on `audio.input.format.rate`; anything else is
# rejected. Non-model rates are resampled server-side.
SUPPORTED_INPUT_SAMPLE_RATES = (16000, 24000, 48000)


class AudioPCM(_AudioPCM):
    """`audio/pcm` input format with the SDK's Literal[24000] rate pin
    relaxed so common ASR client rates can be validated server-side."""

    type: Literal["audio/pcm"] = "audio/pcm"
    rate: int | None = None


class TranscriptionSessionAudioInput(_TranscriptionSessionAudioInput):
    format: AudioPCM | None = None


class TranscriptionSessionAudio(_TranscriptionSessionAudio):
    input: TranscriptionSessionAudioInput | None = None


class TranscriptionSessionConfig(_TranscriptionSessionCreateRequest):
    audio: TranscriptionSessionAudio | None = None


class SessionUpdateEvent(_SessionUpdateEvent):
    session: TranscriptionSessionConfig


class SessionCreatedEvent(_SessionCreatedEvent):
    session: TranscriptionSessionConfig


class SessionUpdatedEvent(_SessionUpdatedEvent):
    session: TranscriptionSessionConfig
