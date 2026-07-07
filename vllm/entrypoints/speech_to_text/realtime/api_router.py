# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, WebSocket

from vllm.logger import init_logger

from .connection import RealtimeConnection

logger = init_logger(__name__)


router = APIRouter()


@router.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket):
    """WebSocket endpoint for realtime audio transcription.

    Implements the transcription subset of the OpenAI Realtime API:
    1. Client connects to ws://host/v1/realtime
    2. Server sends session.created
    3. Client sends session.update (audio format, transcription model);
       server acks with session.updated
    4. Client streams input_audio_buffer.append events with base64 PCM16
       chunks; server emits
       conversation.item.input_audio_transcription.delta events as text
       is generated
    5. Client sends input_audio_buffer.commit to end the utterance;
       server emits input_audio_buffer.committed,
       conversation.item.created, then
       conversation.item.input_audio_transcription.completed with the
       final transcript + usage
    6. Repeat from step 4 for the next utterance;
       input_audio_buffer.clear abandons the current utterance

    Audio format: PCM16 mono, base64-encoded; 24 kHz by default,
    configurable via session.audio.input.format.rate (16/24/48 kHz).
    """
    app = websocket.app
    serving = app.state.openai_serving_realtime

    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()
