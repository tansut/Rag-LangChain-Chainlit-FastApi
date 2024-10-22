import os
from io import BytesIO
import base64
import httpx
import os
from io import BytesIO
from typing import IO
from openai import AsyncOpenAI

from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")

async def speech_to_text(audio_file) -> str:
    
    if (os.getenv("OPENAI_API_KEY")):
        openai_client = AsyncOpenAI()
    
        response = await openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )

        return response.text
    else:
        return "Please set OPENAI_API_KEY for speech_to_text"


def text_to_speech(text: str, mime_type: str, voice_id: str = ELEVENLABS_VOICE_ID) -> IO[bytes]:
    client = ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
    )
    response = client.text_to_speech.convert(
        voice_id=voice_id or ELEVENLABS_VOICE_ID,  
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=True,
        )
    )

    audio_stream = BytesIO()
    audio_stream.name = f"output_audio.{mime_type.split('/')[1]}"

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    # Return the stream for further use
    return audio_stream.name, audio_stream.read()
