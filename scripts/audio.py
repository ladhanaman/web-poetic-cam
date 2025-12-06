import asyncio
import edge_tts
import io
from typing import Optional

class AudioEngine:
    """
    Handles Text-to-Speech synthesis using Edge TTS.
    Returns in-memory bytes for cloud compatibility.
    """

    def __init__(self, output_dir: str = "assets/audio"):
        self.voice = "en-GB-SoniaNeural" 

    async def _generate_audio_async(self, text: str) -> bytes:
        """Internal async handler to fetch audio bytes directly."""
        print(f"[DEBUG] Starting TTS for text length: {len(text)}") # DEBUG LOG
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            print(f"[DEBUG] TTS Complete. Generated {len(audio_data)} bytes.") # DEBUG LOG
            return audio_data
        except Exception as e:
            print(f"[ERROR] EdgeTTS internal error: {e}")
            raise e

    def synthesize(self, text: str, filename: str = None) -> Optional[bytes]:
        try:
            # Run the async function
            return asyncio.run(self._generate_audio_async(text))
        except Exception as e:
            print(f"[ERROR] Audio Synthesis Wrapper failed: {e}")
            return None
