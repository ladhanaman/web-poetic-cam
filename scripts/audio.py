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
        # We keep output_dir in init for compatibility, but won't use it for file saving
        self.voice = "en-GB-SoniaNeural" 

    async def _generate_audio_async(self, text: str) -> bytes:
        """Internal async handler to fetch audio bytes directly."""
        communicate = edge_tts.Communicate(text, self.voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    def synthesize(self, text: str, filename: str = None) -> Optional[bytes]:
        """
        Synthesizes text to speech and returns raw bytes.
        Args:
            text: The poem to read.
            filename: Ignored in cloud version (kept for interface compatibility).
        Returns:
            bytes: The raw audio data.
        """
        try:
            # Engineering Bridge: Running Async code in Sync environment
            return asyncio.run(self._generate_audio_async(text))
        except Exception as e:
            print(f"Error in Audio Synthesis: {e}")
            return None
