import asyncio
import edge_tts
import os
from typing import Optional

class AudioEngine:
    """
    Handles Text-to-Speech synthesis using Edge TTS (Free, High Quality).
    Wraps async methods for synchronous usage in Streamlit.
    """

    def __init__(self, output_dir: str = "assets/audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        
        self.voice = "en-GB-SoniaNeural" 

    async def _generate_audio_async(self, text: str, output_file: str) -> None:
        """Internal async handler for communicating with Edge API."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

    def synthesize(self, text: str, filename: str = "poem_output.mp3") -> Optional[str]:
        """
        Synthesizes text to speech and saves to disk.
        
        Args:
            text: The poem text to read.
            filename: The name of the file to save.
            
        Returns:
            str: Path to the generated audio file.
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # Engineering Bridge: Running Async code in Sync environment
            # This creates a new event loop to run the coroutine
            asyncio.run(self._generate_audio_async(text, output_path))
            return output_path
            
        except Exception as e:
            print(f"Error in Audio Synthesis: {e}")
            return None