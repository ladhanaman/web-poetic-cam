import os
import typing_extensions as typing
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image  # NEW: For resizing
import io

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class ImageAnalysis(typing.TypedDict):
    mood: str
    themes: list[str]
    concrete_nouns: list[str]

def analyze_image(image_path: str) -> str:
    """
    Takes a local image path, resizes it, and analyzes it using Gemini Vision.
    Optimized for low latency (Inline transfer vs Upload API).
    """
    
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        return ""

    print(f"Analyzing image: {image_path}...")
    
    try:
        # 1. OPTIMIZATION: Load and Resize Image locally
        # We don't need 12MP for semantic analysis. 800px is plenty.
        with Image.open(image_path) as img:
            # Resize logic: constrain max dimension to 800px
            img.thumbnail((800, 800))
            # Keep img in memory for the API call
            
            
            model = genai.GenerativeModel('gemini-2.5-flash')

            # 3. The Prompt
            prompt = """
            Analyze this image through the lens of a 19th-century poet. 
            Identify the following three elements:
            Do not describe the image literally. Instead, describe the **soul** of the image.
            Use abstract concepts (Eternity, Solitude, Death, Nature).
            Use archaic, rhythmic phrasing.
            1. Mood: A single adjective (e.g., Melancholic, Serene, Chaotic).
            2. Themes: 2-3 abstract concepts (Eternity, Solitude, Death, Nature).
            Use archaic, rhythmic phrasing.
            3. Concrete Nouns: 3-5 physical objects visible in the scene.

            Return the result as JSON using this schema:
            {
                "mood": str,
                "themes": list[str],
                "concrete_nouns": list[str]
            }
            """

            # 4. Generate Content (Direct/Inline)
            # Passing the 'img' object directly avoids the slow 'upload_file' handshake.
            response = model.generate_content(
                [prompt, img],
                generation_config={"response_mime_type": "application/json"}
            )
            
            # 5. Parse JSON
            import json
            analysis: ImageAnalysis = json.loads(response.text)
            
            # 6. Construct Semantic String
            noun_str = ", ".join(analysis.get('concrete_nouns', []))
            theme_str = ", ".join(analysis.get('themes', []))
            mood_str = analysis.get('mood', 'Unknown')
            
            narrative = f"A {mood_str} poem about {theme_str}, featuring imagery of {noun_str}."
            
            print(f"Analysis Complete.")
            print(f"➡️ Generated Query: '{narrative}'")
            
            return narrative

    except Exception as e:
        print(f"Vision Bridge Failed: {e}")
        return ""


if __name__ == "__main__":

    #testing
    TEST_IMAGE_PATH = "test.jpg"
    if os.path.exists(TEST_IMAGE_PATH):
        query_string = analyze_image(TEST_IMAGE_PATH)
    else:
        print("Please place a .jpg file named 'test.jpg' in this directory.")