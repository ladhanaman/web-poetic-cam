# import os
# import typing_extensions as typing
# import google.generativeai as genai
# from dotenv import load_dotenv
# from PIL import Image
# import io

# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY)

# class ImageAnalysis(typing.TypedDict):
#     mood: str
#     themes: list[str]
#     concrete_nouns: list[str]

# def analyze_image(image_path: str) -> str:
#     """
#     Takes a local image path, resizes it, and analyzes it using Gemini Vision.
#     Optimized for low latency (Inline transfer vs Upload API).
#     """
    
#     if not os.path.exists(image_path):
#         print(f"Image not found at: {image_path}")
#         return ""

#     print(f"Analyzing image: {image_path}...")
    
#     try:
#         # 1. OPTIMIZATION: Load and Resize Image locally
#         # We don't need 12MP for semantic analysis. 800px is plenty.
#         with Image.open(image_path) as img:
#             # Resize logic: constrain max dimension to 800px
#             img.thumbnail((800, 800))
#             # Keep img in memory for the API call
            
            
#             model = genai.GenerativeModel('gemini-2.5-flash')

#             # 3. The Prompt
#             prompt = """
#             Analyze this image through the lens of a 19th-century poet. 
#             Identify the following three elements:
#             1. Mood: A single adjective (e.g., Melancholic, Serene, Chaotic).
#             2. Themes: 2-3 abstract concepts (Eternity, Solitude, Death, Nature).
#             Use archaic, rhythmic phrasing.
#             3. Concrete Nouns: 3-5 physical objects visible in the scene.

#             Return the result as JSON using this schema:
#             {
#                 "mood": str,
#                 "themes": list[str],
#                 "concrete_nouns": list[str]
#             }
#             """

#             # 4. Generate Content (Direct/Inline)
#             # Passing the 'img' object directly avoids the slow 'upload_file' handshake.
#             response = model.generate_content(
#                 [prompt, img],
#                 generation_config={"response_mime_type": "application/json"}
#             )
            
#             # 5. Parse JSON
#             import json
#             analysis: ImageAnalysis = json.loads(response.text)
            
#             # 6. Construct Semantic String
#             noun_str = ", ".join(analysis.get('concrete_nouns', []))
#             theme_str = ", ".join(analysis.get('themes', []))
#             mood_str = analysis.get('mood', 'Unknown')
            
#             narrative = f"A {mood_str} poem about {theme_str}, featuring imagery of {noun_str}."
            
#             print(f"Analysis Complete.")
#             print(f"Generated Query: '{narrative}'")
            
#             return narrative

#     except Exception as e:
#         print(f"Vision Bridge Failed: {e}")
#         return f"ERROR: {str(e)}"


# if __name__ == "__main__":

#     #testing
#     TEST_IMAGE_PATH = "test.jpg"
#     if os.path.exists(TEST_IMAGE_PATH):
#         query_string = analyze_image(TEST_IMAGE_PATH)
#     else:
#          print("Please place a .jpg file named 'test.jpg' in this directory.")
import os
import json
import base64
import io
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv()

# --- CONFIGURATION ---
# Engineering Tip: Define model IDs as constants at the top or in .env
# This makes future updates a 1-line change.
VISION_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct" 

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_image(image_file) -> str:
    """
    Takes a Streamlit UploadedFile (or file-like object), 
    processes it, and analyzes it using Groq Vision.
    """
    
    if image_file is None:
        return ""

    print(f"[SYSTEM] Analyzing image with model: {VISION_MODEL_ID}...")
    
    try:
        # 1. Load and Resize
        with Image.open(image_file) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to optimize token usage and latency
            img.thumbnail((800, 800))
            
            # 2. Convert to Base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 3. The Prompt
        system_prompt = "You are a poetic assistant. Return ONLY valid JSON."
        user_prompt = """
        Analyze this image for a 19th-century poem. Identify:
        1. Mood: Single adjective.
        2. Themes: 2-3 abstract concepts.
        3. Concrete Nouns: 3-5 physical objects.

        Return strictly this JSON:
        {
            "mood": "str",
            "themes": ["str", "str"],
            "concrete_nouns": ["str", "str"]
        }
        """

        # 4. API Call
        response = client.chat.completions.create(
            model=VISION_MODEL_ID, # <--- Updated Reference
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        
        # 5. Parse
        result = json.loads(response.choices[0].message.content)
        
        # 6. Construct String
        noun_str = ", ".join(result.get('concrete_nouns', []))
        theme_str = ", ".join(result.get('themes', []))
        mood_str = result.get('mood', 'Unknown')
        
        narrative = f"A {mood_str} poem about {theme_str}, featuring imagery of {noun_str}."
        
        print(f"[SUCCESS] Generated Query: '{narrative}'")
        return narrative

    except Exception as e:
        print(f"[ERROR] Vision Bridge Failed: {e}")
        # Fallback suggestion for the logs
        if "model_decommissioned" in str(e):
            print("CRITICAL: The model ID is invalid. Check Groq Console for the latest model.")
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    if os.path.exists("test.jpg"):
        with open("test.jpg", "rb") as f:
            print(analyze_image(f))
    else:
        print("No test.jpg found.")
