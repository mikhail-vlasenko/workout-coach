import base64
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

load_dotenv()

app = FastAPI()
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def compare_exercise_images(reference_image_path: str, attempt_image_path: str) -> str:
    """
    Compare two exercise images using OpenAI's Vision model to check if the exercise
    is performed correctly.
    """
    # Encode both images
    reference_base64 = encode_image_to_base64(reference_image_path)
    attempt_base64 = encode_image_to_base64(attempt_image_path)

    # Construct the messages with both images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I'll show you two images. The first is a reference image of an exercise being performed correctly. The second is someone attempting the same exercise. Please analyze if the exercise in the second image is being performed correctly compared to the reference. Point out any differences in form, positioning, or technique that need improvement.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{reference_base64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{attempt_base64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2-VL-72B-Instruct", messages=messages, max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {str(e)}"


@app.get("/")
async def root():
    reference_image_path = "data/correct.png"
    attempt_image_path = "data/wrong.jpg"

    print("Comparing exercise images...")
    result = compare_exercise_images(reference_image_path, attempt_image_path)
    print("\nAnalysis Result:")
    print(result)
