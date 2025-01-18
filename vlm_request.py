import base64
import time
import os
from openai import OpenAI

from prompt import get_prompt
from utils import get_exercise_stage


nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_vlm_feedback(reference_base64, attempt_base64) -> str:
    """
    Compare two exercise images using OpenAI's Vision model to check if the exercise
    is performed correctly.
    """
    stage = get_exercise_stage()

    # Construct the messages with both images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": get_prompt(stage)
                },
                # {
                #     "type": "image_url",
                #     "image_url": {
                #         "url": f"data:image/jpeg;base64,{reference_base64}",
                #         "detail": "high"
                #     }
                # },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{attempt_base64}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    try:
        print(get_prompt(stage))
        start = time.time()
        response = nebius_client.chat.completions.create(
            model="Qwen/Qwen2-VL-72B-Instruct",
            messages=messages,
            max_tokens=500,
            temperature=0.0,
        )
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {str(e)}"
