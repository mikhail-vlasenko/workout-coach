import base64
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

from pose_estimation.mediapipe_3d_frame import post_estimation_3d_from_frame

load_dotenv()

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

app = FastAPI()
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)
pose_api = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
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

@app.post("/process-image")
async def process_image(base64_image: bytes):
    import cv2
    import numpy as np
    """
    Accepts a base64 encoded image in the request payload, decodes it,
    and re-encodes it before returning it as a base64 encoded string.
    """
    try:
        if not base64_image:
            return {"error": "Base64 encoded image is required."}

        # Decode the base64 image
        decoded_image = base64.b64decode(base64_image)

        # Convert to NumPy array
        np_array = np.frombuffer(decoded_image, np.uint8)

        # Convert to OpenCV image format
        cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Process cv_image if needed (e.g., transformations, analysis)
        bgr_frame_viz = post_estimation_3d_from_frame(
            cv_image, pose_api, True, 1.0
        )

        # Re-encode the OpenCV image back to Base64
        _, buffer = cv2.imencode('.jpg', bgr_frame_viz)
        re_encoded_image = base64.b64encode(buffer).decode("utf-8")

        return re_encoded_image
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.get("/")
async def root():
    reference_image_path = "data/correct.png"
    attempt_image_path = "data/wrong.jpg"

    print("Comparing exercise images...")
    result = compare_exercise_images(reference_image_path, attempt_image_path)
    print("\nAnalysis Result:")
    print(result)
