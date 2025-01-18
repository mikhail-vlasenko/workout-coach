import json
import os
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from vlm_request import get_vlm_feedback, encode_image_to_base64, nebius_client


app = FastAPI()


@app.get("/")
async def root():
    reference_image_path = "data/correct.png"
    attempt_image_path = "data/bad/deadlift-hips_0.webp"

    print("Comparing exercise images...")
    attempt_base64 = encode_image_to_base64(attempt_image_path)
    result = get_vlm_feedback(None, attempt_base64=attempt_base64)
    print("\nAnalysis Result:")
    print(result)


class WorkoutForm(BaseModel):
    age: int
    weight: int  # in kg
    difficulty: str  # easy, medium, hard
    muscle_groups: List[str]
    length: int  # in minutes


class Set(BaseModel):
    weight: int
    reps: int

class Exercise(BaseModel):
    name: str
    target_muscle: str
    order: int
    sets: List[Set]

class WorkoutResponse(BaseModel):
    title: str
    description: str
    exercises: List[Exercise]


@app.post("/make-workout")
async def make_workout(workout_form: WorkoutForm):
    def get_exercises(workout_form: WorkoutForm):
        prompt = 'You are a personal trainer creating a workout plan for a client.'
        prompt += f'\nYour client is a {workout_form.age}-year-old and weighs {workout_form.weight} kg.'
        prompt += (f'\nCreate an approximately {workout_form.length}-minute '
                   f'{workout_form.difficulty} workout that targets the following muscle groups:')
        for group in workout_form.muscle_groups:
            prompt += f'\n- {group}'
        prompt += '\nUse the following exercises:\n'
        with open('data/exercise_summary.txt', 'r') as f:
            prompt += ''.join(f.readlines())
        prompt += '\nOutput the exercises in the provided json format. Use 0 weight for bodyweight exercises.'
        response = nebius_client.beta.chat.completions.parse(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.0,
            response_format=WorkoutResponse
        )
        return response.choices[0].message.parsed

    return get_exercises(workout_form)

@app.get("/one-rep-left")
async def play_one_rep_audio():
    """Endpoint to serve the 'one rep to go' audio file."""
    audio_path = "audio/you can do it.mp3"
    if not os.path.exists(audio_path):
        return {"error": "Audio file not found"}

    # Return the audio file with appropriate headers
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": "inline; filename=one_rep_left.mp3"
        }
    )
