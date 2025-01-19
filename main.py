import json
import os
import uuid
from typing import List
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

load_dotenv()

from db import get_latest_workout

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from workout_response import WorkoutUnprocessedResponse, put_workout_in_db, convert_workout_to_prompt_format, \
    convert_unprocessed_to_processed, WorkoutResponse

from vlm_request import get_vlm_feedback, encode_image_to_base64, nebius_client, json_llm_request

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    user_id: uuid.UUID
    age: int
    weight: int  # in kg
    difficulty: str  # easy, medium, hard
    muscle_groups: List[str]
    length: int  # in minutes
    comment: str  # user's comments for the workout creation "I'm feeling strong today"


@app.post("/make-workout")
async def make_workout(workout_form: WorkoutForm) -> WorkoutResponse:
    workout_form.user_id = uuid.UUID("a879f12d-8834-4987-859a-0d53e72d76a3")

    last_workout = None
    try:
        last_workout = convert_workout_to_prompt_format(await get_latest_workout(workout_form.user_id))
    except Exception as e:
        print(f"Error fetching last workout: {e}. Continuing without it.")

    def get_exercises(workout_form: WorkoutForm) -> WorkoutUnprocessedResponse:
        prompt = 'You are a personal trainer creating a workout plan for a client.'
        prompt += f'\nYour client is a {workout_form.age}-year-old and weighs {workout_form.weight} kg.'
        prompt += (f'\nCreate an approximately {workout_form.length}-minute '
                   f'(which is about {int(workout_form.length / 3)} total sets) '
                   f'{workout_form.difficulty} workout that targets the following muscle groups:')
        for group in workout_form.muscle_groups:
            prompt += f'\n- {group}'

        prompt += f"\n\nClient's comment: {workout_form.comment}"
        prompt += '\nUse the following exercises:\n'
        with open('data/exercise_summary.txt', 'r') as f:
            prompt += ''.join(f.readlines())
        prompt += '\nOutput the exercises in the provided json format. Write set as "weight x reps". Vary the reps and weight between sets.'
        prompt += '\nUse 0 weight for bodyweight exercises. Use up to 5 exercises in total'
        prompt += '\nIn the description, include a short (2 sentences max) explanation of why you created the workout like this.'
        if last_workout:
            try:
                prompt += '\nHere is the last workout of this user:\n'
                prompt += json.dumps(last_workout.dict(), indent=4)
            except Exception as e:
                print(f"Error adding last workout to prompt: {e}")
        return json_llm_request(prompt, WorkoutUnprocessedResponse)

    workout = get_exercises(workout_form)
    workout = convert_unprocessed_to_processed(workout)
    # await put_workout_in_db(workout, workout_form.user_id)
    return workout

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
