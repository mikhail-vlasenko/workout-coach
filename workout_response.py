import re
import uuid
from collections import defaultdict
from typing import List, Dict

from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime

from db import WorkoutSet, WorkoutSession, create_workout_session


class Exercise(BaseModel):
    name: str
    target_muscle: str
    sets: List[str]  # set is "weight x reps"

class WorkoutResponse(BaseModel):
    title: str
    exercises: List[Exercise]
    description: str


async def put_workout_in_db(workout_response: WorkoutResponse, user_id: uuid.UUID):
    # Convert the workout response to WorkoutSession format
    sets = []
    set_number = 1

    for exercise in workout_response.exercises:
        for set_data in exercise.sets:
            weight, reps = set_data.split('x')

            def remove_non_digits(string):
                # keep only digits and '.' for floats
                return re.sub(r'[^\d.]', '', string)
            weight = remove_non_digits(weight)
            reps = remove_non_digits(reps)
            workout_set = WorkoutSet(
                exercise_name=exercise.name,
                set_number=set_number,
                weight=float(weight),  # Convert to float as required by WorkoutSet
                reps=int(reps),
                is_personal_record=False,  # Default value
                completed_at=datetime.utcnow()
            )
            sets.append(workout_set)
            set_number += 1

    # Create WorkoutSession object
    workout_session = WorkoutSession(
        user_id=str(user_id),
        title=workout_response.title,
        description=workout_response.description,
        sets=sets
    )

    result = await create_workout_session(workout_session)
    return result


def convert_workout_to_prompt_format(workout_data: Dict) -> WorkoutResponse:
    session = workout_data['session']
    sets = workout_data['sets']

    # Group sets by exercise
    exercise_sets = defaultdict(list)
    for set_data in sets:
        exercise_name = set_data['exercise_name']
        # Format the set as "weight x reps"
        set_str = f"{set_data['weight']} x {set_data['reps']}"
        exercise_sets[exercise_name].append(set_str)

    # Create Exercise objects for each unique exercise
    exercises = []
    for exercise_name, sets_list in exercise_sets.items():
        # Note: In a real system, you'd want to look up the target muscle
        # from your exercise database. For now, we'll use a placeholder
        exercise = Exercise(
            name=exercise_name,
            target_muscle="unknown",  # This should be fetched from exercise database
            sets=sets_list
        )
        exercises.append(exercise)

    # Create the WorkoutResponse
    workout_response = WorkoutResponse(
        title=session['title'],
        exercises=exercises,
        description=session.get('description', '')
    )

    return workout_response
