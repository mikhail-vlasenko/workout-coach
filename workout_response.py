import re
import uuid
from collections import defaultdict
from typing import List, Dict

from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime

from db import WorkoutSet, WorkoutSession, create_workout_session


class Set(BaseModel):
    weight: float
    reps: int

class UnprocessedExercise(BaseModel):
    name: str
    target_muscle: str
    sets: List[str]  # set is "weight x reps"

class Exercise(BaseModel):
    name: str
    target_muscle: str
    sets: List[Set]

class WorkoutUnprocessedResponse(BaseModel):
    title: str
    exercises: List[UnprocessedExercise]
    description: str

class WorkoutResponse(BaseModel):
    title: str
    exercises: List[Exercise]
    description: str


def convert_unprocessed_to_processed(unprocessed: WorkoutUnprocessedResponse) -> WorkoutResponse:
    processed_exercises = []

    for exercise in unprocessed.exercises:
        processed_sets = []
        for set_str in exercise.sets:
            # Split the set string and clean up any whitespace
            try:
                weight_str, reps_str = set_str.split('x')
                weight_str = re.sub(r'[^\d.]', '', weight_str.strip())
                reps_str = re.sub(r'[^\d.]', '', reps_str.strip())

                processed_sets.append(Set(
                    weight=float(weight_str),
                    reps=int(reps_str)
                ))
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid set format for exercise {exercise.name}: {set_str}") from e

        processed_exercises.append(Exercise(
            name=exercise.name,
            target_muscle=exercise.target_muscle,
            sets=processed_sets
        ))

    return WorkoutResponse(
        title=unprocessed.title,
        exercises=processed_exercises,
        description=unprocessed.description
    )


async def put_workout_in_db(workout_response: WorkoutResponse, user_id: uuid.UUID):
    # Convert the workout response to WorkoutSession format
    sets = []
    set_number = 1

    for exercise in workout_response.exercises:
        for set_data in exercise.sets:
            workout_set = WorkoutSet(
                exercise_name=exercise.name,
                set_number=set_number,
                weight=set_data.weight,
                reps=set_data.reps,
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


def convert_workout_to_prompt_format(workout_data: Dict) -> WorkoutUnprocessedResponse:
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
        exercise = UnprocessedExercise(
            name=exercise_name,
            target_muscle="unknown",  # This should be fetched from exercise database
            sets=sets_list
        )
        exercises.append(exercise)

    # Create the WorkoutResponse
    workout_response = WorkoutUnprocessedResponse(
        title=session['title'],
        exercises=exercises,
        description=session.get('description', '')
    )

    return workout_response
