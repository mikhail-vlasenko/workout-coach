import uuid
from typing import List

from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime

from db import WorkoutSet, WorkoutSession, create_workout_session


class Set(BaseModel):
    weight: int
    reps: int

class Exercise(BaseModel):
    name: str
    target_muscle: str
    sets: List[Set]

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
            workout_set = WorkoutSet(
                exercise_name=exercise.name,
                set_number=set_number,
                weight=float(set_data.weight),  # Convert to float as required by WorkoutSet
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
