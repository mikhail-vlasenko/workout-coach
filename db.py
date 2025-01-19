import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from supabase import create_client
import os

supabase = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)


class WorkoutSet(BaseModel):
    exercise_name: str
    set_number: int
    weight: float
    reps: int
    is_personal_record: Optional[bool] = False
    notes: Optional[str] = None
    completed_at: Optional[datetime] = None


class WorkoutSession(BaseModel):
    user_id: uuid.UUID
    title: str
    description: Optional[str] = None
    sets: List[WorkoutSet]


async def get_latest_workout(user_id: uuid.UUID):
    # Get the most recent workout session
    session_response = supabase.table("workout_sessions") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("started_at", desc=True) \
        .limit(1) \
        .execute()

    if not session_response.data:
        raise HTTPException(status_code=404, detail="No workouts found for this user")

    session = session_response.data[0]

    # Get all sets for this session with exercise information
    sets_response = supabase.table("workout_sets") \
        .select("*, exercises(*)") \
        .eq("session_id", session['id']) \
        .order("set_number") \
        .execute()

    return {
        "session": session,
        "sets": sets_response.data
    }


async def create_workout_session(workout: WorkoutSession):
    # Create the workout session first
    session_data = {
        "user_id": str(workout.user_id),
        "template_id": None,
        "title": workout.title,
        "started_at": datetime.utcnow().isoformat(),
        "description": workout.description
    }

    session_response = supabase.table("workout_sessions").insert(session_data).execute()
    session_id = session_response.data[0]['id']

    # Create all sets for this session
    sets_data = [
        {
            "session_id": session_id,
            "exercise_name": set.exercise_name,
            "set_number": set.set_number,
            "weight": set.weight,
            "reps": set.reps,
            "is_personal_record": set.is_personal_record,
            "notes": set.notes,
            "completed_at": set.completed_at.isoformat() if set.completed_at else datetime.utcnow().isoformat()
        }
        for set in workout.sets
    ]

    sets_response = supabase.table("workout_sets").insert(sets_data).execute()

    return {
        "session": session_response.data[0],
        "sets": sets_response.data
    }
