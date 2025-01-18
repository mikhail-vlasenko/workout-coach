import requests

fronted_url = "http://localhost:3000"


def get_exercise_stage():
    return "preparation"


def request_audio_id(id):
    requests.post(f"{fronted_url}/play_audio", json={"id": id})

