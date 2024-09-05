import os
import base64
from time import perf_counter
import requests

OCTOAI_TOKEN = os.getenv("OCTOAI_API_KEY")
WHISPER_API_URL = os.getenv(
    "WHISPER_API_URL", "https://whisper2-or1pkb9b656p.octoai.run/predict"
)
OCTOAI_BASE_URL = os.getenv("OCTOAI_BASE_URL", "https://text.octoai.run/v1")

MIXTRAL8X7B_BASE_URL = os.getenv("MIXTRAL8X7B_BASE_URL", "https://text.octoai.run/v1")
HERMES2THETA_API_URL = os.getenv("HERMES2THETA_API_URL", "https://text.octoai.run/v1")
MISTRAL7B_BASE_URL = os.getenv("MISTRAL7B_BASE_URL", "https://text.octoai.run/v1")
WIZARDLM_BASE_URL = os.getenv("WIZARDLM_BASE_URL", "https://text.octoai.run/v1")

RUNNING_SNOWFLAKE_NATIVE = os.getenv("RUNNING_SNOWFLAKE_NATIVE", False)


def transcribe_audio(
    file_path: str, api_key: str, whisper_api_url: str = WHISPER_API_URL
) -> dict:
    """
    Takes the file path of an audio file and transcribes it to text.

    Returns a string with the transcribed text.
    """
    with open(file_path, "rb") as f:
        encoded_audio = str(base64.b64encode(f.read()), "utf-8")

        start = perf_counter()
        reply = requests.post(
            whisper_api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={"audio": encoded_audio},
            timeout=300,
        )
        elapsed_time = perf_counter() - start
        try:
            transcript = reply.json()["transcription"]
        except Exception as e:
            print(e)
            print(reply.text)
            raise ValueError("The transcription could not be completed.")

    return {"transcript": transcript, "elapsed_time": elapsed_time}
