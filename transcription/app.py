import os
import streamlit as st

from utils import (
    WHISPER_API_URL,
    RUNNING_SNOWFLAKE_NATIVE,
    transcribe_audio,
)
from audiorecorder import audiorecorder
from datetime import datetime
from time import perf_counter


def render_transcription(transcription):
    st.write(transcription[:500] + "...")

    with st.expander("See full transcription"):
        with st.container():
            st.write(transcription)


def do_transcription(audio_file_path, base_url):
    with st.status("Transcribing the audio to text..."):
        try:
            results = transcribe_audio(
                audio_file_path,
                api_key=st.session_state.octoai_api_key,
                whisper_api_url=base_url,
            )
        except Exception as e:
            print(e)
            results = None

    if results is None:
        st.error("Transcription failed.")
    else:
        st.session_state["transcript"] = results["transcript"]
        st.session_state["transcript_time"] = results["elapsed_time"]


def do_audio_recording():
    st.subheader("Click button to record audio for transcription:")
    audio = audiorecorder("Click to record", "Click to stop recording")
    if len(audio) > 0:
        st.audio(audio.export().read())
        # To get audio properties, use pydub AudioSegment properties:
        st.caption(
            f"Frame rate: {audio.frame_rate}, Duration: {audio.duration_seconds} seconds"
        )
        if st.button("Transcribe Recording"):
            # To save audio to a file, use pydub export method:
            audio.export("audio", format="wav")
            st.session_state["new_audio"] = True


def do_file_upload():
    with st.form(key="input_file_form"):
        file_uploader_result = st.file_uploader(
            "Upload audio file to be transcribed:",
            type=["mp3", "mp4", "m4a", "wav"],
            accept_multiple_files=False,
            key=None,
            help=None,
            on_change=None,
            args=None,
            kwargs=None,
            disabled=False,
            label_visibility="visible",
        )
        if st.form_submit_button():
            if file_uploader_result is not None:
                with open("audio", "wb") as f:
                    f.write(file_uploader_result.read())
                st.session_state["new_audio"] = True


def submit_new_token():
    st.session_state.octoai_api_key = st.session_state.token_text_input


def render_sidebar():
    with st.sidebar:
        if st.session_state.octoai_api_key is None:
            st.text_input(
                "OctoAI API Token (get yours [here](https://octoai.cloud/))",
                type="password",
                key="token_text_input",
                on_change=submit_new_token,
            )
            st.caption(
                """
                See our [docs](https://octo.ai/docs/getting-started/how-to-create-octoai-access-token) for more information on how to get an API token.
            """
            )
        else:

            tab1, tab2 = st.tabs(["Record", "Upload"])

            # 1. Record audio then transcribe
            with tab1:
                do_audio_recording()

            # 2. Transcription from file
            with tab2:
                do_file_upload()

            if RUNNING_SNOWFLAKE_NATIVE:
                st.divider()
                st.write(
                    "This app also installs a SQL UDF which can be called directly within your SQL scripts to transcribe audio stored in Snowflake stages."
                )
                st.markdown(
                    "Usage:\n\n ```SELECT <APP NAME>.transcribe('@STAGE_LOCATION/FILE')```"
                )


def reset_transcript():
    st.session_state["transcript"] = None
    st.session_state["transcript_time"] = None


def main():
    # Set page config
    st.set_page_config(page_title="Transcript and Analysis", layout="wide")

    # Sidebar
    render_sidebar()

    # Main section
    st.title(":loud_sound: Audio Transcription")

    if st.session_state["new_audio"]:
        do_transcription("audio", base_url=st.session_state.whisper_url)
        st.session_state["new_audio"] = False

    # Transcript Section
    if ("transcript" not in st.session_state) or (not st.session_state["transcript"]):
        st.write(
            "This app provides access to a Whisper model endpoint, which you can use to transcribe audio to text."
        )
        st.write(
            "Use the side bar to record audio, or upload a file to be transcribed."
        )
    else:
        if st.button("Reset", on_click=reset_transcript):
            pass
        else:
            render_transcription(transcription=st.session_state["transcript"])

            if "transcript_time" in st.session_state and (
                st.session_state["transcript_time"] is not None
            ):
                st.write(
                    f"Transcription took {st.session_state['transcript_time']:.2f} seconds"
                )

    st.divider()


if __name__ == "__main__":
    if "new_audio" not in st.session_state or st.session_state["new_audio"] is None:
        st.session_state["new_audio"] = False

    if "octoai_api_key" not in st.session_state:
        st.session_state.octoai_api_key = os.environ.get("OCTOAI_API_KEY", None)

    if "whisper_url" not in st.session_state:
        st.session_state["whisper_url"] = WHISPER_API_URL

    main()
