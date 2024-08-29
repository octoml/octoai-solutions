import base64
import json
import os
import pandas as pd
import requests
import snowflake
import streamlit as st
import tempfile
import yaml

from code_editor import code_editor
from llama_parse import LlamaParse
from firecrawl import FirecrawlApp
from openai import OpenAI
from pathlib import Path
from snowflake.connector.pandas_tools import write_pandas


def convert_to_json_schema(yaml_str):
    # Process yaml_dict
    yaml_dict = yaml.load(yaml_str, Loader=yaml.SafeLoader)

    # Prepare the return string
    ret_str = '{"properties": {'
    for name, value in yaml_dict.items():
        description = ""
        if "desc" in value:
            description = value["desc"]
        ret_str += '"{}": {{'.format(name)
        ret_str += '"description": "{}", '.format(description)
        ret_str += '"title": "{}", '.format(name.replace("_", " ").title())
        ret_str += '"type": "string"}, '
    ret_str = ret_str[:-2]
    ret_str += '}, "required": ['
    for name, value in yaml_dict.items():
        ret_str += '"{}", '.format(name)
    ret_str = ret_str[:-2]
    ret_str += '], "title": "JSONObject", "type": "object"}'

    return ret_str


def transcribe_audio(file_path: str, octoai_token: str):
    """
    Takes the file path of an audio file and transcribes it to text.

    Returns a string with the transcribed text.
    """
    with open(file_path, "rb") as f:
        encoded_audio = str(base64.b64encode(f.read()), "utf-8")
        reply = requests.post(
            "https://whisper-4jkxk521l3v1.octoai.run/predict",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {octoai_token}",
            },
            json={"audio": encoded_audio},
            timeout=300,
        )
        try:
            transcript = reply.json()["transcription"]
        except Exception as e:
            print(e)
            print(reply.text)
            raise ValueError("The transcription could not be completed.")

    return transcript


def reset_dataframe():
    """
    Resets the dataframe to an empty state.
    """
    st.session_state["data_frame"] = pd.DataFrame()


def update_dataframe(json_output):
    """
    Takes the JSON output from the LLM and updates the dataframe with the extracted entities.

    It will extend the dataframe with the new data.

    This will directly update st.session_state.data_frame.
    """
    if "data_frame" not in st.session_state:
        reset_dataframe()

    # Extend the dataframe
    data_frame = st.session_state.data_frame
    new_data = pd.json_normalize(json_output)
    data_frame = pd.concat([data_frame, new_data], ignore_index=True)
    st.session_state.data_frame = data_frame


st.set_page_config(layout="wide", page_title="NER Playground")
st.write("## NER Playground")
st.caption("Named Entity Recognition Playground.")

# st.sidebar.image("assets/octoai_electric_blue.png", width=200)

if "octoai_api_key" not in st.session_state:
    st.session_state["octoai_api_key"] = os.environ.get("OCTOAI_API_KEY", "")

octoai_api_key = st.sidebar.text_input(
    "OctoAI API Token (get yours [here](https://octoai.cloud/))",
    type="password",
    value=st.session_state.octoai_api_key,
)
st.sidebar.caption(
    """
    See our [docs](https://octo.ai/docs/getting-started/how-to-create-octoai-access-token) for more information on how to get an API token.
"""
)

with st.sidebar.expander("Snowflake Settings", expanded=False):
    snowflake_account = st.text_input("Snowflake Account", value="")
    snowflake_user = st.text_input("Snowflake User", value="")
    snowflake_password = st.text_input("Snowflake Password", value="", type="password")
    snowflake_warehouse = st.text_input("Snowflake Warehouse", value="")
    snowflake_database = st.text_input("Snowflake Database", value="")
    snowflake_schema = st.text_input("Snowflake Schema", value="")
    snowflake_table = st.text_input("Snowflake Table", value="")

#################################################
# Section 1: Inputs


upload_files = st.sidebar.file_uploader(
    "Upload your PDF file here",
    type=[".pdf", ".mp3", ".mp4", ".wav"],
    accept_multiple_files=True,
    key="upload_files",
)

# Default schema - in a YAML file format
yaml_format = """
# Describe the fields of information in YAML format
# Tip: Ctrl + Enter saves the schema (Cmd + Enter on Mac).
doc_title:
    desc: title of the document
authors:
    desc: list of authors
author_emails:
    desc: list of author emails
executive_summary:
    desc: executive summary of the document
"""


def update_json_schema(code):
    # Prepare the JSON schema
    json_schema = convert_to_json_schema(code)
    st.session_state["json_schema"] = json_schema


if "json_schema" not in st.session_state:
    update_json_schema(yaml_format)


# add a button with text: 'Copy'
custom_btns = [
    {
        "name": "Copy",
        "feather": "Copy",
        "alwaysOn": True,
        "commands": ["copyAll"],
        "hasText": True,
        "style": {"top": "0.46rem", "right": "0.4rem"},
    },
    {
        "name": "Save",
        "feather": "Save",
        "alwaysOn": True,
        "commands": ["submit"],
        "hasText": True,
        "style": {"bottom": "0.46rem", "right": "0.4rem"},
    },
]
code_response = code_editor(code=yaml_format, lang="yaml", buttons=custom_btns)
if code_response["text"]:
    print(code_response["text"])
    update_json_schema(code_response["text"])

# Set up LlamaParse extractor
parser = LlamaParse(
    # Get API key from https://github.com/run-llama/llama_parse
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown",
)

website_url = st.sidebar.text_input(
    "Enter the URL of the website to scrape", key="website_url"
)
st.sidebar.caption("Use comma for multiple URLs.")

web_parser = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])

#################################################
# Section 2: Processing the inputs
st.session_state.doc_str = []
if octoai_api_key:

    if len(upload_files):
        if len(upload_files) == 1:
            spinner_message = f"Processing {upload_files[0].name} into Markdown..."
        else:
            spinner_message = f"Processing {len(upload_files)} files into Markdown..."
        # Preprocess documents
        with st.status(spinner_message):
            for upload_file in upload_files:
                # Store to disk
                with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
                    with open(tf.name, mode="wb") as w:
                        w.write(upload_file.read())
                    # PDF handling
                    if upload_file.name.endswith(".pdf"):
                        # Read in first document
                        documents = parser.load_data(tf.name)
                        doc_str = ""
                        for document in documents:
                            doc_str += document.text
                            doc_str += "\n"
                    # Audio file handling
                    elif (
                        upload_file.name.endswith(".mp3")
                        or upload_file.name.endswith(".mp4")
                        or upload_file.name.endswith(".wav")
                    ):
                        doc_str = transcribe_audio(tf.name, octoai_api_key)
                    st.session_state.doc_str.append(doc_str)

    elif website_url:
        if "," not in website_url:
            website_url_list = [website_url]
            spinner_message = f"Scrapping {website_url} into Markdown..."
        else:
            website_url_list = website_url.split(",")
            spinner_message = (
                f"Scraping {len(website_url_list)} websites into Markdown..."
            )

        # Remove whitespaces
        website_url_list = [url.strip() for url in website_url_list]

        with st.status(spinner_message):
            for url in website_url_list:
                # Crawl a website:
                crawl_status = web_parser.crawl_url(
                    url,
                    params={
                        "limit": 5,
                        "scrapeOptions": {"formats": ["markdown"]},
                        "excludePaths": ["/blog", "/docs"],
                    },
                    wait_until_done=True,
                    poll_interval=20,
                )
                doc_str = ""
                for page in crawl_status["data"]:
                    doc_str += f"# {page['metadata']['title']}\n"
                    doc_str += page["markdown"]
                    doc_str += "\n"
                st.session_state.doc_str.append(doc_str)


#################################################
# Section 3: Processing the outputs

if "doc_str" in st.session_state.keys() and len(st.session_state.doc_str) > 0:
    with st.expander(
        f"See the extracted markdown:\n `{st.session_state.doc_str[0][:32]}`...",
        expanded=False,
    ):
        tab1, tab2 = st.tabs(["Markdown", "Raw"])

        with tab1:
            st.markdown(st.session_state.doc_str[0])
        with tab2:
            st.code(st.session_state.doc_str[0], language="markdown")

    # Let's do some LLM magic here
    with st.status("Converting to JSON form..."):
        json_outputs = []
        for doc_str in st.session_state.doc_str:
            client = OpenAI(
                base_url="https://text.octoai.run/v1",
                api_key=octoai_api_key,
            )
            system_prompt = """
You are an expert LLM that processes large files and extracts entities according to the provided JSON schema:

{}

ONLY RETURN THE JSON OBJECT, DON'T SAY ANYTHING ELSE, THIS IS CRUCIAL.
"""

            data = {
                "model": "meta-llama-3.1-70b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt.format(st.session_state.json_schema),
                    },
                    {"role": "user", "content": doc_str},
                ],
                "temperature": 0,
                "max_tokens": 131072,
            }
            # Derive output values
            response = client.chat.completions.create(**data)
            json_output = response.choices[0].message.content
            json_output = json_output.replace("```json", "")
            json_output = json_output.replace("```", "")
            json_output = json.loads(json_output)
            json_outputs.append(json_output)

            # Update the dataframe
            update_dataframe(json_output)

if "data_frame" in st.session_state and not st.session_state.data_frame.empty:
    st.dataframe(st.session_state.data_frame)
    st.button("Reset Dataframe", on_click=reset_dataframe)

    # Store to Snowflake
    if (
        snowflake_account
        and snowflake_user
        and snowflake_password
        and snowflake_warehouse
        and snowflake_database
        and snowflake_schema
        and snowflake_table
    ):
        if st.button("Store to Snowflake table"):

            with st.status("Uploading to Snowflake..."):

                # Snowflake Connector
                conn = snowflake.connector.connect(
                    user=snowflake_user,
                    password=snowflake_password,
                    account=snowflake_account,
                    warehouse=snowflake_warehouse,
                    database=snowflake_database,
                    schema=snowflake_schema,
                )

                # Create Snowflake Warehouse, Database, Schema if they do not exist
                conn.cursor().execute(
                    "CREATE WAREHOUSE IF NOT EXISTS {}".format(snowflake_warehouse)
                )
                conn.cursor().execute(
                    "CREATE DATABASE IF NOT EXISTS {}".format(snowflake_database)
                )
                conn.cursor().execute("USE DATABASE {}".format(snowflake_database))
                conn.cursor().execute(
                    "CREATE SCHEMA IF NOT EXISTS {}".format(snowflake_schema)
                )
                conn.cursor().execute("USE WAREHOUSE {}".format(snowflake_warehouse))
                conn.cursor().execute("USE DATABASE {}".format(snowflake_database))
                conn.cursor().execute("USE SCHEMA {}".format(snowflake_schema))

                # Write pandas DataFrame to Snowflake
                success, _, _, _ = write_pandas(
                    conn=conn,
                    df=st.session_state.data_frame,
                    table_name=snowflake_table,
                    schema=snowflake_schema,
                    database=snowflake_database,
                    auto_create_table=True,
                    overwrite=True,
                )
