import base64
import json
import os
import pandas as pd
import requests
import boto3

import streamlit as st
import tempfile
import yaml

from code_editor import code_editor
from llama_parse import LlamaParse
from firecrawl import FirecrawlApp
from openai import OpenAI
from pathlib import Path


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


def transcribe_audio(encoded_audio: str, octoai_token: str):
    """
    Takes the file path of an audio file and transcribes it to text.

    Returns a string with the transcribed text.
    """
    reply = requests.post(
        "https://whisper2-or1pkb9b656p.octoai.run/predict",
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


def process_image(encoded_image: str, octoai_token: str, yaml: str):
    print(yaml)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe what you see in the image in great detail. Be as exhaustive and factual as possible. Provide detail according to the JSON description below:\n{}".format(yaml),
                },
                {"type": "image_url", "image_url": {"url": encoded_image}},
            ],
        }
    ]
    url = "https://text.octoai.run/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {octoai_token}",
    }
    data = {
        "messages": messages,
        "model": "phi-3.5-vision-instruct",
        "max_tokens": 1024,
        "presence_penalty": 0,
        "temperature": 0.1,
        "top_p": 0.9,
        "stream": "False",
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = json.loads(response.content.decode("utf-8"))
    return response["choices"][0]["message"]["content"]


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


def submit_onclick():
    st.session_state["process_new_inputs"] = True


def submit_new_token():
    st.session_state.octoai_api_key = st.session_state.token_text_input


st.set_page_config(layout="wide", page_title="Multi-Modal Data Extractor")

if "octoai_api_key" not in st.session_state:
    st.session_state.octoai_api_key = os.environ.get("OCTOAI_API_KEY", None)

# Sidebar tabs sections
with st.sidebar:

    if st.session_state.octoai_api_key is None:
        octoai_api_key = st.text_input(
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
        with st.form("input-form", clear_on_submit=False, border=True):
            tab1, tab2, tab3  = st.tabs(["Local Files", "URLs", "S3"])

            # Local files
            with tab1:
                upload_files = st.file_uploader(
                    "Upload your PDFs/audio/JPEG files here",
                    type=[".pdf", ".mp3", ".mp4", ".wav", ".jpg", ".jpeg"],
                    accept_multiple_files=True,
                    key="upload_files",
                )
                st.caption("Click on submit after uploading to process the files.")

            # URLs
            with tab2:
                website_url = st.text_input(
                    "Enter the URL(s) of the website to scrape", key="website_url"
                )
                st.caption("Use comma for multiple URLs.")

            # S3
            with tab3:
                aws_access_key_id = st.text_input(
                    "AWS Access Key ID", value="AWSACCESSKEYID"
                )
                aws_secret_access_key = st.text_input(
                    "AWS Secret Key", type="password", value="asdf"
                )
                aws_s3_bucket = st.text_input(
                    "AWS S3 bucket", value="bucket-name"
                )
                aws_s3_bucket_path = st.text_input(
                    "Path to directory to process", value="path/to/dir/"
                )

            st.form_submit_button("Submit", on_click=submit_onclick)

    st.write(
        "See the code in [GitHub](https://github.com/octoml/octoai-solutions/tree/main/ner)."
    )
    st.write(
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/octoml/octoai-solutions)"
    )

st.write("## Multi-Modal Data Extractor")
st.caption("Powered by OctoAI.")

#################################################
# Section 1: Inputs


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
st.session_state["yaml_format"] = yaml_format


def update_json_schema(code):
    # Prepare the JSON schema
    json_schema = convert_to_json_schema(code)
    st.session_state["json_schema"] = json_schema
    st.session_state["yaml_format"] = code


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
    update_json_schema(code_response["text"])

if not st.session_state.get("process_new_inputs", False) and (
    "data_frame" not in st.session_state or st.session_state.data_frame.empty
):
    st.write("👈 Upload files or enter URLs on the side bar to extract entities.")

#################################################
# Section 2: Processing the inputs

# Set up LlamaParse extractor
parser = LlamaParse(
    # Get API key from https://github.com/run-llama/llama_parse
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown",
)

web_parser = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])


st.session_state.doc_str = []
if (st.session_state.octoai_api_key is not None) and (
    st.session_state.get("process_new_inputs", False)
):

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
                        # Convert the image to base64 string
                        with open(tf.name, "rb") as f:
                            encoded_audio = str(base64.b64encode(f.read()), "utf-8")
                            doc_str = transcribe_audio(
                                encoded_audio, st.session_state.octoai_api_key
                            )
                    # Image file handling
                    elif (
                        upload_file.name.endswith("jpg")
                        or upload_file.name.endswith("jpeg")
                    ):
                        # Convert the images to base64 string
                        with open(tf.name, "rb") as f:
                            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                            encoded_image = f"data:image/png;base64,{encoded_image}"
                            doc_str = process_image(
                                encoded_image,
                                st.session_state.octoai_api_key,
                                str(yaml.load(st.session_state["yaml_format"], Loader=yaml.SafeLoader))
                            )
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
            got_error = ""
            for url in website_url_list:
                # Crawl a website:
                try:
                    crawl_status = web_parser.crawl_url(
                        url,
                        params={
                            "limit": 3,
                            "scrapeOptions": {"formats": ["markdown"]},
                            "excludePaths": ["/blog", "/docs"],
                        },
                        poll_interval=20,
                    )
                except Exception as e:
                    print(e)
                    got_error = url
                    break
                else:
                    doc_str = ""
                    for page in crawl_status["data"]:
                        doc_str += f"# {page['metadata']['title']}\n"
                        doc_str += page["markdown"]
                        doc_str += "\n"
                    st.session_state.doc_str.append(doc_str)
        if got_error:
            st.error(
                f"An error occurred while processing {got_error}. Please refresh and try again."
            )

    elif aws_access_key_id and aws_secret_access_key:

        # Create an S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        # Get the list in the bucket directory
        result = s3_client.list_objects(
            Bucket=aws_s3_bucket,
            Prefix=aws_s3_bucket_path,
            Delimiter='/'
        )

        if len(result.get('Contents')) == 1:
            spinner_message = f"Processing {result.get('Contents')[0].get('Key')} into Markdown..."
        else:
            spinner_message = f"Processing {len(result.get('Contents'))-1} files into Markdown..."
        # Preprocess documents
        with st.status(spinner_message):
            for bucket_file in result.get('Contents'):
                f_name = bucket_file.get('Key')
                data = s3_client.get_object(Bucket=aws_s3_bucket, Key=f_name)
                if f_name == bucket_file:
                    continue
                # PDF handling
                if f_name.endswith(".pdf"):
                    # Read in first document
                    documents = parser.load_data(
                        data['Body'].read(),
                        extra_info={"file_name": f_name}
                    )
                    doc_str = ""
                    for document in documents:
                        doc_str += document.text
                        doc_str += "\n"
                    st.session_state.doc_str.append(doc_str)
                # Audio file handling
                elif (
                    f_name.endswith(".mp3")
                    or f_name.endswith(".mp4")
                    or f_name.endswith(".wav")
                ):
                    encoded_audio = str(base64.b64encode(data['Body'].read()), "utf-8")
                    doc_str = transcribe_audio(
                        encoded_audio, st.session_state.octoai_api_key
                    )
                    st.session_state.doc_str.append(doc_str)
                elif f_name.endswith("jpg") or f_name.endswith("jpeg"):
                    # Convert the images to base64 string
                    encoded_image = base64.b64encode(data['Body'].read()).decode("utf-8")
                    encoded_image = f"data:image/png;base64,{encoded_image}"
                    doc_str = process_image(
                        encoded_image,
                        st.session_state.octoai_api_key,
                        str(yaml.load(st.session_state["yaml_format"], Loader=yaml.SafeLoader))
                    )
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
                api_key=st.session_state.octoai_api_key,
            )
            system_prompt = """
You are an expert LLM that processes large files and extracts entities according to the provided JSON schema:

{}

ONLY RETURN THE JSON OBJECT, DON'T SAY ANYTHING ELSE, THIS IS CRUCIAL.
"""

            data = {
                "model": "meta-llama-3.1-405b-instruct",
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
    st.session_state.doc_str = []
    st.session_state.process_new_inputs = False

if "data_frame" in st.session_state and not st.session_state.data_frame.empty:
    st.dataframe(st.session_state.data_frame)
    col1, col2 = st.columns(2)
    with col1:
        st.caption(
            "Upload files or enter URLs on the side bar to extract more entities."
        )
    with col2:
        st.button("Reset Dataframe", on_click=reset_dataframe)
