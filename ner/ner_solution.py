import json
import os
import pandas as pd
import streamlit as st
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


st.set_page_config(layout="wide", page_title="NER Solution")
st.write("## NER Playground")

if "octoai_api_key" not in st.session_state:
    st.session_state["octoai_api_key"] = os.environ.get("OCTOAI_API_KEY", "")

octoai_api_key = st.sidebar.text_input(
    "OctoAI API Token [(get yours here)](https://octoai.cloud/n)",
    type="password",
    value=st.session_state.octoai_api_key,
)


#################################################
# Section 1: Inputs


pdf_file = st.sidebar.file_uploader("Upload your PDF file here", type=".pdf")

# Default schema - in a YAML file format

yaml_format = """
# Describe the fields of information in YAML format
doc_title:
    desc: title of the document
authors:
    desc: list of authors
author_emails:
    desc: list of author emails
executive_summary:
    desc: executive summary of the document
"""

code_response = code_editor(code=yaml_format, lang="yaml")

# Set up LlamaParse extractor
parser = LlamaParse(
    # Get API key from https://github.com/run-llama/llama_parse
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown",
)

website_url = st.sidebar.text_input("Enter the URL of the website to scrape")

web_parser = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])

#################################################
# Section 2: Processing the inputs

if pdf_file and octoai_api_key:
    # Preprocess PDF
    with st.status("Processing the PDFs into Markdown form..."):
        # Store to disk
        # FIXME - tmoreau: let's not do this in the final version
        fp = Path("./", pdf_file.name)
        with open(fp, mode="wb") as w:
            w.write(pdf_file.read())
        # Read in first document
        documents = parser.load_data(Path("./", pdf_file.name))
        doc_str = ""
        for document in documents:
            doc_str += document.text
            doc_str += "\n"
        st.session_state.doc_str = doc_str

elif website_url:
    with st.status("Processing the website into Markdown form..."):
        # Crawl a website:
        crawl_status = web_parser.crawl_url(
            website_url,
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

        st.session_state.doc_str = doc_str


#################################################
# Section 3: Processing the outputs

if "doc_str" in st.session_state.keys() and st.session_state.doc_str != "":
    with st.expander(
        f"See extracted markdown: {st.session_state.doc_str[:50]}", expanded=False
    ):
        tab1, tab2 = st.tabs(["Markdown", "Raw"])

        with tab1:
            st.markdown(st.session_state.doc_str)
        with tab2:
            st.code(st.session_state.doc_str, language="markdown")

    # Prepare the JSON schema
    json_schema = convert_to_json_schema(code_response["text"])

    # Let's do some LLM magic here
    with st.status("Converting to JSON form..."):
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
                {"role": "system", "content": system_prompt.format(json_schema)},
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
        data_frame = pd.json_normalize(json_output)

    st.dataframe(data_frame)
