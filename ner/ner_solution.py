import json
import os
import pandas as pd
import streamlit as st
import yaml

from code_editor import code_editor
from llama_parse import LlamaParse
from openai import OpenAI
from pathlib import Path

def convert_to_json_schema(yaml_str):
    # Process yaml_dict
    yaml_dict = yaml.load(yaml_str, Loader=yaml.SafeLoader)

    # Prepare the return string
    ret_str = "{\"properties\": {"
    for name, value in yaml_dict.items():
        description = ""
        if "desc" in value:
            description = value["desc"]
        ret_str += "\"{}\": {{".format(name)
        ret_str += "\"description\": \"{}\", ".format(description)
        ret_str += "\"title\": \"{}\", ".format(name.replace("_", " ").title())
        ret_str += "\"type\": \"string\"}, "
    ret_str = ret_str[:-2]
    ret_str += "}, \"required\": ["
    for name, value in yaml_dict.items():
        ret_str += "\"{}\", ".format(name)
    ret_str = ret_str[:-2]
    ret_str += "], \"title\": \"JSONObject\", \"type\": \"object\"}"

    return ret_str

st.set_page_config(layout="wide", page_title="NER Solution")
st.write("## NER Solution")

octoai_api_key = st.sidebar.text_input("OctoAI API Token [(get yours here)](https://octoai.cloud/n)", type="password")

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

response_dict = code_editor(code=yaml_format, lang="yaml")

pdf_file = st.file_uploader("Upload your PDF file here", type=".pdf")

# Set up LlamaParse extractor
parser = LlamaParse(
    # Get API key from https://github.com/run-llama/llama_parse
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown"
)

if pdf_file and octoai_api_key:
    # Preprocess PDF
    with st.status("Processing the PDFs into Markdown form..."):
        # Store to disk
        # FIXME - tmoreau: let's not do this in the final version
        fp = Path("./", pdf_file.name)
        with open(fp, mode='wb') as w:
            w.write(pdf_file.read())
        # Read in first document
        documents = parser.load_data(Path("./", pdf_file.name))
        doc_str = ""
        for document in documents:
            doc_str += document.text
            doc_str += "\n"

    # Prepare the JSON schema
    json_schema = convert_to_json_schema(response_dict['text'])

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
                {"role": "user", "content": doc_str}
            ],
            "temperature": 0,
            "max_tokens": 131072
        }
        # Derive output values
        response = client.chat.completions.create(**data)
        json_output = response.choices[0].message.content
        json_output = json_output.replace("```json", "")
        json_output = json_output.replace("```", "")
        json_output = json.loads(json_output)
        data_frame = pd.json_normalize(json_output)

    st.dataframe(data_frame)
