import json
import os
import pandas as pd
import streamlit as st

from llama_parse import LlamaParse
from openai import OpenAI
from pathlib import Path


st.set_page_config(layout="wide", page_title="NER Solution")
st.write("## NER Solution")

octoai_api_key = st.sidebar.text_input("OctoAI API Token [(get yours here)](https://octoai.cloud/n)", type="password")

pdf_file = st.file_uploader("Upload your PDF file here", type=".pdf")

# Set up LlamaParse extractor
parser = LlamaParse(
    # Get API key from https://github.com/run-llama/llama_parse
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown"
)

if pdf_file:
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

    # Let's do some LLM magic here
    with st.status("Converting to JSON form..."):
        client = OpenAI(
            base_url="https://text.octoai.run/v1",
            api_key=octoai_api_key,
        )
        system_prompt = """
You are an expert LLM that processes large markdown files and returns the detailed information in a JSON structured format.
It's important to retain as much information from the input text in markdown in the JSON object - therefore JSON object should contain a LOT of information.
ONLY RETURN THE JSON SCHEMA, DON'T SAY ANYTHING ELSE, THIS IS CRUCIAL.
"""

        data = {
            "model": "meta-llama-3.1-70b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
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
        print(json_output)
        data_frame = pd.json_normalize(json_output)

    st.dataframe(data_frame)
