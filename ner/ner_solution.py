import json
import os
import pandas as pd
import snowflake
import streamlit as st
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

st.set_page_config(layout="wide", page_title="NER Playground")
st.write("## NER Playground")

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


pdf_files = st.sidebar.file_uploader(
    "Upload your PDF file here", type=".pdf", accept_multiple_files=True
)

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

website_url = st.sidebar.text_input(
    "Enter the URL of the website to scrape (use comma for multiple URLs)"
)

web_parser = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])

#################################################
# Section 2: Processing the inputs
st.session_state.doc_str = []
if octoai_api_key:

    if len(pdf_files):
        if len(pdf_files) == 1:
            spinner_message = f"Processing {pdf_files[0].name} into Markdown..."
        else:
            spinner_message = f"Processing {len(pdf_files)} PDFs into Markdown..."
        # Preprocess PDF
        with st.status(spinner_message):
            for pdf_file in pdf_files:
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
                st.session_state.doc_str.append(doc_str)

    elif website_url:
        if "," not in website_url:
            website_url_list = [website_url]
            spinner_message = f"Scraping {website_url} into Markdown..."
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
        f"See extracted markdown:\n{st.session_state.doc_str[0][:64]}...",
        expanded=False,
    ):
        tab1, tab2 = st.tabs(["Markdown", "Raw"])

        with tab1:
            st.markdown(st.session_state.doc_str[0])
        with tab2:
            st.code(st.session_state.doc_str[0], language="markdown")

    # Prepare the JSON schema
    json_schema = convert_to_json_schema(code_response["text"])

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
            json_outputs.append(json_output)
        data_frame = pd.json_normalize(json_outputs)

    st.dataframe(data_frame)

    # Store to Snowflake
    if snowflake_account and snowflake_user and snowflake_password and snowflake_warehouse and snowflake_database and snowflake_schema and snowflake_table:
        if st.button("Store to Snowflake table"):

            with st.status("Uploading to Snowflake..."):

                # Snowflake Connector
                conn = snowflake.connector.connect(
                    user=snowflake_user,
                    password=snowflake_password,
                    account=snowflake_account,
                    warehouse=snowflake_warehouse,
                    database=snowflake_database,
                    schema=snowflake_schema
                )

                # Create Snowflake Warehouse, Database, Schema if they do not exist
                conn.cursor().execute("CREATE WAREHOUSE IF NOT EXISTS {}".format(snowflake_warehouse))
                conn.cursor().execute("CREATE DATABASE IF NOT EXISTS {}".format(snowflake_database))
                conn.cursor().execute("USE DATABASE {}".format(snowflake_database))
                conn.cursor().execute("CREATE SCHEMA IF NOT EXISTS {}".format(snowflake_schema))
                conn.cursor().execute("USE WAREHOUSE {}".format(snowflake_warehouse))
                conn.cursor().execute("USE DATABASE {}".format(snowflake_database))
                conn.cursor().execute("USE SCHEMA {}".format(snowflake_schema))

                # Write pandas DataFrame to Snowflake
                success, _, _, _ = write_pandas(
                    conn=conn,
                    df=data_frame,
                    table_name=snowflake_table,
                    schema=snowflake_schema,
                    database=snowflake_database,
                    auto_create_table=True,
                    overwrite=True
                )
