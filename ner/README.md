# NER Playground
Today, data can be found in many different places. In order to use it however one must extract it and process it to have it in a useable format. As it turns out Large Language Models (LLMs) are a great tool for this purpose.

NER Playground is a place to try Open Source LLMs for data extraction from a varied source of media: PDFs, images, audio, websites, and more!

NET Playground is powered by LLMs served by [OctoAI](https://octo.ai)


## Prerequisites
The main prerequisite to get started with the NER Playground is an OctoAI account and an API Key. These can be created at [Octo.AI](https://octo.ai).

## Overview
Using LLMs for data extraction can benefit several industries and applications:

* Healthcare: Create reports from recordings of patient consultations.
* Finance: Extract structured pieces of information from large sets of irregular documents like company quarterly reportings.
* Law: Extract subject names, company names, addresses, facts, and other key points from contracts.
* Education: Create preparation cards based on student assignments.

## For Developers
Developers are encouraged to see the code for this application and create their own. This project has been packaed with [Poetry](https://python-poetry.org/docs/) for Python.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/octoml/octoai-solutions)

### Running locally
First input your different api keys in `env.sh`. Then:
```bash
source env.sh
```

Now, let's setup a new poetry environment:
```bash
poetry install --no-root
```

Now run via:
```bash
poetry run streamlit run ner_solution.py
```
