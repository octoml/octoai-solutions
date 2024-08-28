# OctoAI Solutions
A collection of reference solutions built on top of OctoAI SaaS

## Project setup
Some of the solutions are implemented as poetry projects.

### Creating requirements.txt from poetry
```bash
poetry export --without-hashes --format=requirements.txt > requirements.txt
```

### Synching requirements.txt to poetry
```bash
cat requirements.txt | xargs poetry add
```
