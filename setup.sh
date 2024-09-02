#!/bin/bash

# Install spaCy model from Hugging Face
pip install https://huggingface.co/spacy/es_core_news_sm/resolve/main/es_core_news_sm-any-py3-none-any.whl

# Validate the installation of Spacy models
python -m spacy validate
