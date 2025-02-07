#!/bin/bash
# Install Python
curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install dependencies
pip install -r requirements.txt
