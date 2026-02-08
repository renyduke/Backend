#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
