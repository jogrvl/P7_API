#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

# Forcer la version Python
echo "python-3.10.13" > runtime.txt

# Mise à jour de pip et installation
pip install --upgrade pip
pip install -r requirements.txt
