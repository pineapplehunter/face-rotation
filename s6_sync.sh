#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REMOTE_DIR="/home/sailer/daniel/sail/face_rotation"

EXCLUDE1="poetry.lock"
EXCLUDE2=".venv"

# rsync -avrzu --exclude="$EXCLUDE1" --exclude="$EXCLUDE2" "$BASE_DIR/" s6:"$REMOTE_DIR/"
# rsync -avrzu --exclude="$EXCLUDE1" --exclude="$EXCLUDE2" s6:"$REMOTE_DIR/" "$BASE_DIR/"

rsync -avrzu --exclude="$EXCLUDE2" "$BASE_DIR/" s6sailer:"$REMOTE_DIR/"
rsync -avrzu --exclude="$EXCLUDE2" s6sailer:"$REMOTE_DIR/" "$BASE_DIR/"
