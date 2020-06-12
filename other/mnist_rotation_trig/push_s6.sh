#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REMOTE_DIR="/net/nfs/home/takata/sail/"

rsync -avr -P $BASE_DIR s6:$REMOTE_DIR