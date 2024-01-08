#!/bin/bash
# shellcheck source=tasks/pretty_log.sh
source /home/mi/git_repos/ask-fsdl/.env.dev

echo "MONGODB_USER: $MONGODB_USER"
echo "MONGODB_HOST: $MONGODB_HOST"
echo "MONGODB_PASSWORD: $MONGODB_PASSWORD"
echo "OPENAI_API_KEY: $OPENAI_API_KEY"
echo "GANTRY_API_KEY: $GANTRY_API_KEY"

set -euo pipefail

GANTRY_API_KEY=${GANTRY_API_KEY:-""}

# clear command-line parameters
set --
source tasks/pretty_log.sh

modal secret create mongodb-fsdl MONGODB_USER="$MONGODB_USER" MONGODB_HOST="$MONGODB_HOST" MONGODB_PASSWORD="$MONGODB_PASSWORD"
modal secret create openai-api-key-fsdl OPENAI_API_KEY="$OPENAI_API_KEY"
modal secret create bigquery_dataset BIGQUERY_DATASET="$BIGQUERY_DATASET"