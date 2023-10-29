#!/bin/bash
# shellcheck source=tasks/pretty_log.sh
set -euo pipefail

DISCORD_MAINTAINER_ID=${DISCORD_MAINTAINER_ID:-""}

# clear command-line parameters
set --
source tasks/pretty_log.sh

modal secret create docchat-frontend-secret \
  DISCORD_AUTH="$DISCORD_AUTH" \
  DISCORD_PUBLIC_KEY="$DISCORD_PUBLIC_KEY" \
  DISCORD_CLIENT_ID="$DISCORD_CLIENT_ID" \
  DISCORD_MAINTAINER_ID="$DISCORD_MAINTAINER_ID" \
  DISCORD_ENVIRONMENT="$DISCORD_ENVIRONMENT" \
  MODAL_USER_NAME="$MODAL_USER_NAME" \
  MODAL_BACKEND_NAME="$MODAL_BACKEND_NAME" \
  MODAL_ENDPOINT_NAME="$MODAL_ENDPOINT_NAME"