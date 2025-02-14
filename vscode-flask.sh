#!/bin/bash
#SBATCH --job-name rag-flask-app
#SBATCH --output rag-flask-app.out
#SBATCH --partition gpu
#SBATCH --gpus 2
#SBATCH --time 10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aw181@hdm-stuttgart.de

set -euo pipefail

NORMALIZED_ARGS=$(getopt -o r --long root -- "$@")
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$NORMALIZED_ARGS"

ROOT=false
while true; do
  case "$1" in
    -r | --root ) ROOT=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

mkdir -p "${HOME}/.local/run/iaailauncher/"
SLURM_JOB_ID="${SLURM_JOB_ID:-"interactive"}"
LAUNCHER_INFO_FILE="${HOME}/.local/run/iaailauncher/${SLURM_JOB_ID}.json"

# get a random, free port
SERVER_PORT=$(/usr/bin/python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
USERNAME="$(id -un)"
HOST="$(hostname).mi.hdm-stuttgart.de"

PROXY_ROUTE_NAME="rag-flask-app"
TRAEIK_CONFIG="/var/traefik_routes/${PROXY_ROUTE_NAME}.yml"

CONFIG_FILE="${HOME}/.config/code-server/config.yaml"
CONTAINER_IMAGE_NAME="vscode-server"

function cleanup {
	rm -f "${TRAEIK_CONFIG}"
	rm -f "${LAUNCHER_INFO_FILE}"
}
trap cleanup EXIT

cat << EOF > ${LAUNCHER_INFO_FILE}
	{
		"id": "${SLURM_JOB_ID}",
		"type": "vscode",
		"state": "BUILDING"
	}
EOF

if [[ "$(hostname)" == "deeplearn" ]]; then
	echo "running on deeplearn host not supported. Executing this script with sbatch"
	srun --pty --partition=gpu --gres=gpu:2 --job-name vscode-server --ntasks=1 $0 $NORMALIZED_ARGS $@
	exit 1
fi

run_as_user () {
	enroot start --gpu --mount "${HOME}:${HOME}/mounted_home" --rw "${CONTAINER_IMAGE_NAME}" bash -c "$*"
}

run_as_root () {
	enroot start --gpu --mount "${HOME}:${HOME}/mounted_home" --root --rw "${CONTAINER_IMAGE_NAME}" bash -c "$*"
}

if ! enroot list | grep -q -x "${CONTAINER_IMAGE_NAME}"; then
	echo "${CONTAINER_IMAGE_NAME} container not found, building from image..."
	enroot create --name "${CONTAINER_IMAGE_NAME}" /opt/images/base_221024.sqsh

	run_as_root "curl -fsSL https://code-server.dev/install.sh | sh"

	run_as_user "mkdir -p ${HOME}/.config"
	run_as_user "/usr/bin/code-server --config ${CONFIG_FILE} --install-extension ms-python.python --force"
fi

# get the configured password. the config file is created the first time vscode-server is run which creates a random password
SERVER_PASSWORD=$(run_as_user "awk '/password:/{print \$2}' ${CONFIG_FILE}")

cat << EOF > ${TRAEIK_CONFIG}
http:
  routers:
    ${PROXY_ROUTE_NAME}:
      service: vscode_${PROXY_ROUTE_NAME}_service
      tls: true
      middlewares:
        - ${PROXY_ROUTE_NAME}_redirector
        - ${PROXY_ROUTE_NAME}_prefixstripper
      entrypoints: websecure
      rule: "(Host(\`${HOST}\`) && PathPrefix(\`/${PROXY_ROUTE_NAME}\`))"
  middlewares:
    ${PROXY_ROUTE_NAME}_redirector:
      redirectScheme:
        scheme: https
        permanent: true
    ${PROXY_ROUTE_NAME}_prefixstripper:
      stripPrefix:
        forceSlash: true
        prefixes:
          - "/${PROXY_ROUTE_NAME}"
  services:
    vscode_${PROXY_ROUTE_NAME}_service:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:5000"
EOF

OPEN_URL="https://${HOST}/${PROXY_ROUTE_NAME}/"
cat << EOF > ${LAUNCHER_INFO_FILE}
{
	"id": "${SLURM_JOB_ID}",
	"openUrl": "${OPEN_URL}",
	"password": "${SERVER_PASSWORD}",
	"type": "vscode"
}
EOF

echo "Starting VS Code Server on ${HOST}"
echo -e "Connect using: \033[0;32m${OPEN_URL}\033[0m with password \033[0;32m${SERVER_PASSWORD}\033[0m"

VSCODE_START_CMD="/opt/miniconda3/envs/BAp11/bin/python -m flask--app ${HOME}/mounted_home/RAG_Chatbot/webpage.py run"
if ${ROOT}; then
	run_as_root "${VSCODE_START_CMD}"
else
	run_as_user "${VSCODE_START_CMD}"
fi

cleanup
