#!/bin/bash
# bootstrap.sh - run ON a freshly bought spot instance, as root.
#
# Purpose: bring a replacement exam machine into the tailnet and authorize
# the two operator keys, with zero interactive steps:
#   1. Install Tailscale (official install script).
#   2. Join the tailnet with a reusable auth key supplied AT RUNTIME via the
#      TS_AUTH_KEY environment variable. The key is never hardcoded here,
#      never committed, and must be entered through a secure channel only.
#   3. Idempotently authorize the hatch (war-room VM) and DGX carbon_relay
#      public keys in /root/.ssh/authorized_keys.
#   4. Print BOOTSTRAP-READY plus this machine's tailnet IPv4.
#
# Usage:
#   TS_AUTH_KEY=<key> SPOT_NAME=carbon-spot-31 bash bootstrap.sh
# SPOT_NAME is optional (default: carbon-spot-new).
set -euo pipefail

: "${TS_AUTH_KEY:?must be supplied at runtime (env var, secure entry only)}"
SPOT_NAME="${SPOT_NAME:-carbon-spot-new}"

if [ "$(id -u)" != "0" ]; then
  echo "BOOTSTRAP-FAIL: must run as root" >&2
  exit 1
fi

# --- 1. Install Tailscale -------------------------------------------------
if ! command -v tailscale >/dev/null 2>&1; then
  echo "installing tailscale..."
  curl -fsSL https://tailscale.com/install.sh | sh
fi
tailscale version

# --- 2. Join the tailnet (non-interactive, auth key from env only) --------
echo "joining tailnet as ${SPOT_NAME}..."
tailscale up --auth-key="${TS_AUTH_KEY}" --hostname="${SPOT_NAME}" --ssh=false

# Wait for a tailnet IPv4 to be assigned.
TS_IP=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
  TS_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  [ -n "${TS_IP}" ] && break
  sleep 2
done
if [ -z "${TS_IP}" ]; then
  echo "BOOTSTRAP-FAIL: no tailnet IPv4 assigned after tailscale up" >&2
  exit 1
fi

# --- 3. Authorize operator keys (idempotent) -------------------------------
HATCH_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJp9AH0D7psbm85J4iXvRbDRtDTW5kbydl7NRixWAjlW hatch"
RELAY_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFnNLECe8NSVrZaBOQ9E1KVJCkzM3cGJLEqKt1WrjMBJ dgx-carbon-relay-20261109"

install -d -m 700 /root/.ssh
touch /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
for key in "${HATCH_KEY}" "${RELAY_KEY}"; do
  if ! grep -qxF "${key}" /root/.ssh/authorized_keys; then
    echo "${key}" >> /root/.ssh/authorized_keys
    echo "authorized_keys: added ${key##* }"
  else
    echo "authorized_keys: already present ${key##* }"
  fi
done
chmod 600 /root/.ssh/authorized_keys

# --- 4. Report --------------------------------------------------------------
echo "BOOTSTRAP-READY"
echo "hostname: ${SPOT_NAME}"
echo "tailnet-ipv4: ${TS_IP}"
tailscale status | head -n 3 || true
