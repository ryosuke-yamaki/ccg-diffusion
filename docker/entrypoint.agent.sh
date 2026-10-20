#!/bin/bash
# Entry point of the optional agent image (docker/Dockerfile.agent): Git identity,
# Claude Code update, Codex sign-in and MCP server registration, then a shell.
# Every credential comes from the environment (docker-compose.override.example.yml
# passes the keys of .env); a step whose variable is unset is skipped.
set -e

# Git identity of commits made inside the container
if [ -n "${GIT_AUTHOR_NAME:-}" ]; then
    git config --global user.name "${GIT_AUTHOR_NAME}"
fi
if [ -n "${GIT_AUTHOR_EMAIL:-}" ]; then
    git config --global user.email "${GIT_AUTHOR_EMAIL}"
fi
if [ -n "${GIT_SAFE_DIRECTORY:-}" ]; then
    git config --global --add safe.directory "${GIT_SAFE_DIRECTORY}"
fi

# Update Claude Code (the container still starts when this fails)
set +e
echo "[entrypoint] Updating Claude Code..."
claude update 2>/dev/null \
    && echo "[entrypoint] Claude Code is up to date." \
    || echo "[entrypoint] Skipped the Claude Code update (already current or offline)."
set -e

# Codex sign-in. With a ChatGPT plan, mount a host ~/.codex that already holds
# auth.json (CODEX_CONFIG_DIR); CODEX_ACCESS_TOKEN is for trusted automation
# tokens, not for Platform API keys.
set +e
if command -v codex > /dev/null 2>&1; then
    export CODEX_HOME="${CODEX_HOME:-/home/devuser/.codex}"
    mkdir -p "${CODEX_HOME}"
    if [ -n "${CODEX_ACCESS_TOKEN:-}" ]; then
        echo "[entrypoint] Signing in to Codex with CODEX_ACCESS_TOKEN..."
        printf '%s' "${CODEX_ACCESS_TOKEN}" \
            | timeout 60s codex login --with-access-token > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "[entrypoint] Codex login complete."
        else
            echo "[entrypoint] Codex login failed. Check CODEX_ACCESS_TOKEN."
        fi
    elif [ -f "${CODEX_HOME}/auth.json" ]; then
        echo "[entrypoint] Found the Codex auth cache at ${CODEX_HOME}/auth.json."
    else
        echo "[entrypoint] Codex is not signed in: mount a host ~/.codex or run codex login in the container."
    fi
else
    echo "[entrypoint] codex not found; skipped the Codex setup."
fi
set -e

# Register the MCP servers whose token is set. Existing entries are replaced on
# every start, so changed tokens take effect; failures do not stop the container.
set +e

mcp_add() {
    local name=$1; shift
    claude mcp remove "${name}" > /dev/null 2>&1 || true
    claude mcp add "$@" > /dev/null 2>&1
}

echo "[entrypoint] Configuring MCP servers..."

# Remove project-level MCP entries (without credentials) so that only the
# user-level entries below remain.
python3 - <<'EOF'
import json, os, sys

claude_json = os.path.expanduser("~/.claude.json")
if not os.path.exists(claude_json):
    sys.exit(0)

try:
    with open(claude_json) as f:
        data = json.load(f)
except (json.JSONDecodeError, OSError):
    sys.exit(0)

changed = False
for proj in data.get("projects", {}).values():
    mcp = proj.get("mcpServers", {})
    for name in ["notion", "wandb", "github"]:
        if name in mcp:
            del mcp[name]
            changed = True

if changed:
    with open(claude_json, "w") as f:
        json.dump(data, f, indent=2)
EOF

if [ -n "${GITHUB_PERSONAL_ACCESS_TOKEN:-}" ]; then
    mcp_add github --transport http github https://api.githubcopilot.com/mcp \
        --scope user \
        -H "Authorization: Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}"
fi

if [ -n "${NOTION_TOKEN:-}" ]; then
    mcp_add notion notion notion-mcp-server \
        --scope user \
        -e "NOTION_TOKEN=${NOTION_TOKEN}"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    mcp_add wandb --transport http wandb https://mcp.withwandb.com/mcp \
        --scope user \
        -H "Authorization: Bearer ${WANDB_API_KEY}"
fi

echo "[entrypoint] MCP servers configured."

set -e

exec bash
