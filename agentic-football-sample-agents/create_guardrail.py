"""Create the Bedrock Guardrail for the Guardrails trophy.

Creates a minimal content-filter guardrail (reuses it if it already exists)
and prints the env vars to export before deploying the agents.

Usage:
    export AWS_DEFAULT_REGION=us-east-1   # + event credentials in env
    python create_guardrail.py
"""

import os
import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
NAME = "nubox-dream-team-guardrail"

client = boto3.client("bedrock", region_name=REGION)

# Reuse if it already exists
existing = None
for g in client.list_guardrails().get("guardrails", []):
    if g["name"] == NAME:
        existing = g
        break

if existing:
    gid = existing["id"]
    print(f"Guardrail '{NAME}' already exists: {gid}")
else:
    resp = client.create_guardrail(
        name=NAME,
        description="Guardrail basico del equipo Nubox Dream Team (AWS Agentic Football Cup)",
        contentPolicyConfig={
            "filtersConfig": [
                {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "INSULTS", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "VIOLENCE", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
            ]
        },
        blockedInputMessaging="Instruccion bloqueada por guardrail del equipo.",
        blockedOutputsMessaging="Respuesta bloqueada por guardrail del equipo.",
    )
    gid = resp["guardrailId"]
    print(f"Guardrail created: {gid}")

print("\nExport these before deploying (PowerShell):")
print(f'  $env:GUARDRAIL_ID = "{gid}"')
print('  $env:GUARDRAIL_VERSION = "DRAFT"')
print("\nOr bash:")
print(f'  export GUARDRAIL_ID={gid} GUARDRAIL_VERSION=DRAFT')
