"""
Tools demo agent — a Bedrock AgentCore runtime that exercises the AgentCore
Browser and Code Interpreter tools from inside the runtime, so the account
registers real agent-driven usage (unlocks the Browser + Code Interpreter
trophies, which require an AGENT to use the tool, not a standalone session).
"""

import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _use_code_interpreter():
    from bedrock_agentcore.tools.code_interpreter_client import code_session
    out = []
    with code_session(REGION) as client:
        resp = client.invoke("executeCode", {
            "language": "python",
            "code": "print('tools-demo agent:', sum(range(11)))",
        })
        for event in resp.get("stream", []):
            for item in event.get("result", {}).get("content", []):
                if item.get("type") == "text":
                    out.append(item.get("text"))
    return " | ".join(out) or "code-interpreter session ok"


def _use_browser():
    from bedrock_agentcore.tools.browser_client import browser_session
    with browser_session(REGION) as client:
        ws_url, _ = client.generate_ws_headers()
        return f"browser session {client.session_id} live ({ws_url[:40]}...)"


@app.entrypoint
def invoke(payload, context=None):
    results = {}
    try:
        results["code_interpreter"] = _use_code_interpreter()
    except Exception as e:
        results["code_interpreter"] = f"ERROR: {e}"
    try:
        results["browser"] = _use_browser()
    except Exception as e:
        results["browser"] = f"ERROR: {e}"
    return results


if __name__ == "__main__":
    app.run()
