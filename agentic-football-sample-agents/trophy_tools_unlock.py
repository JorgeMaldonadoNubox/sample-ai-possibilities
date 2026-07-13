"""Unlock the AgentCore Browser and Code Interpreter trophies.

Starts a real AgentCore Browser session and a Code Interpreter session in the
account (that's the service usage the event portal tracks), runs a trivial
action in each, and cleans up. Safe to run multiple times.

Usage:
    export AWS_DEFAULT_REGION=us-east-1   # + event credentials in env
    python trophy_tools_unlock.py
"""

import os
import sys

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def unlock_code_interpreter() -> bool:
    print("=== AgentCore Code Interpreter ===")
    try:
        from bedrock_agentcore.tools.code_interpreter_client import code_session

        with code_session(REGION) as client:
            response = client.invoke("executeCode", {
                "language": "python",
                "code": "print('Nubox Dream Team - trophy unlock', 2+2)",
            })
            for event in response.get("stream", []):
                result = event.get("result", {})
                for item in result.get("content", []):
                    if item.get("type") == "text":
                        print(f"  output: {item.get('text')}")
        print("  OK — Code Interpreter session created and used\n")
        return True
    except Exception as e:
        print(f"  FAILED: {e}\n")
        return False


def unlock_browser() -> bool:
    print("=== AgentCore Browser ===")
    try:
        from bedrock_agentcore.tools.browser_client import browser_session

        with browser_session(REGION) as client:
            # Starting the session is the tracked usage; grab connection
            # details to prove the browser is live.
            ws_url, headers = client.generate_ws_headers()
            print(f"  session id: {client.session_id}")
            print(f"  ws endpoint: {ws_url[:80]}...")
        print("  OK — Browser session created and closed\n")
        return True
    except Exception as e:
        print(f"  FAILED: {e}\n")
        return False


if __name__ == "__main__":
    ok_ci = unlock_code_interpreter()
    ok_br = unlock_browser()
    print("=" * 40)
    print(f"Code Interpreter: {'UNLOCKED' if ok_ci else 'check IAM/creds'}")
    print(f"Browser:          {'UNLOCKED' if ok_br else 'check IAM/creds'}")
    sys.exit(0 if (ok_ci and ok_br) else 1)
