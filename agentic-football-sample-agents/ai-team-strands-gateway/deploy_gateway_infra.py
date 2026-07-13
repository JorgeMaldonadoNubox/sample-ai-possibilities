"""Deploy the Gateway infra (4 Lambdas + IAM roles + MCP Gateway) via boto3.

Windows/Git-Bash friendly: no `zip` binary, no fileb:// path translation.
Prints GATEWAY_URL=... on stdout for the caller to capture.

Usage:
    export AWS_DEFAULT_REGION=us-east-1   # + event credentials
    python deploy_gateway_infra.py
"""

import io
import os
import sys
import json
import time
import zipfile
import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAMBDA_PREFIX = "afwc-gateway-tool"
LAMBDA_ROLE_NAME = "afwc-gateway-tool-lambda-role"
GW_ROLE_NAME = "AfwcGatewayExecutionRole"
TOOLS = ["calculate_pass_options", "evaluate_shot", "find_open_space", "get_defensive_assignment"]

iam = boto3.client("iam", region_name=REGION)
lam = boto3.client("lambda", region_name=REGION)
sts = boto3.client("sts", region_name=REGION)
ACCOUNT_ID = sts.get_caller_identity()["Account"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def ensure_lambda_role():
    try:
        arn = iam.get_role(RoleName=LAMBDA_ROLE_NAME)["Role"]["Arn"]
        log(f"  Lambda role exists: {arn}")
        return arn
    except iam.exceptions.NoSuchEntityException:
        log("  Creating Lambda execution role...")
        trust = {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
             "Action": "sts:AssumeRole"}]}
        arn = iam.create_role(RoleName=LAMBDA_ROLE_NAME,
                              AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        iam.attach_role_policy(RoleName=LAMBDA_ROLE_NAME,
                               PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
        log("  Waiting 10s for role propagation...")
        time.sleep(10)
        return arn


def package(tool):
    """Zip the tool file in-memory, renaming handler() -> lambda_handler()."""
    src = os.path.join(SCRIPT_DIR, "gateway_tools", f"{tool}.py")
    with open(src, "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace("\ndef handler(", "\ndef lambda_handler(")
    if code.startswith("def handler("):
        code = code.replace("def handler(", "def lambda_handler(", 1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("lambda_function.py", code)
    return buf.getvalue()


def deploy_lambdas(role_arn):
    for tool in TOOLS:
        name = f"{LAMBDA_PREFIX}-{tool.replace('_', '-')}"
        zip_bytes = package(tool)
        try:
            lam.get_function(FunctionName=name)
            log(f"  Updating: {name}")
            lam.update_function_code(FunctionName=name, ZipFile=zip_bytes)
        except lam.exceptions.ResourceNotFoundException:
            log(f"  Creating: {name}")
            lam.create_function(
                FunctionName=name, Runtime="python3.12",
                Handler="lambda_function.lambda_handler", Role=role_arn,
                Code={"ZipFile": zip_bytes}, Timeout=10, MemorySize=128)


def ensure_gateway_role():
    try:
        arn = iam.get_role(RoleName=GW_ROLE_NAME)["Role"]["Arn"]
        log(f"  Gateway role exists: {arn}")
        return arn
    except iam.exceptions.NoSuchEntityException:
        log("  Creating gateway execution role...")
        trust = {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
             "Action": "sts:AssumeRole"}]}
        arn = iam.create_role(RoleName=GW_ROLE_NAME,
                              AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        iam.put_role_policy(RoleName=GW_ROLE_NAME, PolicyName="InvokeLambdaTargets",
                            PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [
                                {"Effect": "Allow", "Action": "lambda:InvokeFunction",
                                 "Resource": "arn:aws:lambda:*:*:function:afwc-gateway-tool-*"}]}))
        log("  Waiting 10s for role propagation...")
        time.sleep(10)
        return arn


def main():
    log("=== Step 1/3: Lambda role ===")
    lambda_role = ensure_lambda_role()
    log("=== Step 2/3: Lambda functions ===")
    deploy_lambdas(lambda_role)
    log("=== Step 3/3: Gateway role + MCP Gateway ===")
    gw_role = ensure_gateway_role()

    # Hand off to manage_gateway.py for the gateway + target registration
    os.environ["GATEWAY_ROLE_ARN"] = gw_role
    os.environ["LAMBDA_PREFIX"] = LAMBDA_PREFIX
    os.environ["AWS_ACCOUNT_ID"] = ACCOUNT_ID
    sys.argv = ["manage_gateway.py"]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "manage_gateway", os.path.join(SCRIPT_DIR, "manage_gateway.py"))
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)  # prints GATEWAY_ID / GATEWAY_URL to stdout


if __name__ == "__main__":
    main()
