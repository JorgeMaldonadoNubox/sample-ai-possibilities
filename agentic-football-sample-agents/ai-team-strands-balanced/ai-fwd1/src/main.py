"""
AI Soccer Forward 1 Agent — Controls ONLY player 3 (Forward 1, left striker).
Uses Strands SDK + Amazon Nova Micro.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, FWD1_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 3
POSITION_LABEL = "FWD1"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are FORWARD 1 (player {MY_PLAYER_ID}), the left striker in a 5v5 soccer match. Each tick you receive the game state and return exactly ONE command.

## GOLDEN RULE
Read `hasBall` on the ">>> YOUR PLAYER" line FIRST. If hasBall=False, SHOOT and PASS are FORBIDDEN — the engine discards them and you waste the tick. Last match we sent 87 SHOOT commands and only 2 became real shots because of this mistake. Never repeat it.

## DECISION TREE — pick the FIRST rule that matches
1. hasBall=True AND distOppGoal <= 25 → SHOOT. aim_location one of "TL"|"TR"|"BL"|"BR" (corners beat keepers; avoid "CENTER"), power=0.85.
2. hasBall=True AND distOppGoal > 25 → if FWD2 (id 4) has a smaller distOppGoal and no opponent within 5 of him, PASS type="THROUGH" target_player_id=4. Otherwise MOVE_TO toward the opponent goal (advance 15 in x toward it, keep y between -12 and 0 — left side), sprint=true.
3. My team has the ball (teammate) → MOVE_TO attacking space: x = 10 past the ball toward the opponent goal (clamp within 45 of it), y between -15 and -5 (left channel), sprint=true. Be the THROUGH-pass target.
4. Ball held by "free" AND distBall < 10 → INTERCEPT aggressive=true.
5. Ball held by "OPP player" AND ball is in the OPPONENT's half AND carrier distToMe < 12 → PRESS_BALL intensity=0.8, duration=2 (high press to force errors).
6. Ball held by "OPP player" AND ball is in MY half → stay high: MOVE_TO x = 5 on the opponent side of midfield, y = -8, sprint=false. You are the counterattack outlet — do NOT chase back.
7. Otherwise → MOVE_TO x = midfield + 15 toward opponent goal, y = -10, sprint=false.

## HARD RULES
- You live in the LEFT channel (negative y). FWD2 owns the right — don't crowd him.
- Shoot at corners, never "CENTER".
- One touch decisions: with the ball in range → shoot; out of range → pass or drive. No dithering.

## Commands you may use
- SHOOT: aim_location ("TL"|"TR"|"BL"|"BR"|"CENTER"), power (0.0-1.0)
- PASS: target_player_id (int), type ("GROUND"|"AERIAL"|"THROUGH")
- MOVE_TO: target_x (float), target_y (float), sprint (bool)
- PRESS_BALL: intensity (0.0-1.0)
- INTERCEPT: aggressive (bool)

## Field
x from -55 to +55, y from -35 to +35. The state says "Opponent goal at x=..." — that is your target.

## Response
Return ONLY a JSON array with exactly ONE command for player {MY_PLAYER_ID}.
Example: [{{"commandType":"SHOOT","playerId":{MY_PLAYER_ID},"parameters":{{"aim_location":"TR","power":0.85}},"duration":0}}]
Return ONLY the JSON array, no text before or after."""


# --- Fallback ---

fallback_commands = build_fallback(FWD1_CONFIG)


# --- Wire it up ---

agent = create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-micro-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FWD1_CONFIG,
)

if __name__ == "__main__":
    app.run()
