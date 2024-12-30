"""
Sanity Check Module

This module contains all logic for verifying a spike in "bad" tickets via GPT-based analysis.
We extract this logic from pipeline.py to keep pipeline orchestration separate
from GPT-based sanity checks.

Usage:
  from sanity_check import run_sanity_check
  
  decision = run_sanity_check(cfg, recent_bad_samples)
  if decision == "stop":
      # shut down pipeline
  else:
      # resume normal processing
"""

import logging
import time
import json as pyjson
import openai
from typing import List, Dict, Any

# Dedicated logger for sanity checks
sanity_logger = logging.getLogger("sanity_check")

def run_sanity_check(
    cfg,
    recent_bad_samples: List[Dict[str, Any]]
) -> str:
    """
    Uses GPT to confirm whether recently flagged 'bad' tickets are truly bad
    or if the pipeline might be misclassifying them.

    Steps:
      1. We receive a list of recently bad tickets (as dicts with "raw_ticket" and "error").
      2. Construct a system + user prompt explaining the situation.
      3. Call GPT, expecting a JSON response with:
         { "decision": "continue" or "stop", "explanation": "..." }
      4. Return "stop" or "continue" based on GPT's response.

    If GPT fails or says "stop", we return "stop". Otherwise "continue".

    Args:
        cfg: The Config object containing the OpenAI API key.
        recent_bad_samples: Up to 5 recent bad ticket dicts for inspection.

    Returns:
        "stop" or "continue" as a string.
    """
    if not recent_bad_samples:
        sanity_logger.warning("No recent bad tickets available for sanity check; defaulting to stop.")
        return "stop"

    # We'll only send the last 5 or so
    samples_to_send = recent_bad_samples[-5:]

    system_msg = (
        "You are a system verifying if another AI is accurately tagging tickets as 'bad'. The tickets are being converted and used as training data for another LLM. "
        "Below are recently flagged 'bad' tickets. Confirm if they're truly unparseable, we have an API error, they are unfit for LLM training or if there's a pipeline error."
        "Basically, do we have a technical issue and we need to stop processing tickets or do we keep the process running because the AI and our pipeline are doing their job? "
        "Finally, return JSON exactly in this format: {\"decision\": \"continue\" or \"stop\", \"explanation\": \"...\"}"
    )

    user_msg = "Recently flagged 'bad' tickets:\n"
    for i, item in enumerate(samples_to_send, start=1):
        raw = item.get("raw_ticket", "")
        err = item.get("error", "")
        user_msg += f"\nTicket {i}:\nRaw data:\n{raw}\nError reason:\n{err}\n"
    user_msg += (
        "\nIf these are genuinely 'bad', say 'continue'. If there's a pipeline misfire, say 'stop'. "
        "Your output must be strictly valid JSON with keys 'decision' and 'explanation'."
    )

    def call_gpt_sanity_check(sys_text: str, usr_text: str) -> str:
        """
        Makes an actual GPT call using openai.ChatCompletion.
        Returns the raw JSON string from the model, or raises an Exception if anything fails.
        """
        openai.api_key = cfg.api_key
        resp = openai.ChatCompletion.create(
            model="gpt-4",   # You may change this to the model you prefer for sanity checks
            messages=[
                {"role": "system", "content": sys_text},
                {"role": "user", "content": usr_text},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return resp.choices[0].message.content

    # Attempt up to 3 times
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            sanity_logger.info("Running sanity check attempt %d with GPT-4...", attempt)
            raw_json_str = call_gpt_sanity_check(system_msg, user_msg)
            data = pyjson.loads(raw_json_str)

            decision = data.get("decision", "stop").lower()
            explanation = data.get("explanation", "")
            sanity_logger.info("GPT sanity check => decision: %s, explanation: %s", decision, explanation)

            if decision not in ("continue", "stop"):
                sanity_logger.error("GPT-4 returned invalid decision, defaulting to stop.")
                return "stop"
            return decision

        except Exception as e:
            sanity_logger.warning("Sanity check attempt %d failed: %s", attempt, e)
            if attempt == retries:
                sanity_logger.error("All sanity check attempts failed, defaulting to stop.")
                return "stop"
            time.sleep(1 + 2 ** attempt)

    # Fallback if something truly unexpected occurs
    return "stop"
