"""
Thin wrapper around Google Gemini 2.5 Flash.

Environment:
    GEMINI_API_KEY    – required
"""

import os, json
import google.generativeai as genai
from google.generativeai import types

genai.configure(api_key="AIzaSyBG3_kufdSrTgT2JN95dzo2t4kIdh33bjI")
MODEL_NAME = "gemini-2.0-flash"

# single global model instance is fine
_model = genai.GenerativeModel(MODEL_NAME)


def _call(prompt: str, system: str | None = None) -> str:
    model = _model
    if system:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=system)
    
    resp = model.generate_content(prompt)
    return resp.text.strip()


# -------------------------------------------------------------------------
def gen_initial_prompts(demo_data: str, output_desc: str) -> tuple[str, str]:
    """Ask Gemini to craft two candidate prompts."""
    prompt = f"""You are a prompt‑writer for language‑model agents.

        Demo data (triple back‑ticks):

        {demo_data}

        Desired output (rough human description):

        {output_desc}

        Return **exactly** two standalone prompts suitable for Gemini 2.5 Flash,
        numbered (A) and (B).  Each should give the agent clear instructions,
        tone, required format, etc. Do not provide examples. Do NOT write anything except the two prompts."""

    raw = _call(prompt)
    # very light parsing – split at (A)/(B)
    parts = [p.strip() for p in raw.split("(B)") if p.strip()]
    if len(parts) != 2:
        # fallback: treat whole response as one prompt duplicated
        return raw.strip(), raw.strip()
    prompt_a = parts[0].removeprefix("(A)").strip()
    prompt_b = parts[1].strip()
    return prompt_a, prompt_b

# -------------------------------------------------------------------------
def run_prompt_on_data(prompt: str, demo_data: str) -> str:
    """Run candidate prompt against demo data and return agent output."""
    system_msg = f"You are an LLM agent.  Use this prompt:\n{prompt}"
    user_msg = f"Here is the demo data you must operate on:\n\n{demo_data}"
    return _call(user_msg, system=system_msg)


def _apply_patch(prompt: str, patch: str) -> str:
    """Apply text replacements from the mutator patch. If the patch provides
    pairs of 'Original:' and 'Revised:' lines, replace occurrences of each
    original line with its revision. Otherwise, treat the patch as text to
    append to the prompt."""
    new_prompt = prompt
    replacements = []
    orig = None
    for line in patch.splitlines():
        line = line.strip()
        if line.startswith('Original:'):
            orig = line[len('Original:'):].strip()
        elif line.startswith('Revised:') and orig is not None:
            revised = line[len('Revised:'):].strip()
            replacements.append((orig, revised))
            orig = None
    if replacements:
        for o, r in replacements:
            new_prompt = new_prompt.replace(o, r)
        return new_prompt
    if patch:
        return f"{prompt}\n{patch if patch.startswith('\n') else '\n' + patch}"
    return prompt

# -------------------------------------------------------------------------
def mutate_prompt(prompt: str) -> str:
    """Ask Gemini to create a slight mutation of the prompt, returning the
    *new whole prompt* (original+delta merged) so we have a fresh standalone."""
    system = "You are the Mutator Agent.  You receive a winning prompt."
    mutation_spec = """
Provide a *JSON* object with a single field "patch" that either
 (a) rewrites specific lines, OR
 (b) appends clarifying instructions.

Keep changes subtle – adjust tone, add an example, change list style, etc.
"""
    resp = _call(f"{mutation_spec}\n\nOriginal prompt:\n{prompt}", system=system)
    try:
        patch_obj = json.loads(resp)
        patch = patch_obj["patch"].strip()
        return _apply_patch(prompt, patch)
    except Exception:
        return _apply_patch(prompt, resp.strip())
