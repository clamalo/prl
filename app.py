import os
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, abort
)
from gemini import (
    gen_initial_prompts,
    run_prompt_on_data,
    mutate_prompt
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev‑secret‑change‑me")

# === In‑memory experiment store (for demo / single‑process dev) ============
EXPERIMENTS = {}   # experiment_id -> dict with history / state


# ---------------------------------------------------------------------------
# 1.   START PAGE
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


# ---------------------------------------------------------------------------
# 2.   GENERATE FIRST TWO PROMPTS + THEIR OUTPUTS
# ---------------------------------------------------------------------------
@app.post("/generate")
def generate() -> str:
    demo_data = request.form.get("demo_data", "")
    output_desc = request.form.get("output_desc", "")
    if not demo_data or not output_desc:
        abort(400, "Both text areas are required.")

    # call Gemini to craft two seed prompts
    prompt_a, prompt_b = gen_initial_prompts(demo_data, output_desc)

    # produce outputs for both candidates
    out_a = run_prompt_on_data(prompt_a, demo_data)
    out_b = run_prompt_on_data(prompt_b, demo_data)

    # persist an experiment session
    exp_id = str(uuid.uuid4())
    EXPERIMENTS[exp_id] = {
        "demo_data": demo_data,
        "history": [
            {
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "out_a": out_a,
                "out_b": out_b
            }
        ]
    }
    return redirect(url_for("evaluate", exp_id=exp_id))


# ---------------------------------------------------------------------------
# 3.   SHOW SIDE‑BY‑SIDE EVALUATION
# ---------------------------------------------------------------------------
@app.get("/evaluate/<exp_id>")
def evaluate(exp_id: str) -> str:
    exp = EXPERIMENTS.get(exp_id)
    if exp is None:
        abort(404)
    last = exp["history"][-1]
    return render_template(
        "evaluate.html",
        exp_id=exp_id,
        iteration=len(exp["history"]),
        out_left=last["out_a"],
        out_right=last["out_b"]
    )


# ---------------------------------------------------------------------------
# 4.   RECORD USER VOTE, MUTATE, RUN NEXT MATCHUP
# ---------------------------------------------------------------------------
@app.post("/vote")
def vote():
    exp_id = request.json.get("exp_id")
    winner = request.json.get("winner")     # "left" or "right"
    exp = EXPERIMENTS.get(exp_id)
    if exp is None or winner not in {"left", "right"}:
        abort(400)

    last_round = exp["history"][-1]
    winning_prompt = (
        last_round["prompt_a"] if winner == "left" else last_round["prompt_b"]
    )

    # create slight mutation of the winning prompt
    mutated_prompt = mutate_prompt(winning_prompt)

    # run both prompts on same demo data
    demo_data = exp["demo_data"]
    out_orig = run_prompt_on_data(winning_prompt, demo_data)
    out_mut  = run_prompt_on_data(mutated_prompt, demo_data)

    # store new round (winner always stays as prompt_a to keep “left” / “right” meaningless)
    exp["history"].append({
        "prompt_a": winning_prompt,
        "prompt_b": mutated_prompt,
        "out_a": out_orig,
        "out_b": out_mut
    })

    return jsonify({
        "iteration": len(exp["history"]),
        "left":  out_orig,
        "right": out_mut
    })


# ---------------------------------------------------------------------------
# 5.   GET CURRENT PROMPTS FOR VIEWING
# ---------------------------------------------------------------------------
@app.get("/prompts/<exp_id>")
def get_prompts(exp_id: str):
    exp = EXPERIMENTS.get(exp_id)
    if exp is None:
        abort(404)
    last = exp["history"][-1]
    return jsonify({
        "left_prompt": last["prompt_a"],
        "right_prompt": last["prompt_b"]
    })


if __name__ == "__main__":   # optional when running via `python app.py`
    app.run(debug=True)