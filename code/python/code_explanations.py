"""
Qualitative coding of free-text explanations in explanations.csv.

Source data: `generalization_check_survey` trials from experiment 3, where
participants explain why they predicted a cube would land further on one
side. Each explanation is coded on two independent layers:

1. Feature layer — which stimulus property the explanation cites.
   Ramps are always yellow/blue (or white/grey for the untrained
   generalization case); cubes are always red/black (or white/grey for the
   untrained generalization case). Because color vocabulary maps cleanly
   onto feature type in this stimulus set, color words are used as the
   detector for "is this participant reasoning about the ramp or the cube."

2. Reasoning-strategy layer — orthogonal to the feature layer. A response
   can name a color feature *and* invoke a reasoning strategy (e.g. "the
   yellow ramp switched sides, so now it's on the left" is both
   ramp_color and orientation_flip). Strategies:
     - orientation_flip:  explicit reasoning that the ramp's training
                          orientation reversed, so the outcome should too
     - guess:             guessing / uncertainty language
     - motion_mechanism:  naive physics (gravity, momentum, friction,
                          weight, slide/roll/fall, steepness, height)
     - outcome_line:      cites "the finish line" / "crossing the line"
                          as evidence, without naming a color
     - memory_generic:    "based on previous trials", without specifying
                          which feature
     - repeat_prior:      "same as before" / "same reason"


Run: `python3 code_explanations.py` from this directory. Writes 
`explanations_coded.csv` — the original rows
from explanations.csv with the coding columns (ramp_color, cube_color,
feature, orientation_flip, guess, motion_mechanism, outcome_line,
repeat_prior, memory_generic) appended.
"""

import csv
import re
from collections import Counter, defaultdict

PATH = "../R/data/explanations.csv"

# --- Feature (color-based) detectors ---------------------------------------
RAMP_COLOR = re.compile(r'\b(yellow|blue|green)\b', re.I)  # 'green' = observed misremembering of ramp color
CUBE_COLOR = re.compile(r'\b(red|black)\b', re.I)
WHITE_RAMP = re.compile(r'white\s+(ramp|triangle|wedge|slide|platform|slope|base|lamp|ramps|triangles)', re.I)
WHITE_CUBE = re.compile(r'white\s+(cube|block|square|cubes|blocks|squares)', re.I)
GREY_RAMP = re.compile(r'gr[ae]y\s+(ramp|triangle|wedge|slide|platform|slope|base|ramps|triangles)', re.I)
GREY_CUBE = re.compile(r'gr[ae]y\s+(cube|block|square|cubes|blocks|squares)', re.I)

# --- Reasoning-strategy detectors -------------------------------------------
ORIENTATION_FLIP = re.compile(
    r'\b(opposite direction|different direction|reversed?|switched (sides|direction)|'
    r'facing (the )?(other|opposite|different|left|right)|flipped|changed direction|'
    r'direction (changed|switched|reversed)|the other way|position(s)? (got |were )?switched|'
    r'reverse direction|orientation)\b', re.I)

GUESS = re.compile(
    r'\b(guess(ed|ing)?|not sure|no idea|no clue|random|hunch|gut( feeling)?|intuition|instinct(s)?|'
    r'shot in the dark|dont know|don\'t know|unsure|took a guess|best guess|just felt|felt like|'
    r'trust(ed)? my|idk)\b', re.I)

MOTION_MECHANISM = re.compile(
    r'\b(momentum|gravity|friction|force|weight|heavier|lighter|mass|speed|velocity|incline|angle|'
    r'steep|slope|acceleration|collide[ds]?|collision|slide[ds]?|slid|roll(ed|ing)?|fall(ing)?|fell|'
    r'drop(ped)?|high up|elevated|higher up|top of the|launch(ed|es)?)\b', re.I)

OUTCOME_LINE = re.compile(r'\b(finish line|the line|cross(ed|ing)? the line|past the line|near the line)\b', re.I)

REPEAT_PRIOR = re.compile(r'\bsame (as|reason|logic|thing)\b', re.I)

MEMORY_GENERIC = re.compile(
    r'\b(previous (trial|trials|images|example|examples|encounters|results|predictions)|'
    r'past (trial|trials|results|predictions|encounters)|earlier (trial|trials|scenarios)|'
    r'based on (the )?(other|previous|past)|in the trials|last time|prior (trial|trials)|'
    r'based on (past|previous) (results|predictions))\b', re.I)

def classify(resp):
    ramp_color = bool(RAMP_COLOR.search(resp)) or bool(WHITE_RAMP.search(resp)) or bool(GREY_RAMP.search(resp))
    cube_color = bool(CUBE_COLOR.search(resp)) or bool(WHITE_CUBE.search(resp)) or bool(GREY_CUBE.search(resp))

    tags = {
        "ramp_color": ramp_color,
        "cube_color": cube_color,
        "orientation_flip": bool(ORIENTATION_FLIP.search(resp)),
        "guess": bool(GUESS.search(resp)),
        "motion_mechanism": bool(MOTION_MECHANISM.search(resp)),
        "outcome_line": bool(OUTCOME_LINE.search(resp)),
        "repeat_prior": bool(REPEAT_PRIOR.search(resp)),
        "memory_generic": bool(MEMORY_GENERIC.search(resp)),
    }

    if ramp_color and cube_color:
        feature = "both"
    elif ramp_color:
        feature = "ramp_feature"
    elif cube_color:
        feature = "cube_feature"
    else:
        feature = "neither"
    tags["feature"] = feature

    return tags


def load_rows(path=PATH):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            resp = (r.get('response') or '').strip()
            if not resp:
                continue
            r.update(classify(resp))
            rows.append(r)
    return rows


CODE_COLUMNS = [
    "ramp_color", "cube_color", "feature", "orientation_flip", "guess",
    "motion_mechanism", "outcome_line", "repeat_prior", "memory_generic"
]


def write_coded_csv(in_path=PATH, out_path="../R/data/explanations_coded.csv"):
    """Write explanations.csv back out with the coding columns appended,
    preserving every original row (including any with an empty response,
    which get blank codes) so row order and count match the source file."""
    with open(in_path, newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + CODE_COLUMNS
        out_rows = []
        for r in reader:
            resp = (r.get('response') or '').strip()
            if resp:
                r.update(classify(resp))
            else:
                r.update({col: "" for col in CODE_COLUMNS})
            out_rows.append(r)

    with open(out_path, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_path

if __name__ == "__main__":
    out_path = write_coded_csv()
    print(f"\nWrote coded CSV to {out_path}")
