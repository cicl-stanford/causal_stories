"""Parameter grid search over noise, bandwidth, p, q, beta, r."""
import os
import argparse
import json
import time
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from run_paths import (
    DATA_DIR,
    KDE_DIR,
    RUN_DIR,
    FIGURES_DIR as RUN_FIGURES_DIR,
    print_progress,
)

from step2_bayesian_inference import (
    calculate_posterior,
    create_probability_table,
    define_prior,
    posterior_weighted_score,
    softmax_choice_probabilities,
)
from step3_optimize_parameters import load_participant_data

RESULTS_DIR = os.path.join(RUN_DIR, "results")
BESTFIT_PATH = os.path.join(DATA_DIR, "bestfit", "best_parameters.csv")
CONFIG_PATH = os.path.join(RUN_DIR, "step1_config.json")
PROGRESS_PATH = os.path.join(RUN_DIR, "step2_progress.json")
MARGINAL_PATH = os.path.join(RUN_DIR, "step3_marginal_summary.csv")
GLOBAL_BEST_PATH = os.path.join(RUN_DIR, "step3_global_best.json")
GRID_SUMMARY_PATH = os.path.join(RUN_DIR, "grid_search_summary.json")

# =========================
# CONFIGURATION
# =========================
# Grid values
GRID_BLOCK_NOISE = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
GRID_RAMP_NOISE = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
GRID_BANDWIDTH = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
GRID_P = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
GRID_Q = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
GRID_BETA = [
    100e3, 150e3, 200e3, 250e3, 300e3, 350e3, 400e3, 450e3,
    500e3, 550e3, 600e3, 650e3, 700e3, 750e3, 800e3, 850e3]
GRID_R = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]

PARAM_NAMES = [
    "block_noise",
    "ramp_noise",
    "bandwidth",
    "p",
    "q",
    "beta",
    "r",
]

RESULT_COLUMNS = PARAM_NAMES + ["log_likelihood"]

# Likelihood evaluation
CONDITIONS = ["forward_ramp_condition", "backward_ramp_condition"]
TRIALS = ["trial_a", "trial_b", "trial_c", "trial_d"]
CHOICES = ["choice_1", "choice_2", "choice_3", "choice_4"]

# Plotting
FIGURES_DIR = os.path.join(RUN_FIGURES_DIR, "parameter_sensitivity")
PLOT_FIGSIZE = (6, 6 * 670 / 1024)
PLOT_DPI = 300
PLOT_MARKERSIZE = 5
PLOT_MARKER_EDGEWIDTH = 2.0
PLOT_BESTFIT_COLOR = "#7B1FA2"
PLOT_GRID_BEST_COLOR = "#FFD700"
PLOT_Y_N_BINS = 5
PLOT_YTICK_LENGTH = 6
PLOT_XLIM_PAD = 0.5
PLOT_AXIS_FONTSIZE = 13
PLOT_TITLE_FONTSIZE = 15
PLOT_TITLE_PAD = 12
PLOT_TICK_FONTSIZE = 10
PLOT_TICK_FONTSIZE_DENSE = 9
PAPER_PARAM_LABELS = {
    "p": r"$p$",
    "q": r"$q$",
    "r": r"$\epsilon$",
    "block_noise": r"$\sigma_{\mathrm{cube}}$",
    "ramp_noise": r"$\sigma_{\mathrm{ramp}}$",
    "bandwidth": r"$\kappa$",
    "beta": r"$\beta$",
}
PAPER_PARAM_FILENAMES = {
    "p": "p",
    "q": "q",
    "r": "ε",
    "block_noise": "σ_cube",
    "ramp_noise": "σ_ramp",
    "bandwidth": "κ",
    "beta": "β",
}
PLOT_PARAM_ORDER = [
    "block_noise",
    "ramp_noise",
    "bandwidth",
    "p",
    "q",
    "r",
    "beta",
]

def default_worker_count():
    return 6

# =========================
# MAIN GRID SEARCH FUNCTIONS
# =========================

# Reads the best fitting parameters from the best_parameters.csv file - purple dot
def load_best_fit_parameters():
    if not os.path.exists(BESTFIT_PATH):
        raise FileNotFoundError(
            f"Best-fit parameters not found: {BESTFIT_PATH}. Run step3_optimize_parameters.py first."
        )
    return pd.read_csv(BESTFIT_PATH).set_index("parameter")["value"].to_dict()

# Reads the best fitting parameters combo from the grid search - yellow dot
def load_grid_global_best():
    if not os.path.exists(GLOBAL_BEST_PATH):
        return None
    return load_json(GLOBAL_BEST_PATH)

# Builds the parameter grid for the search
def build_grids():
    return {
        "block_noise": GRID_BLOCK_NOISE,
        "ramp_noise": GRID_RAMP_NOISE,
        "bandwidth": GRID_BANDWIDTH,
        "p": GRID_P,
        "q": GRID_Q,
        "r": GRID_R,
        "beta": GRID_BETA,
    }

# Creating one gzip file for each KDE triplet (block_noise, ramp_noise, bandwidth)
def result_shard_path(block_noise, ramp_noise, bandwidth):
    name = (
        f"blk{block_noise:.3f}_rmp{ramp_noise:.3f}_bw{bandwidth:.3f}.csv.gz"
    )
    return os.path.join(RESULTS_DIR, name)

# Creates a unique key for a KDE triplet
def task_key(block_noise, ramp_noise, bandwidth):
    return f"{block_noise:.3f},{ramp_noise:.3f},{bandwidth:.3f}"

# Returns the path to the KDE file for a given triplet from step 2
def kde_path(block_noise, ramp_noise, bandwidth):
    return os.path.join(
        KDE_DIR,
        f"kde_results_blk{block_noise:.3f}_rmp{ramp_noise:.3f}_bw{bandwidth:.3f}_seed1.csv",
    )

# Returns the path to the physics simulation file for a given triplet from step 1
def physics_sim_path(block_noise, ramp_noise):
    return os.path.join(
        DATA_DIR,
        "physics_simulations",
        f"trial_results_blk{block_noise:.3f}_rmp{ramp_noise:.3f}.csv",
    )

# Returns a list of missing physics simulation scenarios
def missing_physics_tasks(grids):
    return [
        (block_noise, ramp_noise)
        for block_noise, ramp_noise in product(
            grids["block_noise"], grids["ramp_noise"]
        )
        if not os.path.exists(physics_sim_path(block_noise, ramp_noise))
    ]

# Runs physics sims for any missing block×ramp pairs (reuses existing CSVs)
def ensure_physics_sims(grids):
    missing = missing_physics_tasks(grids)
    if not missing:
        return 0

    from step1_physics_simulations import (
        DEFAULT_SEED,
        N_TRIALS,
        combine_and_save_results,
        run_parallel_simulations,
    )

    block_values = sorted({b for b, _ in missing})
    ramp_values = sorted({r for _, r in missing})
    print(f"\n=== Generating {len(missing)} missing physics sims ===")
    print(f"  Unique block_noise values to simulate: {block_values}")
    print(f"  Unique ramp_noise values to simulate: {ramp_values}")

    trial_a_cache, trial_b_cache, failed_a_cache, failed_b_cache = (
        run_parallel_simulations(block_values, ramp_values, N_TRIALS, DEFAULT_SEED)
    )
    combine_and_save_results(
        missing,
        trial_a_cache,
        trial_b_cache,
        failed_a_cache,
        failed_b_cache,
        save_results=True,
    )

    still_missing = missing_physics_tasks(grids)
    generated = len(missing) - len(still_missing)
    if still_missing:
        print(f"Warning: {len(still_missing)} physics sim files still missing")
    print(f"Physics sim generation done: {generated} created")
    return generated


def missing_kde_tasks(grids):
    return [
        task
        for task in kde_combo(grids)
        if not os.path.exists(kde_path(*task))
    ]

def ensure_kde_files(grids, kde_workers=1):
    tasks = missing_kde_tasks(grids)
    if not tasks:
        return 0

    from step2_bayesian_inference import process_grid_kdes

    print(f"\n=== {len(tasks)} KDE files missing; running step 2 ===")
    return process_grid_kdes(
        workers=max(1, kde_workers),
        skip_existing=True,
        block_noises=grids["block_noise"],
        ramp_noises=grids["ramp_noise"],
        bandwidths=grids["bandwidth"],
    )


def empty_results_frame():
    return pd.DataFrame(columns=RESULT_COLUMNS)


# Write log-likelihoods over (p, q, beta, r) for one (block_noise, ramp_noise, bandwidth) cell to a compressed shard.
def save_result_shard(df, block_noise, ramp_noise, bandwidth):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = result_shard_path(block_noise, ramp_noise, bandwidth)
    df.to_csv(path, index=False, compression="gzip")
    return path


def list_result_shard_paths():
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted(
        os.path.join(RESULTS_DIR, name)
        for name in os.listdir(RESULTS_DIR)
        if name.endswith(".csv.gz")
    )


def mean_ll_by_parameter(show_progress=True):
    """Mean log-likelihood for each value of each parameter, plus SD across the rest of the grid.
    Reads one CSV at a time. For each parameter value, averages over the
    other six parameters. Also tracks the single best combination.
    """
    shard_paths = list_result_shard_paths()
    if len(shard_paths) == 0:
        return pd.DataFrame(), None, 0

    # totals[(parameter_name, parameter_value)] holds running sum / sum of squares / count
    # across every file. Created once, before the loop; each file only adds to it.
    totals = {}
    best_combo = None
    n_rows = 0
    start = time.time()

    i = 0
    for path in shard_paths:
        i = i + 1
        df = pd.read_csv(path)
        n_rows += len(df)
        if len(df) == 0:
            continue

        # Best row in this file. Keep it if it beats the global best so far.
        best_row = df.loc[df["log_likelihood"].idxmax()]
        this_ll = float(best_row["log_likelihood"])
        if best_combo is None or this_ll > best_combo["log_likelihood"]:
            best_combo = {}
            for name in PARAM_NAMES:
                best_combo[name] = float(best_row[name])
            best_combo["log_likelihood"] = this_ll
            best_combo["kde_file"] = kde_path(
                best_combo["block_noise"],
                best_combo["ramp_noise"],
                best_combo["bandwidth"],
            )

        # Add this file's log-likelihoods into the running totals.
        for name in PARAM_NAMES:
            grouped = df.groupby(name)["log_likelihood"]
            for value, ll_values in grouped:
                key = (name, float(value))
                if key not in totals:
                    totals[key] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
                totals[key]["sum"] += ll_values.sum()
                totals[key]["sum_sq"] += (ll_values ** 2).sum()
                totals[key]["count"] += len(ll_values)

        if show_progress:
            print_progress(i, len(shard_paths), start, label="summarize")

    rows = []
    for key in totals:
        name, value = key
        n = totals[key]["count"]
        mean = totals[key]["sum"] / n
        mean_sq = totals[key]["sum_sq"] / n
        variance = mean_sq - mean ** 2
        if variance < 0:
            variance = 0.0
        rows.append(
            {
                "parameter": name,
                "value": value,
                "mean_loglikelihood": mean,
                "std_loglikelihood": np.sqrt(variance),
                "count": n,
            }
        )
    return pd.DataFrame(rows), best_combo, n_rows


def compute_weighted_results(p, q, r, kde_df):
    """Posterior-weighted KDE scores for fixed (p, q, r); independent of beta."""
    consistency_table = create_probability_table(r=r)
    prior = define_prior(p, q, show_details=False)
    posterior_forward = calculate_posterior(
        prior, "forward_ramp_condition", consistency_table, show_details=False
    )
    posterior_backward = calculate_posterior(
        prior, "backward_ramp_condition", consistency_table, show_details=False
    )
    if (
        posterior_forward is None
        or posterior_backward is None
        or any(np.isnan(list(posterior_forward.values())))
        or any(np.isnan(list(posterior_backward.values())))
    ):
        return None
    return posterior_weighted_score(
        kde_df, posterior_forward, posterior_backward, show_details=False
    )


def softmax_log_likelihood(beta, weighted_results, participant_data):
    """Softmax with beta, then Σ n * log P (incorporating human choice counts)."""
    probability_results = softmax_choice_probabilities(
        weighted_results, beta, show_details=False
    )
    total_log_likelihood = 0.0
    for condition in CONDITIONS:
        for trial in TRIALS:
            if condition not in probability_results or trial not in probability_results[condition]:
                continue
            predicted_probs = probability_results[condition][trial]
            choice_counts = participant_data[condition][trial]
            for choice in CHOICES:
                if choice in predicted_probs and choice in choice_counts:
                    total_log_likelihood += choice_counts[choice] * np.log(
                        predicted_probs[choice]
                    )
    if np.isnan(total_log_likelihood) or total_log_likelihood == float("-inf"):
        return None
    return total_log_likelihood


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def kde_combo(grids):
    return [
        (block_noise, ramp_noise, bandwidth)
        for block_noise, ramp_noise, bandwidth in product(
            grids["block_noise"], grids["ramp_noise"], grids["bandwidth"]
        )
    ]


def evaluate_kde_cell(block_noise, ramp_noise, bandwidth, grids, participant_data):
    """Score every (p, q, beta, r) on one KDE file. Returns a table for the shard."""
    path = kde_path(block_noise, ramp_noise, bandwidth)
    key = task_key(block_noise, ramp_noise, bandwidth)
    n_combos = (
        len(grids["p"]) * len(grids["q"]) * len(grids["beta"]) * len(grids["r"])
    )

    if not os.path.exists(path):
        return empty_results_frame(), None, 0, n_combos, key

    kde_df = pd.read_csv(path)
    rows = []
    n_valid = 0
    n_invalid = 0
    local_best = None

    for p, q, r in product(grids["p"], grids["q"], grids["r"]):
        weighted_results = compute_weighted_results(p, q, r, kde_df)
        if weighted_results is None:
            n_invalid += len(grids["beta"])
            continue

        for beta in grids["beta"]:
            log_likelihood = softmax_log_likelihood(
                beta, weighted_results, participant_data
            )
            if log_likelihood is None:
                n_invalid += 1
                continue

            n_valid += 1
            param_values = {
                "block_noise": block_noise,
                "ramp_noise": ramp_noise,
                "bandwidth": bandwidth,
                "p": p,
                "q": q,
                "beta": beta,
                "r": r,
            }
            rows.append({**param_values, "log_likelihood": log_likelihood})

            if local_best is None or log_likelihood > local_best["log_likelihood"]:
                local_best = {
                    **param_values,
                    "log_likelihood": log_likelihood,
                    "kde_file": path,
                }

    result_df = pd.DataFrame(rows) if rows else empty_results_frame()
    return result_df, local_best, n_valid, n_invalid, key


def step1_setup(grids, generate_kde=True, kde_workers=1):
    print("\n=== Step 1: Setup ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ensure_physics_sims(grids)

    if generate_kde:
        ensure_kde_files(grids, kde_workers=kde_workers)

    tasks = kde_combo(grids)
    missing_tasks = missing_kde_tasks(grids)
    missing = [kde_path(*task) for task in missing_tasks]

    config = {
        "metric": "log_likelihood",
        "storage": "shards",
        "timestamp": datetime.now().isoformat(),
        "grids": grids,
        "grid_sizes": {name: len(grids[name]) for name in PARAM_NAMES},
        "total_combinations": int(np.prod([len(grids[name]) for name in PARAM_NAMES])),
        "kde_combo": len(tasks),
        "model_combos_per_kde": (
            len(grids["p"]) * len(grids["q"]) * len(grids["beta"]) * len(grids["r"])
        ),
        "missing_kde_files": len(missing),
        "missing_kde_examples": missing[:5],
    }

    save_json(CONFIG_PATH, config)

    print(f"Run directory: {RUN_DIR}")
    print(f"Metric: log-likelihood")
    print(f"Total combinations: {config['total_combinations']:,}")
    print(f"KDE tasks: {config['kde_combo']}")
    print(f"Missing KDE files: {config['missing_kde_files']}")
    if missing:
        print("Warning: some KDE files are missing. Search will skip those tasks.")
    print(f"Saved {CONFIG_PATH}")
    return config


def load_step1():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Grid setup output not found: {CONFIG_PATH}. Run python step4_parameter_grid.py first."
        )
    return load_json(CONFIG_PATH)


def load_step2_progress():
    if os.path.exists(PROGRESS_PATH):
        return load_json(PROGRESS_PATH)
    return {
        "completed_tasks": [],
        "valid_evaluations": 0,
        "invalid_evaluations": 0,
        "global_best": None,
    }


def save_step2_progress(progress):
    save_json(PROGRESS_PATH, progress)


def step2_grid_search(grids, workers=1, generate_kde=True, kde_workers=1):
    print("\n=== Step 2: Grid search ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ensure_physics_sims(grids)
    if generate_kde:
        ensure_kde_files(grids, kde_workers=kde_workers)

    participant_data = load_participant_data()
    progress = load_step2_progress()
    completed = set(progress.get("completed_tasks", []))
    global_best = progress.get("global_best")

    tasks = kde_combo(grids)

    pending = []
    for block_noise, ramp_noise, bandwidth in tasks:
        key = task_key(block_noise, ramp_noise, bandwidth)
        shard_path = result_shard_path(block_noise, ramp_noise, bandwidth)
        if key in completed and os.path.exists(shard_path):
            continue
        pending.append((block_noise, ramp_noise, bandwidth))

    total_tasks = len(tasks)
    already_done = total_tasks - len(pending)

    model_per_task = (
        len(grids["p"]) * len(grids["q"]) * len(grids["beta"]) * len(grids["r"])
    )
    print(f"KDE tasks total: {total_tasks}")
    print(f"Already completed: {already_done}")
    print(f"Remaining: {len(pending)}")
    print(f"Model combos per KDE: {model_per_task}")
    print(f"Workers: {workers}")
    print(f"Result shards: {RESULTS_DIR}")

    n_valid = progress.get("valid_evaluations", 0)
    n_invalid = progress.get("invalid_evaluations", 0)
    start = time.time()

    def handle_result(result_df, local_best, valid, invalid, block_noise, ramp_noise, bandwidth, current_idx, task_label):
        nonlocal n_valid, n_invalid, global_best
        save_result_shard(result_df, block_noise, ramp_noise, bandwidth)
        n_valid += valid
        n_invalid += invalid
        if local_best is not None and (
            global_best is None
            or local_best["log_likelihood"] > global_best["log_likelihood"]
        ):
            global_best = local_best
        if task_label not in completed:
            progress["completed_tasks"].append(task_label)
        progress["valid_evaluations"] = n_valid
        progress["invalid_evaluations"] = n_invalid
        progress["global_best"] = global_best
        save_step2_progress(progress)
        extra = ""
        if global_best is not None:
            extra = f"best {global_best['log_likelihood']:.1f}"
        print_progress(
            already_done + current_idx,
            total_tasks,
            start,
            label="search",
            extra=extra,
            done_this_run=current_idx,
        )

    if workers <= 1:
        for i, (block_noise, ramp_noise, bandwidth) in enumerate(pending, 1):
            key = task_key(block_noise, ramp_noise, bandwidth)
            result_df, local_best, valid, invalid, _ = evaluate_kde_cell(
                block_noise,
                ramp_noise,
                bandwidth,
                grids,
                participant_data,
            )
            handle_result(
                result_df,
                local_best,
                valid,
                invalid,
                block_noise,
                ramp_noise,
                bandwidth,
                i,
                key,
            )
    else:
        import multiprocessing as mp

        task_payloads = [
            {
                "block_noise": b,
                "ramp_noise": r,
                "bandwidth": bw,
                "grids": grids,
                "participant_data": participant_data,
            }
            for b, r, bw in pending
        ]

        with mp.Pool(processes=workers, maxtasksperchild=16) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_worker_task, task_payloads), 1
            ):
                result_df, local_best, valid, invalid, key = result
                parts = [float(x) for x in key.split(",")]
                handle_result(
                    result_df,
                    local_best,
                    valid,
                    invalid,
                    parts[0],
                    parts[1],
                    parts[2],
                    i,
                    key,
                )

    elapsed = time.time() - start
    metadata = {
        "elapsed_seconds": elapsed,
        "metric": "log_likelihood",
        "storage": "shards",
        "hyperparam_combos": total_tasks,
        "model_combos_per_kde": model_per_task,
        "total_evaluations_attempted": total_tasks * model_per_task,
        "valid_evaluations": n_valid,
        "invalid_evaluations": n_invalid,
        "workers": workers,
        "completed_tasks": len(progress["completed_tasks"]),
        "result_shards": len(list_result_shard_paths()),
    }
    print(f"\nSaved progress: {PROGRESS_PATH}")
    return global_best, metadata


def _worker_task(task):
    return evaluate_kde_cell(
        task["block_noise"],
        task["ramp_noise"],
        task["bandwidth"],
        task["grids"],
        task["participant_data"],
    )


def step3_summarize(grids, search_metadata=None):
    print("\n=== Step 3: Summarize ===")
    marginal_df, global_best, total_rows = mean_ll_by_parameter()
    progress = load_step2_progress()
    if global_best is None:
        print("Warning: no log-likelihood data in result shards.")

    marginal_df.to_csv(MARGINAL_PATH, index=False)

    metadata = {
        **(search_metadata or {}),
        "metric": "log_likelihood",
        "storage": "shards",
        "grid_sizes": {name: len(grids[name]) for name in PARAM_NAMES},
        "timestamp": datetime.now().isoformat(),
        "completed_tasks": len(progress.get("completed_tasks", [])),
        "valid_evaluations": int(total_rows),
        "invalid_evaluations": progress.get("invalid_evaluations", 0),
        "result_rows": int(total_rows),
    }
    save_json(GRID_SUMMARY_PATH, metadata)
    if global_best is not None:
        save_json(GLOBAL_BEST_PATH, global_best)

    print(f"Saved {MARGINAL_PATH}")
    print(f"Saved {GRID_SUMMARY_PATH}")
    return marginal_df, global_best, metadata


def _add_reference_markers(ax, name, data, best_row, bestfit, grid_best, y_col="mean_loglikelihood"):
    """Red = best plotted y; yellow = grid MLE; purple = Step 3 best-fit."""
    xs = np.sort(data["value"].unique())
    x_tol = (
        float(np.min(np.diff(xs)) * 0.25)
        if len(xs) > 1
        else max(abs(float(xs[0])) * 0.01, 1e-9)
    )
    ys = data[y_col].values
    y_tol = max(float(ys.max() - ys.min()) * 0.02, 1.0)
    placed = []
    use_rings = name != "q"
    sorted_data = data.sort_values("value")

    def mean_at(x):
        return float(
            np.interp(
                x,
                sorted_data["value"].values,
                sorted_data[y_col].values,
            )
        )

    def draw(x, y, color, zorder):
        n_overlap = sum(
            1 for px, py in placed if abs(x - px) <= x_tol and abs(y - py) <= y_tol
        )
        if use_rings and n_overlap > 0:
            ax.plot(
                [x],
                [y],
                "o",
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=PLOT_MARKER_EDGEWIDTH,
                markersize=PLOT_MARKERSIZE + 1.5 + (n_overlap - 1),
                linestyle="none",
                zorder=zorder,
                clip_on=False,
            )
        else:
            ax.plot(
                [x],
                [y],
                "o",
                color=color,
                markersize=PLOT_MARKERSIZE,
                linestyle="none",
                zorder=zorder,
                clip_on=False,
            )
        placed.append((x, y))

    draw(float(best_row["value"]), float(best_row[y_col]), "red", 5)
    if grid_best and name in grid_best:
        x = float(grid_best[name])
        draw(x, mean_at(x), PLOT_GRID_BEST_COLOR, 6)
    if bestfit and name in bestfit:
        x = float(bestfit[name])
        draw(x, mean_at(x), PLOT_BESTFIT_COLOR, 7)


def _search_grid_xticks(name):
    """X ticks = every value in the search grid, in search order/step.

    Prefer the run's saved step1 config (the grid that was actually searched).
    Otherwise use the GRID_* lists in this file.
    """
    values = None
    if os.path.exists(CONFIG_PATH):
        try:
            values = (load_json(CONFIG_PATH).get("grids") or {}).get(name)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            values = None
    if not values:
        values = build_grids().get(name)
    if not values:
        return None
    return [float(v) for v in values]


def _tick_decimals(ticks):
    """Fewest decimals so each label matches the stored grid value."""
    ticks = [float(t) for t in ticks]
    if not ticks:
        return 2
    for decimals in range(2, 5):
        if all(abs(t - float(f"{t:.{decimals}f}")) < 5e-10 for t in ticks):
            return decimals
    return 4


def _format_xtick(name, value, ticks=None):
    if name == "beta":
        return f"{int(round(value / 1000))}"
    decimals = _tick_decimals(ticks) if ticks else 2
    if name == "bandwidth" and ticks is None:
        return f"{value:.2f}" if abs(value - round(value)) > 1e-9 else f"{value:.0f}"
    return f"{value:.{decimals}f}"


def _nice_tick_steps(span):
    if span <= 0:
        return [1.0]
    raw = span / max(PLOT_Y_N_BINS - 1, 1)
    exp0 = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
    steps = []
    for scale in (0.1, 1.0, 10.0):
        for m in (1.0, 2.0, 5.0):
            steps.append(m * exp0 * scale)
    return sorted(set(np.round(steps, 12)))


def _wrapped_yticks(y_lo, y_hi, n_bins=PLOT_Y_N_BINS, step=None):

    y_lo = float(y_lo)
    y_hi = float(y_hi)
    if not np.isfinite(y_lo) or not np.isfinite(y_hi):
        return y_lo, y_hi, None
    if y_hi < y_lo:
        y_lo, y_hi = y_hi, y_lo
    span = y_hi - y_lo
    if span <= 0:
        pad = max(abs(y_hi) * 0.02, 1.0)
        y_lo -= pad
        y_hi += pad
        span = y_hi - y_lo
    if step is None:
        best = None
        for candidate in _nice_tick_steps(span):
            if candidate <= 0:
                continue
            lo = np.floor(y_lo / candidate) * candidate
            hi = np.ceil(y_hi / candidate) * candidate
            if hi <= lo:
                hi = lo + candidate
            n = int(np.round((hi - lo) / candidate)) + 1
            extra = (hi - y_hi) + (y_lo - lo)
            score = (abs(n - n_bins), extra, candidate)
            if 3 <= n <= n_bins + 3 and (best is None or score < best[0]):
                best = (score, candidate, lo, hi)
        if best is None:
            step = max(_nice_tick_steps(span)[0], span / max(n_bins - 1, 1))
            lo = np.floor(y_lo / step) * step
            hi = np.ceil(y_hi / step) * step
            if hi <= lo:
                hi = lo + step
        else:
            _, step, lo, hi = best
    else:
        step = float(step)
        lo = np.floor(y_lo / step) * step
        hi = np.ceil(y_hi / step) * step
        if hi <= lo:
            hi = lo + step
    ticks = np.round(np.arange(lo, hi + 0.5 * step, step), 10)
    if ticks[0] > y_lo + 1e-9:
        ticks = np.concatenate([[ticks[0] - step], ticks])
    if ticks[-1] < y_hi - 1e-9:
        ticks = np.concatenate([ticks, [ticks[-1] + step]])
    return float(ticks[0]), float(ticks[-1]), ticks


def _shared_ylim(marginal_df, y_col="mean_loglikelihood"):
    """One y range covering every parameter, so plots are comparable."""
    y = np.asarray(marginal_df[y_col].dropna().values, dtype=float)
    if y.size == 0:
        return None, None, None
    return _wrapped_yticks(float(y.min()), float(y.max()))


def _style_plot(
    ax,
    fig,
    name,
    data,
    tight_xlim=True,
    y_col="mean_loglikelihood",
    y_lo=None,
    y_hi=None,
    yticks=None,
):
    label = PAPER_PARAM_LABELS.get(name, name.replace("_", " "))
    ax.set_xlabel(
        r"$\beta$ ($\times 10^{3}$)" if name == "beta" else label,
        fontsize=PLOT_AXIS_FONTSIZE,
    )
    ax.set_ylabel("log-likelihood", fontsize=PLOT_AXIS_FONTSIZE)
    ax.set_title(
        f"{label} vs. log-likelihood",
        fontsize=PLOT_TITLE_FONTSIZE,
        pad=PLOT_TITLE_PAD,
    )

    xticks = _search_grid_xticks(name)
    if xticks:
        # Linear x so equal grid steps have equal spacing.
        ax.set_xscale("linear")
        ax.set_xticks(xticks)
        n = len(xticks)
        rotate = n >= 18
        tick_fontsize = PLOT_TICK_FONTSIZE_DENSE if n >= 14 else PLOT_TICK_FONTSIZE
        ax.set_xticklabels(
            [_format_xtick(name, t, xticks) for t in xticks],
            fontsize=tick_fontsize,
            rotation=45 if rotate else 0,
            ha="right" if rotate else "center",
        )
        pad = (
            float(np.median(np.diff(xticks)) * PLOT_XLIM_PAD)
            if len(xticks) > 1
            else 0
        )
        ax.set_xlim(xticks[0] - pad, xticks[-1] + pad)
    elif tight_xlim:
        x = np.asarray(data["value"].values, dtype=float)
        if x.size:
            x_unique = np.sort(np.unique(x))
            pad = (
                float(np.median(np.diff(x_unique)) * PLOT_XLIM_PAD)
                if len(x_unique) > 1
                else max(abs(float(x_unique[0])) * 0.05, 0.01)
            )
            ax.set_xlim(float(np.min(x) - pad), float(np.max(x) + pad))

    if y_lo is None or y_hi is None:
        y = np.asarray(data[y_col].values, dtype=float)
        y_lo, y_hi, yticks = _wrapped_yticks(float(y.min()), float(y.max()))
    ax.set_ylim(y_lo, y_hi)
    ax.minorticks_off()
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        for label in ax.get_yticklabels():
            label.set_clip_on(False)
            label.set_fontsize(PLOT_TICK_FONTSIZE)

    for spine in ax.spines.values():
        spine.set_capstyle("butt")
        spine.set_joinstyle("miter")
    x_left, x_right = ax.get_xlim()
    ax.spines["top"].set_bounds(x_left, x_right)
    ax.spines["bottom"].set_bounds(x_left, x_right)
    ax.spines["left"].set_bounds(y_lo, y_hi)
    ax.spines["right"].set_bounds(y_lo, y_hi)
    # Inward ticks so the corner marks are not extra leftward stubs on the frame.
    ax.tick_params(axis="y", length=PLOT_YTICK_LENGTH, direction="in")
    fig.tight_layout()


def plot_marginals(
    marginal_df,
    output_dir,
    bestfit=None,
    grid_best=None,
    y_col="mean_loglikelihood",
):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
    y_lo, y_hi, yticks = _shared_ylim(marginal_df, y_col)

    for name in PLOT_PARAM_ORDER:
        data = (
            marginal_df[marginal_df["parameter"] == name]
            .dropna(subset=[y_col])
            .sort_values("value")
        )
        if data.empty:
            print(f"Warning: no data to plot for {name}")
            continue

        best_row = data.loc[data[y_col].idxmax()]
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.plot(
            data["value"].values,
            data[y_col].values,
            "o",
            color="black",
            markersize=PLOT_MARKERSIZE,
            linestyle="none",
            clip_on=False,
        )

        _add_reference_markers(
            ax, name, data, best_row, bestfit, grid_best, y_col=y_col
        )
        _style_plot(
            ax,
            fig,
            name,
            data,
            y_col=y_col,
            y_lo=y_lo,
            y_hi=y_hi,
            yticks=yticks,
        )

        symbol = PAPER_PARAM_FILENAMES.get(name, name)
        out_path = os.path.join(output_dir, f"{symbol}_vs_log-likelihood.png")
        fig.set_size_inches(*PLOT_FIGSIZE, forward=True)
        fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        plot_paths.append(out_path)
        print(f"Saved {out_path}")

    return plot_paths


def step4_plot(figures_dir=None):
    print("\n=== Step 4: Plot ===")
    if figures_dir is None:
        figures_dir = FIGURES_DIR
    if not os.path.exists(MARGINAL_PATH):
        raise FileNotFoundError(
            f"Step 3 output not found: {MARGINAL_PATH}. Run with --step 3 first."
        )
    marginal_df = pd.read_csv(MARGINAL_PATH)
    bestfit = load_best_fit_parameters()
    grid_best = load_grid_global_best()
    print("\n=== Step 4.4: Mean plots (shared y-axis) ===")
    plot_marginals(
        marginal_df, figures_dir, bestfit=bestfit, grid_best=grid_best
    )
    return figures_dir


def compare_with_step3(global_best):
    if not os.path.exists(BESTFIT_PATH):
        print("Step 3 best_parameters.csv not found; skipping comparison.")
        return

    if global_best is None:
        print("\n=== Comparison with Step 3 optimizer ===")
        print("  No grid global best saved.")
        return

    step3 = pd.read_csv(BESTFIT_PATH).set_index("parameter")["value"].to_dict()
    print("\n=== Comparison with Step 3 optimizer ===")
    for key in PARAM_NAMES:
        grid_val = global_best.get(key)
        step3_val = step3.get(key)
        if grid_val is not None and step3_val is not None:
            print(f"  {key:12s}  grid best: {grid_val!s:>12}   step3: {step3_val!s:>12}")
    print(f"  Grid global best log-likelihood: {global_best['log_likelihood']:.6f}")


def print_step_status():
    print("\n=== Step 4 status ===")
    for label, path in [
        ("setup", CONFIG_PATH),
        ("search progress", PROGRESS_PATH),
        ("marginal summary", MARGINAL_PATH),
        ("global best", GLOBAL_BEST_PATH),
    ]:
        status = "done" if os.path.exists(path) else "missing"
        print(f"  Step {label:20s}: {status}  ({path})")
    if os.path.isdir(RESULTS_DIR):
        shard_count = len(
            [name for name in os.listdir(RESULTS_DIR) if name.endswith(".csv.gz")]
        )
        print(f"  Result shards         : {shard_count}  ({RESULTS_DIR})")


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: 7-parameter grid search and log-likelihood plots. Reuses physics/KDE from steps 1–2; writes only grid_search outputs."
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run a single sub-step only",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run this sub-step and all later sub-steps",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for step 4.2 grid search (default: 6)",
    )
    parser.add_argument(
        "--kde-workers",
        type=int,
        default=None,
        help="Parallel workers for KDE generation in step 4.1 (default: same as --workers)",
    )
    parser.add_argument(
        "--no-generate-kde",
        action="store_true",
        help="Do not generate missing physics/KDE files (default: reuse existing, generate any that are missing)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show which sub-step outputs exist",
    )
    args = parser.parse_args()

    if args.status:
        print_step_status()
        return

    if args.step == 4 or args.from_step == 4:
        step4_plot(figures_dir=FIGURES_DIR)
        return

    grids = build_grids()

    steps_to_run = []
    if args.step is not None:
        steps_to_run = [args.step]
    elif args.from_step is not None:
        steps_to_run = list(range(args.from_step, 5))
    else:
        steps_to_run = [1, 2, 3, 4]

    search_metadata = None
    global_best = None

    generate_kde = not args.no_generate_kde
    workers = args.workers if args.workers is not None else default_worker_count()
    kde_workers = args.kde_workers if args.kde_workers is not None else workers

    if 1 in steps_to_run:
        step1_setup(grids, generate_kde=generate_kde, kde_workers=kde_workers)
    elif any(s in steps_to_run for s in (2, 3, 4)):
        config = load_step1()
        grids = config["grids"]

    if 2 in steps_to_run:
        global_best, search_metadata = step2_grid_search(
            grids,
            workers=max(1, workers),
            generate_kde=generate_kde,
            kde_workers=kde_workers,
        )

    if 3 in steps_to_run:
        _, global_best, search_metadata = step3_summarize(grids, search_metadata)

    if 4 in steps_to_run:
        step4_plot(figures_dir=FIGURES_DIR)

    if global_best:
        compare_with_step3(global_best)

    if search_metadata:
        print("\n=== Run summary ===")
        print(f"Valid evaluations: {search_metadata.get('valid_evaluations', 0):,}")
        print(f"Invalid evaluations: {search_metadata.get('invalid_evaluations', 0):,}")
        if "elapsed_seconds" in search_metadata:
            print(f"Step 2 elapsed: {search_metadata['elapsed_seconds']:.1f}s")

    print_step_status()


if __name__ == "__main__":
    main()
