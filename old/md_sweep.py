"""
Parallel composition sweep for MD conductivity using MACE-MP-0.

Generates conductivity data across the electrolyte design space by running
independent MD simulations for each composition. Results are saved in a
format compatible with the XGBoost training pipeline.

Usage:
    python -m conductivity.md_sweep                   # default sweep grid
    python -m conductivity.md_sweep --recipes recipes.json  # custom recipes
    python -m conductivity.md_sweep --quick            # fast test (tiny box, short MD)

Output:
    conductivity/md_runs/sweep_results.json  — all results
    conductivity/md_runs/md_training_data.json — format matching electrolyte_property_db
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed



logger = logging.getLogger(__name__)


def generate_sweep_grid() -> list[dict]:
    """
    Generate a grid of electrolyte recipes covering the design space.

    Covers:
        - EC/DMC ratios (0.2-0.5 EC in 0.1 steps)
        - EC/EMC ratios (same range)
        - LiPF6 concentration (0.8, 1.0, 1.2 M)
        - LiFSI concentration (0.8, 1.0, 1.2 M)
        - FEC additive (0, 2, 5 wt%)
    """
    recipes = []

    # Binary solvent systems — sweep grid covers typical Li-ion electrolyte space
    # EC vol fracs: 0.2-0.5 spans low-ε fast-transport to high-ε high-dissociation
    ec_fractions = [0.2, 0.3, 0.4, 0.5]  # vol fraction of EC in binary blend
    salt_configs = [
        {"LiPF6": 0.8}, {"LiPF6": 1.0}, {"LiPF6": 1.2},  # mol/L: dilute, typical, concentrated
        {"LiFSI": 0.8}, {"LiFSI": 1.0}, {"LiFSI": 1.2},  # mol/L: same range for LiFSI
    ]
    cosolvents = ["DMC", "EMC"]
    fec_loadings = [0.0, 0.02, 0.05]  # wt frac: none, 2% typical, 5% aggressive SEI former

    for ec_frac in ec_fractions:
        for cosolvent in cosolvents:
            for salt in salt_configs:
                for fec in fec_loadings:
                    recipe = {
                        "solvents": {"EC": ec_frac, cosolvent: 1.0 - ec_frac},
                        "salts": dict(salt),
                    }
                    if fec > 0:
                        recipe["additives"] = {"FEC": fec}
                    recipes.append(recipe)

    # Ternary: EC/DMC/EMC — equal cosolvent split at mid-range EC
    for salt in [{"LiPF6": 1.0}, {"LiFSI": 1.0}]:
        recipes.append({
            "solvents": {"EC": 0.3, "DMC": 0.35, "EMC": 0.35},  # vol fracs summing to 1.0
            "salts": dict(salt),
        })

    # Dual salt — 70/30 LiPF6/LiFSI blend (common dual-salt ratio in literature)
    recipes.append({
        "solvents": {"EC": 0.3, "DMC": 0.7},  # vol fracs summing to 1.0
        "salts": {"LiPF6": 0.7, "LiFSI": 0.3},  # mol/L per salt
    })

    logger.info(f"Generated {len(recipes)} sweep recipes")
    return recipes


def _run_single(args: tuple) -> dict:
    """Run a single MD conductivity calculation. Designed for ProcessPoolExecutor."""
    from conductivity.md_conductivity import MDConfig, run_md_conductivity

    recipe, config_dict, idx, total = args

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [worker-{idx}] %(message)s",
    )
    worker_logger = logging.getLogger(f"md_sweep.worker.{idx}")

    config = MDConfig(**config_dict)
    worker_logger.info(f"Starting recipe {idx+1}/{total}: {json.dumps(recipe, sort_keys=True)}")

    t0 = time.time()
    try:
        result = run_md_conductivity(recipe, config)
        elapsed = time.time() - t0
        worker_logger.info(
            f"Recipe {idx+1}/{total} complete: "
            f"σ={result.conductivity_mS_cm:.3f} mS/cm, "
            f"ρ={result.density_g_ml:.4f} g/mL, "
            f"time={elapsed/3600:.2f}h"
        )
        return {
            "recipe": result.recipe,
            "properties": {"conductivity_mS_cm": result.conductivity_mS_cm},
            "md_metadata": {
                "conductivity_std_mS_cm": result.conductivity_std_mS_cm,
                "density_g_ml": result.density_g_ml,
                "temperature_k": result.temperature_k,
                "haven_ratio": result.haven_ratio,
                "n_frames": result.n_frames,
                "wall_time_s": result.wall_time_s,
                "msd_slope_ang2_ps": result.msd_slope_ang2_ps,
            },
        }
    except Exception as e:
        worker_logger.error(f"Recipe {idx+1}/{total} FAILED: {e}")
        raise


def run_sweep(
    recipes: list[dict],
    config: dict,
    n_workers: int = 1,
    output_dir: str = "conductivity/md_runs",
) -> list[dict]:
    """
    Run MD conductivity for a list of recipes.

    Args:
        recipes: list of recipe dicts
        config: MDConfig fields as dict (overrides defaults)
        n_workers: number of parallel workers (1 = sequential)
        output_dir: where to save results

    Returns:
        List of result dicts in training data format
    """
    os.makedirs(output_dir, exist_ok=True)

    config_dict = config

    logger.info(f"Starting sweep: {len(recipes)} recipes, {n_workers} workers")
    logger.info(f"  Config: {json.dumps(config_dict, default=str)}")
    logger.info(f"  Output: {output_dir}")

    # Build args for each worker
    args_list = [
        (recipe, config_dict, idx, len(recipes))
        for idx, recipe in enumerate(recipes)
    ]

    results = []
    failures = []
    t_start = time.time()
    sweep_path = os.path.join(output_dir, "sweep_results.json")
    training_path = os.path.join(output_dir, "md_training_data.json")

    def _save_incremental():
        """Save results after each completion so multi-day sweeps don't lose progress."""
        with open(sweep_path, "w") as f_out:
            json.dump(results, f_out, indent=2)
        training_data = [{"recipe": r["recipe"], "properties": r["properties"]} for r in results]
        with open(training_path, "w") as f_out:
            json.dump(training_data, f_out, indent=2)

    if n_workers <= 1:
        # Sequential execution (simpler debugging, avoids MPS contention)
        for args in args_list:
            try:
                results.append(_run_single(args))
                _save_incremental()
            except Exception as e:
                failures.append({"recipe": args[0], "error": str(e)})
                logger.error(f"Recipe {args[2]+1}/{args[3]} failed: {e}")
    else:
        # Parallel execution with fault tolerance — each worker gets its own
        # MACE instance. submit + as_completed so one failure doesn't kill the
        # entire sweep (pool.map propagates the first exception immediately).
        from utils.worker_guard import init_worker_guard

        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker_guard) as pool:
            future_to_idx = {
                pool.submit(_run_single, args): args[2]
                for args in args_list
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results.append(future.result())
                    _save_incremental()
                except Exception as e:
                    failures.append({"recipe": args_list[idx][0], "error": str(e)})
                    logger.error(f"Recipe {idx+1}/{len(recipes)} failed: {e}")

    elapsed = time.time() - t_start
    logger.info(f"Sweep complete: {len(results)}/{len(recipes)} succeeded, "
                f"{len(failures)} failed, {elapsed/3600:.2f}h")
    if failures:
        fail_path = os.path.join(output_dir, "sweep_failures.json")
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=2)
        logger.info(f"Failure log saved to {fail_path}")

    # Final save
    _save_incremental()
    logger.info(f"Full results saved to {sweep_path}")
    logger.info(f"Training data ({len(results)} recipes) saved to {training_path}")

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="MD conductivity sweep")
    parser.add_argument("--recipes", type=str, help="JSON file with recipe list")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (tiny box, short MD)")
    parser.add_argument("--n-workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--n-molecules", type=int, default=256, help="Molecules per box")  # 256 = good stats/speed tradeoff for MACE
    parser.add_argument("--prod-steps", type=int, default=500_000, help="Production MD steps")
    parser.add_argument("--device", type=str, default="mps", help="Compute device")
    parser.add_argument("--output-dir", type=str, default="conductivity/md_runs")
    args = parser.parse_args()

    # Load or generate recipes
    if args.recipes:
        with open(args.recipes) as f:
            recipes = json.load(f)
        logger.info(f"Loaded {len(recipes)} recipes from {args.recipes}")
    else:
        recipes = generate_sweep_grid()

    # Build config
    if args.quick:
        config = {
            "n_molecules": 64,       # small box for quick validation
            "equil_nvt_steps": 1_000,
            "equil_npt_steps": 2_000,
            "prod_steps": 10_000,    # 10 ps — only for testing, not production
            "save_interval": 10,     # save every 10 fs in quick mode for enough frames
            "device": args.device,
        }
        logger.info("QUICK TEST MODE: tiny box, short MD — results NOT production quality")
    else:
        config = {
            "n_molecules": args.n_molecules,
            "prod_steps": args.prod_steps,
            "device": args.device,
        }

    results = run_sweep(
        recipes=recipes,
        config=config,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
    )

    # Summary
    if results:
        sigmas = [r["properties"]["conductivity_mS_cm"] for r in results]
        logger.info(f"\nSummary: {len(results)} recipes")
        logger.info(f"  σ range: {min(sigmas):.2f} - {max(sigmas):.2f} mS/cm")
        logger.info(f"  σ mean:  {sum(sigmas)/len(sigmas):.2f} mS/cm")
    else:
        logger.error("No results produced!")
        sys.exit(1)
