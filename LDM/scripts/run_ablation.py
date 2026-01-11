"""
Ablation experiment script for DDPM.
Runs experiments with different T values and beta schedules.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ABLATION_TIMESTEPS, ABLATION_SCHEDULES, get_config


def generate_experiment_configs(
    timesteps_list: list = None,
    schedules_list: list = None,
    total_steps: int = 800000,
    output_dir: str = "./outputs/ablation",
) -> list:
    """
    Generate configurations for ablation experiments.

    Args:
        timesteps_list: List of T values to test
        schedules_list: List of beta schedules to test
        total_steps: Total training steps per experiment
        output_dir: Base output directory

    Returns:
        List of experiment configurations
    """
    if timesteps_list is None:
        timesteps_list = ABLATION_TIMESTEPS
    if schedules_list is None:
        schedules_list = ABLATION_SCHEDULES

    experiments = []

    for timesteps, schedule in product(timesteps_list, schedules_list):
        exp_name = f"ddpm_T{timesteps}_{schedule}"
        config = get_config(
            timesteps=timesteps,
            beta_schedule=schedule,
            exp_name=exp_name,
        )
        config.training.total_steps = total_steps
        config.output_dir = output_dir

        experiments.append({
            "name": exp_name,
            "timesteps": timesteps,
            "schedule": schedule,
            "total_steps": total_steps,
            "config": config,
        })

    return experiments


def run_single_experiment(
    timesteps: int,
    schedule: str,
    total_steps: int,
    output_dir: str,
    data_dir: str,
    device: str,
    resume: str = None,
):
    """Run a single ablation experiment."""
    from train import train
    from config import get_config

    exp_name = f"ddpm_T{timesteps}_{schedule}"
    config = get_config(
        timesteps=timesteps,
        beta_schedule=schedule,
        exp_name=exp_name,
    )
    config.training.total_steps = total_steps
    config.output_dir = output_dir
    config.data_dir = data_dir
    config.device = device

    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Schedule: {schedule}")
    print(f"  Total steps: {total_steps}")
    print(f"{'='*60}\n")

    train(config, resume_from=resume)


def run_ablation_experiments(
    timesteps_list: list = None,
    schedules_list: list = None,
    total_steps: int = 800000,
    output_dir: str = "./outputs/ablation",
    data_dir: str = "./data",
    device: str = "cuda",
):
    """
    Run all ablation experiments.

    Args:
        timesteps_list: List of T values to test
        schedules_list: List of beta schedules to test
        total_steps: Total training steps per experiment
        output_dir: Output directory
        data_dir: Data directory
        device: Device to use
    """
    experiments = generate_experiment_configs(
        timesteps_list, schedules_list, total_steps, output_dir
    )

    print(f"Running {len(experiments)} ablation experiments")
    print(f"Experiments: {[e['name'] for e in experiments]}")

    results = []

    for exp in experiments:
        try:
            run_single_experiment(
                timesteps=exp["timesteps"],
                schedule=exp["schedule"],
                total_steps=exp["total_steps"],
                output_dir=output_dir,
                data_dir=data_dir,
                device=device,
            )
            results.append({
                "name": exp["name"],
                "status": "completed",
                "timesteps": exp["timesteps"],
                "schedule": exp["schedule"],
            })
        except Exception as e:
            print(f"Error in experiment {exp['name']}: {e}")
            results.append({
                "name": exp["name"],
                "status": "failed",
                "error": str(e),
            })

    # Save results summary
    summary_path = Path(output_dir) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiments": results,
        }, f, indent=2)

    print(f"\nAblation experiments complete. Summary saved to {summary_path}")

    return results


def evaluate_ablation_experiments(
    output_dir: str = "./outputs/ablation",
    data_dir: str = "./data",
    device: str = "cuda",
    num_samples: int = 50000,
):
    """
    Evaluate all ablation experiments and generate comparison table.

    Args:
        output_dir: Directory containing trained models
        data_dir: Data directory
        device: Device to use
        num_samples: Number of samples for FID calculation
    """
    from evaluate import evaluate_model
    from config import get_config

    output_path = Path(output_dir)
    results = []

    # Find all experiment directories
    exp_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("ddpm_T")]

    print(f"Found {len(exp_dirs)} experiments to evaluate")

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name

        # Parse experiment name to get config
        parts = exp_name.split("_")
        timesteps = int(parts[1][1:])  # Remove 'T' prefix
        schedule = parts[2]

        # Find checkpoint
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"

        if not checkpoint_path.exists():
            print(f"No checkpoint found for {exp_name}, skipping")
            continue

        print(f"\nEvaluating {exp_name}...")

        try:
            config = get_config(timesteps=timesteps, beta_schedule=schedule)

            eval_results = evaluate_model(
                checkpoint_path=str(checkpoint_path),
                config=config,
                device=device,
                num_samples=num_samples,
                data_dir=data_dir,
            )

            results.append({
                "name": exp_name,
                "timesteps": timesteps,
                "schedule": schedule,
                "fid": eval_results["fid"],
            })
        except Exception as e:
            print(f"Error evaluating {exp_name}: {e}")
            results.append({
                "name": exp_name,
                "timesteps": timesteps,
                "schedule": schedule,
                "fid": None,
                "error": str(e),
            })

    # Sort by FID
    results.sort(key=lambda x: x["fid"] if x["fid"] is not None else float("inf"))

    # Print comparison table
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    print(f"{'Experiment':<30} {'T':>8} {'Schedule':<12} {'FID':>10}")
    print("-"*70)

    for r in results:
        fid_str = f"{r['fid']:.4f}" if r["fid"] is not None else "N/A"
        print(f"{r['name']:<30} {r['timesteps']:>8} {r['schedule']:<12} {fid_str:>10}")

    print("="*70)

    # Save results
    results_path = output_path / "ablation_fid_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="DDPM Ablation Experiments")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Run ablation training")
    train_parser.add_argument("--timesteps", type=int, nargs="+",
                              default=ABLATION_TIMESTEPS,
                              help="Timesteps to test")
    train_parser.add_argument("--schedules", type=str, nargs="+",
                              default=ABLATION_SCHEDULES,
                              help="Beta schedules to test")
    train_parser.add_argument("--total-steps", type=int, default=800000,
                              help="Total training steps per experiment")
    train_parser.add_argument("--output-dir", type=str, default="./outputs/ablation",
                              help="Output directory")
    train_parser.add_argument("--data-dir", type=str, default="./data",
                              help="Data directory")
    train_parser.add_argument("--device", type=str, default="cuda",
                              help="Device to use")

    # Single experiment subcommand
    single_parser = subparsers.add_parser("single", help="Run single experiment")
    single_parser.add_argument("--timesteps", type=int, required=True,
                               help="Number of timesteps")
    single_parser.add_argument("--schedule", type=str, required=True,
                               choices=ABLATION_SCHEDULES,
                               help="Beta schedule")
    single_parser.add_argument("--total-steps", type=int, default=800000,
                               help="Total training steps")
    single_parser.add_argument("--output-dir", type=str, default="./outputs/ablation",
                               help="Output directory")
    single_parser.add_argument("--data-dir", type=str, default="./data",
                               help="Data directory")
    single_parser.add_argument("--device", type=str, default="cuda",
                               help="Device to use")
    single_parser.add_argument("--resume", type=str, default=None,
                               help="Resume from checkpoint")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate ablation experiments")
    eval_parser.add_argument("--output-dir", type=str, default="./outputs/ablation",
                             help="Directory containing trained models")
    eval_parser.add_argument("--data-dir", type=str, default="./data",
                             help="Data directory")
    eval_parser.add_argument("--device", type=str, default="cuda",
                             help="Device to use")
    eval_parser.add_argument("--num-samples", type=int, default=50000,
                             help="Number of samples for FID")

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List planned experiments")
    list_parser.add_argument("--timesteps", type=int, nargs="+",
                             default=ABLATION_TIMESTEPS)
    list_parser.add_argument("--schedules", type=str, nargs="+",
                             default=ABLATION_SCHEDULES)

    args = parser.parse_args()

    if args.command == "train":
        run_ablation_experiments(
            timesteps_list=args.timesteps,
            schedules_list=args.schedules,
            total_steps=args.total_steps,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            device=args.device,
        )
    elif args.command == "single":
        run_single_experiment(
            timesteps=args.timesteps,
            schedule=args.schedule,
            total_steps=args.total_steps,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            device=args.device,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        evaluate_ablation_experiments(
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            device=args.device,
            num_samples=args.num_samples,
        )
    elif args.command == "list":
        experiments = generate_experiment_configs(
            args.timesteps, args.schedules
        )
        print(f"\nPlanned experiments ({len(experiments)} total):")
        print("-" * 40)
        for exp in experiments:
            print(f"  {exp['name']}")
        print("-" * 40)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
