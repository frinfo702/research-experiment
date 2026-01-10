"""
DDPM: Denoising Diffusion Probabilistic Models

A reproduction of Ho et al. 2020 "Denoising Diffusion Probabilistic Models"

Usage (with uv):
    # Setup
    uv sync

    # Train baseline model (T=1000, linear schedule)
    uv run python main.py train

    # Train with custom settings
    uv run python main.py train --timesteps 500 --beta-schedule cosine

    # Generate samples from trained model
    uv run python main.py sample --checkpoint outputs/ddpm_baseline/checkpoints/checkpoint_latest.pt

    # Evaluate FID score
    uv run python main.py evaluate --checkpoint outputs/ddpm_baseline/checkpoints/checkpoint_latest.pt

    # Run ablation experiments
    uv run python main.py ablation train
    uv run python main.py ablation evaluate

    # Debug run (Mac M4)
    ./scripts/run_baseline_debug.sh
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="DDPM: Denoising Diffusion Probabilistic Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========== Train command ==========
    train_parser = subparsers.add_parser("train", help="Train DDPM model")
    train_parser.add_argument("--timesteps", type=int, default=1000,
                              help="Number of diffusion timesteps (default: 1000)")
    train_parser.add_argument("--beta-schedule", type=str, default="linear",
                              choices=["linear", "cosine", "quadratic"],
                              help="Beta schedule type (default: linear)")
    train_parser.add_argument("--batch-size", type=int, default=128,
                              help="Training batch size (default: 128)")
    train_parser.add_argument("--total-steps", type=int, default=800000,
                              help="Total training steps (default: 800000)")
    train_parser.add_argument("--lr", type=float, default=2e-4,
                              help="Learning rate (default: 2e-4)")
    train_parser.add_argument("--exp-name", type=str, default=None,
                              help="Experiment name")
    train_parser.add_argument("--output-dir", type=str, default="./outputs",
                              help="Output directory (default: ./outputs)")
    train_parser.add_argument("--data-dir", type=str, default="./data",
                              help="Data directory (default: ./data)")
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Path to checkpoint to resume from")
    train_parser.add_argument("--device", type=str, default="cuda",
                              choices=["cuda", "mps", "cpu"],
                              help="Device to use (default: cuda)")
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed (default: 42)")
    train_parser.add_argument("--debug", action="store_true",
                              help="Debug mode (fewer steps, more logging)")

    # ========== Sample command ==========
    sample_parser = subparsers.add_parser("sample", help="Generate samples from trained model")
    sample_parser.add_argument("--checkpoint", type=str, required=True,
                               help="Path to model checkpoint")
    sample_parser.add_argument("--output-dir", type=str, default="./generated",
                               help="Output directory for samples")
    sample_parser.add_argument("--num-samples", type=int, default=64,
                               help="Number of samples to generate")
    sample_parser.add_argument("--batch-size", type=int, default=64,
                               help="Batch size for generation")
    sample_parser.add_argument("--timesteps", type=int, default=1000,
                               help="Number of diffusion timesteps")
    sample_parser.add_argument("--beta-schedule", type=str, default="linear",
                               choices=["linear", "cosine", "quadratic"],
                               help="Beta schedule type")
    sample_parser.add_argument("--image-size", type=int, default=32,
                               help="Image size")
    sample_parser.add_argument("--no-ema", action="store_true",
                               help="Use model weights instead of EMA")
    sample_parser.add_argument("--save-individual", action="store_true",
                               help="Save individual images")
    sample_parser.add_argument("--interpolation", action="store_true",
                               help="Generate interpolation visualization")
    sample_parser.add_argument("--denoising-vis", action="store_true",
                               help="Generate denoising process visualization")
    sample_parser.add_argument("--device", type=str, default="cuda",
                               choices=["cuda", "mps", "cpu"],
                               help="Device to use")

    # ========== Evaluate command ==========
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model with FID score")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                             help="Path to model checkpoint")
    eval_parser.add_argument("--num-samples", type=int, default=50000,
                             help="Number of samples for FID calculation")
    eval_parser.add_argument("--batch-size", type=int, default=256,
                             help="Batch size")
    eval_parser.add_argument("--timesteps", type=int, default=1000,
                             help="Number of diffusion timesteps")
    eval_parser.add_argument("--beta-schedule", type=str, default="linear",
                             choices=["linear", "cosine", "quadratic"],
                             help="Beta schedule type")
    eval_parser.add_argument("--data-dir", type=str, default="./data",
                             help="Data directory")
    eval_parser.add_argument("--stats-path", type=str, default=None,
                             help="Path to precomputed CIFAR-10 stats")
    eval_parser.add_argument("--no-ema", action="store_true",
                             help="Use model weights instead of EMA")
    eval_parser.add_argument("--device", type=str, default="cuda",
                             choices=["cuda", "mps", "cpu"],
                             help="Device to use")

    # ========== Ablation command ==========
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation experiments")
    ablation_subparsers = ablation_parser.add_subparsers(dest="ablation_command")

    # Ablation train
    abl_train = ablation_subparsers.add_parser("train", help="Run ablation training")
    abl_train.add_argument("--timesteps", type=int, nargs="+",
                           default=[100, 500, 1000, 2000],
                           help="Timesteps to test")
    abl_train.add_argument("--schedules", type=str, nargs="+",
                           default=["linear", "cosine", "quadratic"],
                           help="Beta schedules to test")
    abl_train.add_argument("--total-steps", type=int, default=800000,
                           help="Total training steps per experiment")
    abl_train.add_argument("--output-dir", type=str, default="./outputs/ablation",
                           help="Output directory")
    abl_train.add_argument("--data-dir", type=str, default="./data",
                           help="Data directory")
    abl_train.add_argument("--device", type=str, default="cuda",
                           help="Device to use")

    # Ablation single
    abl_single = ablation_subparsers.add_parser("single", help="Run single ablation experiment")
    abl_single.add_argument("--timesteps", type=int, required=True,
                            help="Number of timesteps")
    abl_single.add_argument("--schedule", type=str, required=True,
                            choices=["linear", "cosine", "quadratic"],
                            help="Beta schedule")
    abl_single.add_argument("--total-steps", type=int, default=800000,
                            help="Total training steps")
    abl_single.add_argument("--output-dir", type=str, default="./outputs/ablation",
                            help="Output directory")
    abl_single.add_argument("--data-dir", type=str, default="./data",
                            help="Data directory")
    abl_single.add_argument("--device", type=str, default="cuda",
                            help="Device to use")
    abl_single.add_argument("--resume", type=str, default=None,
                            help="Resume from checkpoint")

    # Ablation evaluate
    abl_eval = ablation_subparsers.add_parser("evaluate", help="Evaluate ablation experiments")
    abl_eval.add_argument("--output-dir", type=str, default="./outputs/ablation",
                          help="Directory containing trained models")
    abl_eval.add_argument("--data-dir", type=str, default="./data",
                          help="Data directory")
    abl_eval.add_argument("--device", type=str, default="cuda",
                          help="Device to use")
    abl_eval.add_argument("--num-samples", type=int, default=50000,
                          help="Number of samples for FID")

    # Ablation list
    abl_list = ablation_subparsers.add_parser("list", help="List planned experiments")
    abl_list.add_argument("--timesteps", type=int, nargs="+",
                          default=[100, 500, 1000, 2000])
    abl_list.add_argument("--schedules", type=str, nargs="+",
                          default=["linear", "cosine", "quadratic"])

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # ========== Execute commands ==========
    if args.command == "train":
        from train import main as train_main
        sys.argv = ["train.py"]
        for key, value in vars(args).items():
            if key == "command":
                continue
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        train_main()

    elif args.command == "sample":
        from sample import main as sample_main
        sys.argv = ["sample.py"]
        for key, value in vars(args).items():
            if key == "command":
                continue
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        sample_main()

    elif args.command == "evaluate":
        from evaluate import main as eval_main
        sys.argv = ["evaluate.py"]
        for key, value in vars(args).items():
            if key == "command":
                continue
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        eval_main()

    elif args.command == "ablation":
        from scripts.run_ablation import main as ablation_main
        # Reconstruct sys.argv for ablation script
        sys.argv = ["run_ablation.py"]
        if args.ablation_command:
            sys.argv.append(args.ablation_command)
            for key, value in vars(args).items():
                if key in ["command", "ablation_command"]:
                    continue
                if value is not None:
                    if isinstance(value, bool):
                        if value:
                            sys.argv.append(f"--{key.replace('_', '-')}")
                    elif isinstance(value, list):
                        sys.argv.append(f"--{key.replace('_', '-')}")
                        sys.argv.extend([str(v) for v in value])
                    else:
                        sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        ablation_main()


if __name__ == "__main__":
    main()
