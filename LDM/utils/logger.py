"""
Logging utilities for DDPM training.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch


def setup_logger(
    name: str,
    log_dir: str,
    level: int = logging.INFO,
    to_console: bool = True,
) -> logging.Logger:
    """
    Setup logger with file and optional console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        to_console: Whether to also log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str, exp_name: str):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory for log files
            exp_name: Experiment name
        """
        self.log_dir = Path(log_dir)
        self.exp_name = exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / f"{exp_name}_metrics.jsonl"
        self.metrics = []

    def log(self, step: int, metrics: Dict[str, Any]):
        """
        Log metrics for a step.

        Args:
            step: Training step
            metrics: Dictionary of metric names and values
        """
        entry = {"step": step, **metrics}
        self.metrics.append(entry)

        # Append to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_metrics(self) -> list:
        """Get all logged metrics."""
        return self.metrics

    def save_summary(self, summary: Dict[str, Any]):
        """Save experiment summary."""
        summary_file = self.log_dir / f"{self.exp_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)


class CheckpointManager:
    """Manager for model checkpoints."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ):
        """
        Save checkpoint.

        Args:
            step: Training step
            model: Model to save
            optimizer: Optimizer to save
            ema: Optional EMA model
            scheduler: Optional learning rate scheduler
            metrics: Optional metrics to save
            config: Optional config to save
        """
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if ema is not None:
            checkpoint["ema_state_dict"] = ema.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        if config is not None:
            checkpoint["config"] = config.__dict__ if hasattr(config, "__dict__") else config

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        # Also save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        ema: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optional optimizer to load into
            ema: Optional EMA model to load into
            scheduler: Optional scheduler to load into
            device: Device to load to

        Returns:
            Checkpoint dict with step and metrics
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if ema is not None and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "step": checkpoint.get("step", 0),
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint.get("config", {}),
        }

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint if exists."""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            return latest_path
        return None
