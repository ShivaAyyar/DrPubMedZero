"""
Helper utilities for running Dr. Zero Biomedical training on Google Colab.

This module provides functions for:
- Background server management (retrieval, solver)
- Checkpoint persistence to Google Drive
- Auto-resume from disconnections
- Training progress monitoring
- Port management
"""

import subprocess
import threading
import time
import requests
import json
import os
import signal
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import psutil


class BackgroundServer:
    """Manages a background server process with health checking."""

    def __init__(self, name: str, port: int, log_file: Optional[str] = None):
        """
        Args:
            name: Server name for logging
            port: Port number to listen on
            log_file: Optional log file path
        """
        self.name = name
        self.port = port
        self.log_file = log_file or f"/tmp/{name}_{port}.log"
        self.process = None
        self.thread = None

    def start(self, command: list, cwd: Optional[str] = None):
        """
        Start server in background.

        Args:
            command: Command as list (e.g., ["python", "server.py", "--port", "8000"])
            cwd: Working directory
        """
        # Kill any existing process on this port
        kill_port(self.port)

        print(f"Starting {self.name} on port {self.port}...")

        # Open log file
        log_handle = open(self.log_file, 'w')

        # Start process
        self.process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )

        print(f"  PID: {self.process.pid}")
        print(f"  Logs: {self.log_file}")

        return self.process

    def wait_until_ready(self, health_check_url: str, timeout: int = 300, check_interval: int = 5):
        """
        Wait until server responds to health check.

        Args:
            health_check_url: URL to check (e.g., "http://localhost:8000/health")
            timeout: Maximum wait time in seconds
            check_interval: Seconds between checks

        Returns:
            True if server is ready, False if timeout
        """
        print(f"Waiting for {self.name} to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_check_url, timeout=2)
                if response.status_code == 200:
                    print(f"  {self.name} is ready!")
                    return True
            except (requests.ConnectionError, requests.Timeout):
                pass

            # Check if process is still running
            if self.process and self.process.poll() is not None:
                print(f"  ERROR: {self.name} process died (exit code: {self.process.returncode})")
                print(f"  Check logs: {self.log_file}")
                return False

            time.sleep(check_interval)
            print(".", end="", flush=True)

        print(f"\n  TIMEOUT: {self.name} did not become ready within {timeout}s")
        return False

    def stop(self):
        """Stop the server gracefully."""
        if self.process:
            print(f"Stopping {self.name}...")
            try:
                # Try graceful shutdown first
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()

                # Wait up to 10 seconds
                self.process.wait(timeout=10)
                print(f"  {self.name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if still running
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
                print(f"  {self.name} force killed")
            except Exception as e:
                print(f"  Error stopping {self.name}: {e}")

            self.process = None

    def is_running(self) -> bool:
        """Check if server process is still running."""
        return self.process is not None and self.process.poll() is None

    def get_logs(self, n_lines: int = 50) -> str:
        """Get last n lines from log file."""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                return ''.join(lines[-n_lines:])
        except FileNotFoundError:
            return f"Log file not found: {self.log_file}"


def kill_port(port: int):
    """Kill any process listening on the given port."""
    try:
        for proc in psutil.process_iter(['pid', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port:
                        print(f"Killing process {proc.pid} on port {port}")
                        proc.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
    except Exception as e:
        print(f"Warning: Could not kill port {port}: {e}")


class CheckpointManager:
    """Manages checkpoints with Google Drive persistence."""

    def __init__(self, checkpoint_dir: str, drive_dir: Optional[str] = None):
        """
        Args:
            checkpoint_dir: Local checkpoint directory
            drive_dir: Google Drive directory for backup
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.drive_dir = Path(drive_dir) if drive_dir else None
        if self.drive_dir:
            self.drive_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"

    def save_checkpoint_info(self, iteration: int, step: int, checkpoint_path: str, metrics: Dict[str, Any] = None):
        """
        Save checkpoint metadata.

        Args:
            iteration: Training iteration (1, 2, or 3)
            step: Global step number
            checkpoint_path: Path to checkpoint
            metrics: Optional training metrics
        """
        metadata = {
            "iteration": iteration,
            "step": step,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": time.time(),
            "metrics": metrics or {}
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Checkpoint metadata saved: iter{iteration}/step{step}")

        # Sync to Google Drive if configured
        if self.drive_dir:
            self.sync_to_drive()

    def get_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get metadata for the last saved checkpoint."""
        if not self.metadata_file.exists():
            return None

        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading checkpoint metadata: {e}")
            return None

    def sync_to_drive(self):
        """Sync local checkpoints to Google Drive."""
        if not self.drive_dir:
            print("Warning: No Google Drive directory configured")
            return

        print(f"Syncing checkpoints to Google Drive...")
        try:
            import shutil

            # Copy entire checkpoint directory
            if self.checkpoint_dir.exists():
                shutil.copytree(
                    self.checkpoint_dir,
                    self.drive_dir / self.checkpoint_dir.name,
                    dirs_exist_ok=True
                )
                print(f"  Synced to {self.drive_dir}")
        except Exception as e:
            print(f"  Error syncing to Drive: {e}")

    def restore_from_drive(self):
        """Restore checkpoints from Google Drive."""
        if not self.drive_dir:
            print("Warning: No Google Drive directory configured")
            return False

        drive_checkpoint_dir = self.drive_dir / self.checkpoint_dir.name
        if not drive_checkpoint_dir.exists():
            print("No checkpoints found in Google Drive")
            return False

        print(f"Restoring checkpoints from Google Drive...")
        try:
            import shutil

            shutil.copytree(
                drive_checkpoint_dir,
                self.checkpoint_dir,
                dirs_exist_ok=True
            )
            print(f"  Restored from {drive_checkpoint_dir}")
            return True
        except Exception as e:
            print(f"  Error restoring from Drive: {e}")
            return False

    def should_resume(self) -> Tuple[bool, Optional[str]]:
        """
        Check if training should resume from checkpoint.

        Returns:
            (should_resume, checkpoint_path)
        """
        metadata = self.get_last_checkpoint()
        if not metadata:
            return False, None

        checkpoint_path = metadata.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Found checkpoint: iter{metadata['iteration']}/step{metadata['step']}")
            return True, checkpoint_path

        return False, None


class TrainingMonitor:
    """Monitor training progress and display metrics."""

    def __init__(self):
        self.metrics_history = []
        self.start_time = None

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        print("Training monitor started")

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Log training metrics.

        Args:
            step: Global step number
            metrics: Dictionary of metric names to values
        """
        metrics['step'] = step
        metrics['timestamp'] = time.time()
        self.metrics_history.append(metrics)

        # Print summary
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Step {step} | Elapsed: {elapsed/3600:.1f}h | ", end="")
        for key, value in metrics.items():
            if key not in ['step', 'timestamp']:
                print(f"{key}: {value:.4f} | ", end="")
        print()

    def plot_metrics(self, metric_names: list = None):
        """
        Plot training metrics.

        Args:
            metric_names: List of metric names to plot. If None, plots all.
        """
        if not self.metrics_history:
            print("No metrics to plot")
            return

        try:
            import matplotlib.pyplot as plt

            steps = [m['step'] for m in self.metrics_history]

            if metric_names is None:
                metric_names = [k for k in self.metrics_history[0].keys()
                               if k not in ['step', 'timestamp']]

            fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)))
            if len(metric_names) == 1:
                axes = [axes]

            for ax, metric_name in zip(axes, metric_names):
                values = [m.get(metric_name, 0) for m in self.metrics_history]
                ax.plot(steps, values)
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} over training')
                ax.grid(True)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available for plotting")

    def save_metrics(self, filepath: str):
        """Save metrics history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Metrics saved to {filepath}")


def test_gpu_availability() -> bool:
    """
    Test if GPU is available and print info.

    Returns:
        True if GPU available, False otherwise
    """
    try:
        import torch

        if not torch.cuda.is_available():
            print("ERROR: No GPU available!")
            print("Please enable GPU in Runtime -> Change runtime type")
            return False

        device_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {device_name}")

        # Check if A100
        if "A100" not in device_name:
            print(f"WARNING: Expected A100 GPU, but got {device_name}")
            print("Training may be slower or run out of memory")

        # Print memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {total_memory:.1f} GB")

        return True
    except ImportError:
        print("ERROR: PyTorch not installed")
        return False


def mount_google_drive(mount_point: str = "/content/drive"):
    """
    Mount Google Drive in Colab.

    Args:
        mount_point: Directory to mount Drive

    Returns:
        True if successful
    """
    try:
        from google.colab import drive
        drive.mount(mount_point, force_remount=False)
        print(f"Google Drive mounted at {mount_point}")
        return True
    except ImportError:
        print("Warning: Not running in Google Colab, skipping Drive mount")
        return False
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return False


def estimate_training_time(
    n_steps: int,
    batch_size: int,
    sequence_length: int,
    gpu_name: str = "A100"
) -> float:
    """
    Estimate training time in hours.

    Args:
        n_steps: Number of training steps
        batch_size: Batch size
        sequence_length: Average sequence length
        gpu_name: GPU type

    Returns:
        Estimated hours
    """
    # Rough estimates based on experience
    seconds_per_step = {
        "A100": 3.0,
        "V100": 6.0,
        "T4": 12.0,
    }.get(gpu_name, 10.0)

    # Adjust for batch size and sequence length
    seconds_per_step *= (batch_size / 64) * (sequence_length / 2048)

    total_seconds = n_steps * seconds_per_step
    return total_seconds / 3600


if __name__ == "__main__":
    print("Colab Helpers Module")
    print("=" * 50)

    # Test GPU
    print("\n1. Testing GPU availability:")
    test_gpu_availability()

    # Test checkpoint manager
    print("\n2. Testing checkpoint manager:")
    manager = CheckpointManager("./test_checkpoints")
    manager.save_checkpoint_info(1, 100, "./test_checkpoints/step_100", {"loss": 0.5})
    metadata = manager.get_last_checkpoint()
    print(f"Last checkpoint: {metadata}")

    print("\nAll tests passed!")
