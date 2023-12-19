import csv

# from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

# import aim
from omegaconf import OmegaConf

from .table_printer import RichTablePrinter

logger_fields = {
    "timesteps": {"format": "{:.0f}"},
    "average_score": {"goal": "higher_is_better", "format": "{:.4f}"},
    # "std_score": {"format": "{:.4f}"},
    "alpha": {"format": "{:.4f}"},
    "batch_entropy": {"name": "batch_ent", "format": "{:.4f}"},
    "(.*)_loss": {"goal": "lower_is_better", "name": r"\1_l", "format": "{:.4f}"},
    "duration": {"format": "{:.1f}", "name": "dur(s)"},
    ".*": True,  # Any other field must be logged at the end
    # r"^((?!timesteps).)*$": {"format": "{:.4f}"},
    # r"^(?!.*(duration|timesteps)).*$": {"format": "{:.4f}"},
}


class Logger:
    def __init__(self, exp_dir: Path | str):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.printer = RichTablePrinter(fields=logger_fields, title=self.exp_dir.name)
        # self.printer.hijack_tqdm()

        self.csv_fd = self._add_fd("progress.csv")
        # self.aim_run = self._init_aim_run(self.exp_dir)
        self.log_cnt = 0

    def _init_aim_run(self, exp_dir: Path):
        pass
        # aim_run = aim.Run(repo=exp_dir.parents[2], experiment=exp_dir.parent.stem)
        # aim_run.name = exp_dir.stem

        # return aim_run

    def log(self, info: Dict[str, Any]):
        self.printer.log(info)
        if self.log_cnt == 0:
            self.csv_writer = csv.DictWriter(self.csv_fd, fieldnames=list(info.keys()))
            self.csv_writer.writeheader()
        self.csv_writer.writerow(info)
        self.csv_fd.flush()
        # timesteps, duration = info["timesteps"], info["duration"]
        # self.aim_run.track(step=timesteps, value=info, context={"step": "timesteps"})
        # self.aim_run.track(step=duration, value=info, context={"step": "duration"})
        self.log_cnt += 1

    def save_cfg(self, config):
        OmegaConf.save(config, f=self.exp_dir / "config.yaml")
        # self.aim_run["config"] = asdict(config)

    def _add_fd(self, file_name, mode="w"):
        file_path = self.exp_dir / file_name
        # file_path.mkdir(parents=True, exist_ok=True)
        return open(file_path, mode)

    def close(self):
        self.csv_fd.close()
        # self.aim_run.close()
        self.printer.finalize()
