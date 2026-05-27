# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, TypedDict, List, Tuple
from typing_extensions import Annotated, Literal

import tyro
import torch

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerf_xray.objects import Object


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # which model to evaluate
    which: Literal['forward','backward','mixed'] = 'mixed'

    def main(self) -> None:
        """Main function."""
        CONSOLE.print(f"[underline]Evaluating {self.load_config} in {self.which} mode[/underline]")
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True, which=self.which)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f":white_check_mark: Saved results to: {self.output_path}")

@dataclass
class ComputeDensityLoss:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = {key:val.item() for key,val in pipeline.calculate_density_loss(sampling='grid').items()}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f":white_check_mark: Saved results to: {self.output_path}")

@dataclass
class ComputeVolumeMismatch:
    """Load a checkpoint, compute some metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # how many points to evaluate
    npoints: int = 1<<20 # 1,048,576

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            metrics_dict = {'mismatch': pipeline.get_fields_mismatch_penalty(reduction='none', npoints=self.npoints).cpu().numpy().tolist()}

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f":white_check_mark: Saved results to: {self.output_path}")

@dataclass
class ComputeNormedCorrelation:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # target times
    target_times: List[float]
    # target files
    target_files: List[Path]
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # number of points to evaluate
    npoints: int = 1<<20
    # extent of the scene to evaluate
    extent: Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]] = field(default_factory=lambda: ((-1,1),(-1,1),(-1,1)))

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        assert len(self.target_times)==len(self.target_files)
        # metrics_dict = {key:val.item() for key,val in pipeline.calculate_density_loss().items()}
        metrics_dict = {}
        for t, fn in zip(self.target_times, self.target_files):
            print(f'Loading object from {fn} at time {t}')
            obj = Object.from_file(fn)
            with torch.no_grad():
                metrics_dict[t] = pipeline.get_eval_density_loss(target=obj, npoints=self.npoints, time=t, sampling='grid', extent=self.extent, batch_size=1<<20)
                for key, val in metrics_dict[t].items():
                    if isinstance(val, torch.Tensor):
                        metrics_dict[t][key] = val.item()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
            "extent": self.extent,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f":white_check_mark: Saved results to: {self.output_path}")

Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ComputePSNR, tyro.conf.subcommand(name="compute-psnr")],
        Annotated[ComputeDensityLoss, tyro.conf.subcommand(name="compute-density-loss")],
        Annotated[ComputeVolumeMismatch, tyro.conf.subcommand(name="compute-volume-mismatch")],
        Annotated[ComputeNormedCorrelation, tyro.conf.subcommand(name="compute-normed-correlation")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa TODO document other methods
