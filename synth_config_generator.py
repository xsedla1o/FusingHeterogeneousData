"""Simple module for generating synthetic datasets."""
import os.path
import shutil
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


class ConfigGenerator:
    """Class holding paths and configuration for generating synthesis params."""

    def __init__(self, base_dir: str, output_dir: str, module_names_: list[str], os_labels_: list[str]):
        """
        Holds paths and configuration for generating synthesis params.
        Args:
            base_dir: Path to fully synthetic base config
            output_dir: Where to generate new configurations
            module_names_: Identify modules
            os_labels_: Identify classes
        """
        self.base_dir = base_dir

        self.output_dir = output_dir
        self.curr_output_dir: str

        self.module_names = module_names_
        self.os_labels = os_labels_

    def generate_config(
        self,
        identifier: str,
        dist_generator: Union[Callable[[], pd.DataFrame], list[Callable[[], pd.DataFrame]]],
        *,
        active_modules_cnt: Optional[int] = None,
        class_cnt: Optional[int] = None,
        overlap_mean: Optional[float] = None,
        overlap_std: float = 1,
    ):
        """Generates a config - synthesis params."""
        if active_modules_cnt is None:
            active_modules_cnt = len(self.module_names)
        elif not 0 < active_modules_cnt <= len(self.module_names):
            raise ValueError(
                f"active_modules_cnt must be greater than zero and less than or equal to len(module_names) "
                f"(0 < {active_modules_cnt=} <= {len(self.module_names)})"
            )

        if class_cnt is None:
            class_cnt = len(self.os_labels)
        elif not 0 < class_cnt <= len(self.os_labels):
            raise ValueError(
                f"class_cnt must be greater than zero and less than or equal to len(os_labels) "
                f"(0 < {class_cnt=} <= {len(self.os_labels)})"
            )

        self.curr_output_dir = os.path.join(self.output_dir, f"full_synth_{identifier}")
        print(f"Generating config for {self.curr_output_dir}")
        self._duplicate_base_config()
        self._save_distribution_config(dist_generator)
        self._save_module_distribution(active_modules_cnt)
        self._save_module_overlap_distribution(active_modules_cnt, overlap_mean, overlap_std)
        self._save_os_distribution(class_cnt)

    def _duplicate_base_config(self):
        if os.path.exists(self.curr_output_dir):
            shutil.rmtree(self.curr_output_dir, ignore_errors=True)
        shutil.copytree(self.base_dir, self.curr_output_dir)

    def _save_distribution_config(
        self, dist_generator: Union[Callable[[], pd.DataFrame], list[Callable[[], pd.DataFrame]]]
    ) -> None:
        """Save distribution config for all modules."""
        if isinstance(dist_generator, list):
            for name, generator in zip(self.module_names, dist_generator):
                distribution = generator()
                distribution.to_csv(os.path.join(self.curr_output_dir, f"{name}.csv"))
        else:
            for name in self.module_names:
                distribution = dist_generator()
                distribution.to_csv(os.path.join(self.curr_output_dir, f"{name}.csv"))

    def _save_module_distribution(self, active_cnt: int) -> None:
        """Save probability distribution of modules for every overlap count."""
        dist = self._get_uniform_distribution_base(
            active_cnt, index=self.module_names, columns=range(1, len(self.module_names) + 1)
        )
        # add http_ua
        dist.loc["http_ua"] = [0 for _ in range(1, len(self.module_names) + 1)]
        dist.to_csv(os.path.join(self.curr_output_dir, "module_distribution.csv"))

    def _save_module_overlap_distribution(self, active_cnt: int, mean: float = None, std: float = 1) -> None:
        """Save probability distribution of module overlap."""
        if mean is None:
            mean = len(self.module_names)
        values = get_bell_curve_segment(length=len(self.module_names), mean=mean - 1, std=std)
        values.index += 1
        values[active_cnt:] = 0
        values /= values.sum()
        values.to_csv(os.path.join(self.curr_output_dir, "overlap_distribution.csv"))

    def _save_os_distribution(self, class_cnt: int):
        """Save probability distribution of annotated classes."""
        dist = self._get_uniform_distribution_base(
            class_cnt, index=self.os_labels, columns=range(1, len(self.module_names) + 1)
        )
        dist.to_csv(os.path.join(self.curr_output_dir, "os_distribution.csv"))

    @staticmethod
    def _get_uniform_distribution_base(active_cnt: int, *, index, columns):
        dist = pd.DataFrame(index=index, columns=columns, dtype=float)
        dist.iloc[:active_cnt] = dist.iloc[:active_cnt].fillna(1)
        dist.iloc[active_cnt:] = dist.iloc[active_cnt:].fillna(0)
        dist /= dist.sum()
        return dist


def get_uniform_module_distribution(os_labels: list, acc=1.0) -> pd.DataFrame:
    """Get a uniform distribution with the given accuracy."""
    base = pd.DataFrame(index=os_labels, columns=os_labels, data=np.identity(len(os_labels))) * acc
    for col in base.columns:
        if sum(base[col]) == 1:
            continue
        fill_val = (1 - sum(base[col])) / (len(base.columns) - 1)
        base[col] = base[col].replace(0, fill_val)
    return base


def get_variable_module_distribution(os_labels: list, min_acc=0.0, max_acc=1.0) -> pd.DataFrame:
    """Get a uniform distribution, but accuracy for each class is selected randomly from given bounds."""
    base = pd.DataFrame(index=os_labels, columns=os_labels)
    for i in base.columns:
        random_acc = (max_acc - min_acc) * np.random.random_sample() + min_acc
        fill_val = (1 - random_acc) / (len(base.columns) - 1)
        for j in base.index:
            if i == j:
                base.loc[j, i] = random_acc
            else:
                base.loc[j, i] = fill_val
    return base


def add_split(dist: pd.DataFrame, cnt: int = 1, size: int = 2) -> pd.DataFrame:
    """Simulate module having bad detection abilities for one class (or more)."""
    error_i = np.random.choice(dist.index, size=cnt, replace=False)
    for i, row in dist.iterrows():
        if i not in error_i:
            continue
        error_j = np.random.choice([x for x in dist.index if x != i], size=size - 1, replace=False).tolist()
        error_j.append(row.name)

        split_val = row[error_j].sum() / len(error_j)
        for j in error_j:
            dist.loc[i, j] = split_val
    return dist / dist.sum()


def add_confusion(dist: pd.DataFrame, cnt: int = 1, size: int = 2) -> pd.DataFrame:
    """Simulate module unable to discern between one pair of classes (or more)."""
    dist = add_split(dist, cnt, size).T
    return dist / dist.sum()


def get_bell_curve_segment(length: int, mean: float, std: float = 1) -> pd.Series:
    x = np.arange(0, length, 1)
    y = norm.pdf(x, mean, std)
    return pd.Series(index=x, data=y)
