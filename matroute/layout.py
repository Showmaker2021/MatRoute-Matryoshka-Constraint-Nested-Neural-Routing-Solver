'''
BlockLayout definition for MatRoute (Matryoshka Constraint-Nested Routing Solver).
Defines block dimensions, head allocation, and active index slicing.
'''

from dataclasses import dataclass
from typing import Dict, Tuple, Set, Union, Optional
import torch


CANONICAL_BLOCK_NAMES: Tuple[str, ...] = ("CORE", "DEM", "TW", "PRZ", "PD", "RTE")

PRESET_CONFIGS: Dict[str, Dict[str, int]] = {
    "d192": {
        "CORE": 80,
        "DEM": 32,
        "TW": 32,
        "PRZ": 16,
        "PD": 16,
        "RTE": 16,
    },
    "d128": {
        "CORE": 48,
        "DEM": 24,
        "TW": 24,
        "PRZ": 8,
        "PD": 8,
        "RTE": 16,
    },
    "d64": {
        "CORE": 24,
        "DEM": 8,
        "TW": 8,
        "PRZ": 8,
        "PD": 8,
        "RTE": 8,
    },
}


@dataclass(frozen=True)
class BlockLayout:
    names: Tuple[str, ...] = CANONICAL_BLOCK_NAMES
    sizes: Dict[str, int] = None
    head_dim: int = 8

    def __post_init__(self):
        if self.sizes is None:
            # Default to d192
            object.__setattr__(self, "sizes", dict(PRESET_CONFIGS["d192"]))

        # Verify sizes match names and are multiples of head_dim
        for name in self.names:
            if name not in self.sizes:
                raise ValueError(f"Block '{name}' missing from sizes dictionary: {self.sizes}")
            size = self.sizes[name]
            if size % self.head_dim != 0:
                raise ValueError(
                    f"Block '{name}' size {size} is not a multiple of head_dim {self.head_dim}"
                )

    @classmethod
    def from_preset(cls, preset_name: str, head_dim: int = 8) -> "BlockLayout":
        if preset_name not in PRESET_CONFIGS:
            raise KeyError(f"Unknown layout preset '{preset_name}'. Available: {list(PRESET_CONFIGS.keys())}")
        return cls(sizes=dict(PRESET_CONFIGS[preset_name]), head_dim=head_dim)

    @classmethod
    def preset(cls, preset_name: str, head_dim: int = 8) -> "BlockLayout":
        return cls.from_preset(preset_name, head_dim=head_dim)

    def dim(self) -> int:
        """Total dimension across all defined blocks."""
        return sum(self.sizes[name] for name in self.names)

    def idx(self, name: str) -> torch.LongTensor:
        """Returns 1D Tensor of indices corresponding to block `name` in full embedding."""
        if name not in self.names:
            raise KeyError(f"Unknown block name: '{name}'. Valid names: {self.names}")

        offset = 0
        for b_name in self.names:
            b_size = self.sizes[b_name]
            if b_name == name:
                return torch.arange(offset, offset + b_size, dtype=torch.long)
            offset += b_size
        raise RuntimeError(f"Could not compute index range for block '{name}'")

    def n_heads(self, name: str) -> int:
        """Returns number of attention heads allocated to block `name`."""
        if name not in self.names:
            raise KeyError(f"Unknown block name: '{name}'. Valid names: {self.names}")
        return self.sizes[name] // self.head_dim

    def total_heads(self) -> int:
        """Total number of attention heads across all blocks."""
        return sum(self.n_heads(name) for name in self.names)

    def active_names(self, S: Union[Set[str], frozenset, Tuple[str, ...], None]) -> Tuple[str, ...]:
        """Returns tuple of active block names in canonical order, always including CORE."""
        s_set = set(S) if S is not None else set()
        active = ["CORE"]
        for name in self.names:
            if name != "CORE" and name in s_set:
                active.append(name)
        return tuple(active)

    def active_idx(self, S: Union[Set[str], frozenset, Tuple[str, ...], None]) -> torch.LongTensor:
        """
        Returns 1D LongTensor of indices for active blocks (CORE + S) in full embedding,
        strictly maintaining canonical block ordering.
        """
        active_block_names = self.active_names(S)
        indices_list = [self.idx(name) for name in active_block_names]
        if not indices_list:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(indices_list)

    def active_dim(self, S: Union[Set[str], frozenset, Tuple[str, ...], None]) -> int:
        """Total dimension of active blocks."""
        active_block_names = self.active_names(S)
        return sum(self.sizes[name] for name in active_block_names)

    def active_heads(self, S: Union[Set[str], frozenset, Tuple[str, ...], None]) -> int:
        """Total attention heads of active blocks."""
        active_block_names = self.active_names(S)
        return sum(self.n_heads(name) for name in active_block_names)

    @classmethod
    def preset(cls, name: str, head_dim: int = 8) -> "BlockLayout":
        """Factory for predefined presets ('d192', 'd128', 'd64')."""
        if name not in PRESET_CONFIGS:
            raise KeyError(f"Unknown preset: '{name}'. Available: {list(PRESET_CONFIGS.keys())}")
        return cls(names=CANONICAL_BLOCK_NAMES, sizes=dict(PRESET_CONFIGS[name]), head_dim=head_dim)
