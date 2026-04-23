from __future__ import annotations

import torch

from minisgl.engine.offload import (
    LayerOffloadManager,
    OffloadPolicy,
    TenantOffloadState,
    discover_model_blocks,
)
from minisgl.layers import BaseOP, OPList


class DummyBlock(BaseOP):
    def __init__(self, value: float):
        self.weight = torch.tensor([value], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight


class DummyDecoder(BaseOP):
    def __init__(self):
        self.layers = OPList([DummyBlock(1.0), DummyBlock(2.0), DummyBlock(3.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return x


class DummyModel(BaseOP):
    def __init__(self):
        self.model = DummyDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


def test_offload_policy_idle_and_budget():
    policy = OffloadPolicy(idle_seconds=10.0, max_active_models=1)
    states = [
        TenantOffloadState("active", True, 0.5, activation_count=2, offload_count=0),
        TenantOffloadState("idle", True, 20.0, activation_count=1, offload_count=0),
        TenantOffloadState("inactive", False, 100.0, activation_count=1, offload_count=1),
    ]
    selected = policy.select_tenants_to_offload(states, keep_tenant_id="active")
    assert selected == ["idle"]


def test_discover_model_blocks_uses_common_decoder_path():
    specs = discover_model_blocks(DummyModel())
    assert [spec.name for spec in specs] == [
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
    ]


def test_layer_offload_manager_respects_resident_budget_one():
    model = DummyModel()
    specs = discover_model_blocks(model)
    mgr = LayerOffloadManager(
        model,
        specs,
        device=torch.device("cpu"),
        max_resident_blocks=1,
    )

    assert sum(mgr._resident) == 1
    mgr.prepare(2)
    assert sum(mgr._resident) == 1
    assert mgr._resident[2]
    assert not mgr._resident[0]
    assert not mgr._resident[1]


def test_layer_offload_manager_prefetches_next_block_when_budget_allows():
    model = DummyModel()
    specs = discover_model_blocks(model)
    mgr = LayerOffloadManager(
        model,
        specs,
        device=torch.device("cpu"),
        max_resident_blocks=2,
    )

    mgr.prepare(1)
    assert mgr._resident[1]
    assert mgr._resident[2]
    assert sum(mgr._resident) <= 2
