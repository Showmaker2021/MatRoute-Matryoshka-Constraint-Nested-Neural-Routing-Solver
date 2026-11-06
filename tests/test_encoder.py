'''
Unit tests for Giai đoạn 3 (matroute/nn/, encoder.py).
Implements the 5 mandatory acceptance tests:
1. test_prefix_equivalence
2. test_core_ignores_constraint_blocks
3. test_mask_vs_slice
4. test_asym_heads
5. test_block_norm_independence
'''

import unittest
import sys
import os
import math
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from matroute.layout import BlockLayout
from matroute.nn.block_linear import BlockLinear
from matroute.nn.block_norm import BlockInstanceNorm
from matroute.nn.block_attn import BlockAttention
from matroute.encoder import EncoderLayer, Encoder


class TestEncoderBlockCausal(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(3407)
        self.layout = BlockLayout.preset("d64")  # Fast for unit tests
        self.batch_size = 2
        self.num_nodes = 10

    def test_prefix_equivalence(self):
        """
        Test 1:
        Verifies that CORE output of the encoder is completely invariant to
        random changes in constraint block inputs when running on any active set S.
        """
        encoder = Encoder(self.layout, num_layers=3)
        encoder.eval()

        core_dim = self.layout.sizes["CORE"]
        S = frozenset({"DEM", "TW"})
        active_dim = self.layout.active_dim(S)
        constraint_dim = active_dim - core_dim

        # Fixed CORE input
        h_core = torch.randn(self.batch_size, self.num_nodes, core_dim)
        dist = torch.rand(self.batch_size, self.num_nodes, self.num_nodes)

        # Two different random constraint inputs
        h_const1 = torch.randn(self.batch_size, self.num_nodes, constraint_dim)
        h_const2 = torch.randn(self.batch_size, self.num_nodes, constraint_dim)

        in1 = torch.cat([h_core, h_const1], dim=-1)
        in2 = torch.cat([h_core, h_const2], dim=-1)

        with torch.no_grad():
            out1 = encoder(in1, S=S, dist=dist)
            out2 = encoder(in2, S=S, dist=dist)

            # Extract CORE output
            out1_core = out1[..., :core_dim]
            out2_core = out2[..., :core_dim]

            diff = (out1_core - out2_core).abs().max().item()
            self.assertLess(
                diff,
                1e-6,
                f"Prefix equivalence violated! CORE output differed by {diff:.8e} when constraint inputs changed.",
            )

    def test_core_ignores_constraint_blocks(self):
        """
        Test 2:
        Verifies that the analytical derivative d(h_CORE) / d(h_constraint) is strictly zero.
        """
        encoder = Encoder(self.layout, num_layers=3)
        core_dim = self.layout.sizes["CORE"]
        S = frozenset({"DEM", "TW"})
        active_dim = self.layout.active_dim(S)
        constraint_dim = active_dim - core_dim

        h_core = torch.randn(self.batch_size, self.num_nodes, core_dim, requires_grad=True)
        h_const = torch.randn(self.batch_size, self.num_nodes, constraint_dim, requires_grad=True)
        dist = torch.rand(self.batch_size, self.num_nodes, self.num_nodes)

        h_in = torch.cat([h_core, h_const], dim=-1)
        out = encoder(h_in, S=S, dist=dist)
        out_core = out[..., :core_dim]

        # Scalar loss summing only CORE outputs
        loss = out_core.sum()
        loss.backward()

        self.assertIsNotNone(h_core.grad, "CORE must receive gradients from its own output")
        if h_const.grad is not None:
            max_const_grad = h_const.grad.abs().max().item()
            self.assertEqual(
                max_const_grad,
                0.0,
                f"Constraint block leaked gradients into CORE! Max grad: {max_const_grad:.8e}",
            )

    def test_mask_vs_slice(self):
        """
        Test 3:
        Verifies that sliced inference matches full masked linear computation within tolerance < 1e-5.
        """
        blinear = BlockLinear(self.layout)
        blinear.eval()

        S = frozenset({"DEM", "TW", "PRZ"})
        idx = self.layout.active_idx(S)
        h_active = torch.randn(self.batch_size, self.num_nodes, len(idx))

        # Sliced forward (standard in BlockLinear)
        out_sliced = blinear(h_active, S=S)

        # Full masked weight computation for comparison
        w_eff = blinear.weight * blinear.mask
        w_active_manual = w_eff[idx][:, idx]
        b_active_manual = blinear.bias[idx] if blinear.bias is not None else 0
        out_manual = h_active @ w_active_manual.T + b_active_manual

        diff = (out_sliced - out_manual).abs().max().item()
        self.assertLess(diff, 1e-6, f"Mask vs slice difference too high: {diff:.8e}")

    def test_asym_heads(self):
        """
        Test 4:
        Verifies that for symmetric distance matrices (D = D^T), the two halves of CORE heads
        receive identical attention pair biases.
        """
        attn = BlockAttention(self.layout)
        attn.eval()

        # Symmetric distance matrix: D = D^T
        raw_pts = torch.rand(self.batch_size, self.num_nodes, 2)
        dist_sym = torch.cdist(raw_pts, raw_pts, p=2)
        self.assertTrue(torch.allclose(dist_sym, dist_sym.transpose(1, 2), atol=1e-6))

        # Equalize alpha parameters across CORE heads for exact numerical comparison
        with torch.no_grad():
            attn.core_alpha.fill_(1.5)

        max_dist = dist_sym.amax(dim=(1, 2), keepdim=True) + 1e-8
        dist_norm = dist_sym / max_dist
        dist_norm_t = dist_norm.transpose(1, 2)
        log_scale = math.log2(self.num_nodes)

        bias = attn._compute_pair_bias("CORE", dist_norm, dist_norm_t, log_scale)
        n_heads = self.layout.n_heads("CORE")
        half = n_heads // 2

        bias_first_half = bias[:, :half]
        bias_second_half = bias[:, half : 2 * half]

        diff = (bias_first_half - bias_second_half).abs().max().item()
        self.assertLess(
            diff,
            1e-6,
            f"Symmetric distance did not yield identical pair bias across D and D^T heads: {diff:.8e}",
        )

    def test_block_norm_independence(self):
        """
        Test 5:
        Verifies that changing inputs of block TW does NOT alter the normalization
        statistics or output of block CORE.
        """
        norm = BlockInstanceNorm(self.layout)
        norm.eval()

        core_dim = self.layout.sizes["CORE"]
        tw_dim = self.layout.sizes["TW"]
        S = frozenset({"TW"})

        h_core = torch.randn(self.batch_size, self.num_nodes, core_dim)
        h_tw1 = torch.randn(self.batch_size, self.num_nodes, tw_dim)
        h_tw2 = torch.randn(self.batch_size, self.num_nodes, tw_dim) * 50.0 + 10.0  # radically different stats

        in1 = torch.cat([h_core, h_tw1], dim=-1)
        in2 = torch.cat([h_core, h_tw2], dim=-1)

        out1 = norm(in1, S=S)
        out2 = norm(in2, S=S)

        out1_core = out1[..., :core_dim]
        out2_core = out2[..., :core_dim]

        diff = (out1_core - out2_core).abs().max().item()
        self.assertLess(
            diff,
            1e-6,
            f"BlockInstanceNorm independence violated! CORE output changed by {diff:.8e} when TW changed.",
        )

    def test_sparse_knn_parity(self):
        """
        Test 6 (Stage 7):
        When K >= N, sparse kNN attention produces identical output to dense attention (diff < 1e-5).
        """
        encoder = Encoder(self.layout, num_layers=2)
        encoder.eval()

        S = frozenset({"DEM", "RTE"})
        active_dim = self.layout.active_dim(S)
        h_init = torch.randn(self.batch_size, self.num_nodes, active_dim)

        pts = torch.rand(self.batch_size, self.num_nodes, 2)
        dist = torch.cdist(pts, pts)

        with torch.no_grad():
            out_dense = encoder(h_init, S=S, dist=dist, sparse_knn=None)
            out_sparse_geq = encoder(h_init, S=S, dist=dist, sparse_knn=self.num_nodes + 50)

        diff = (out_dense - out_sparse_geq).abs().max().item()
        self.assertLess(
            diff,
            1e-5,
            f"Sparse kNN with K >= N differed from dense encoder: diff = {diff:.8e}",
        )


if __name__ == "__main__":
    unittest.main()
