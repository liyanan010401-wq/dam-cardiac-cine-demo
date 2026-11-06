"""Exercise the physical read and the r3 build v3 gate on synthetic tensors."""
import json
import math

import torch

from src.core import read, weighted_attention


def gate_from_support(distance_mm, acquired, forward_fraction, reverse_fraction):
    """Return r, s, g, q and the detached spatial-value weight."""
    residual_term = torch.exp(-distance_mm.square() / (2 * 2.0**2))
    support = acquired * forward_fraction * reverse_fraction
    score = support * (0.25 + 0.5 * residual_term)
    response = torch.sigmoid((score - 0.5) / 0.25)
    weight = torch.where(response >= 0.5, response, 0.0).detach()
    return residual_term, support, score, response, weight


def main():
    # A quarter-slice shift between feature planes valued 0 and 4 reads 1.
    memory = torch.zeros(1, 2, 3, 3)
    memory[:, 1] = 4.0
    affine = torch.diag(torch.tensor([1.6, 1.6, 8.0, 1.0]))
    displacement = torch.zeros(2, 3, 3, 3)
    displacement[..., 2] = 2.0
    aligned, inside = read(memory, displacement, affine)
    assert bool(inside[0, 1, 1]) and not bool(inside[1, 1, 1])
    torch.testing.assert_close(aligned[0, 0, 1, 1], torch.tensor(1.0))

    # Each target point is the origin. Its return point sets the distance in mm.
    threshold_mm = 2.0 * math.sqrt(2.0 * math.log(2.0))
    expected_distances = [0.0, threshold_mm, threshold_mm - 1.e-5,
                          threshold_mm + 1.e-5, 100.0, 0.0, 0.0, 0.0]
    returns = torch.zeros(len(expected_distances), 3, dtype=torch.float64)
    returns[:, 0] = torch.tensor(expected_distances, dtype=torch.float64)
    distance_mm = torch.linalg.vector_norm(returns.requires_grad_(), dim=-1)
    acquired = torch.ones_like(distance_mm)
    acquired[5] = 0.0
    forward_fraction = torch.ones_like(distance_mm)
    reverse_fraction = torch.ones_like(distance_mm)
    forward_fraction[6] = 0.75
    reverse_fraction[6] = 0.75
    forward_fraction[7] = 2.0 / 3.0
    r, s, g, q, w = gate_from_support(
        distance_mm, acquired, forward_fraction, reverse_fraction
    )
    expected_scores = torch.tensor(
        [0.75, 0.5, 0.25, 0.0, 0.75**3, 0.5], dtype=g.dtype
    )
    torch.testing.assert_close(g[[0, 1, 4, 5, 6, 7]], expected_scores)
    assert w[0] > 0.73105 and w[0] < 0.73107
    assert q[1] == w[1] == 0.5
    assert w[2] > 0.5 and w[3] == 0.0
    assert w[4] == w[5] == w[6] == 0.0
    assert w[7] == 0.5
    assert not w.requires_grad

    # Two supported spatial keys plus a pointer enter softmax. The first
    # spatial value is suppressed by distance. The third key is unsupported.
    weights = torch.cat((w[[4, 1, 5]], torch.ones(1, dtype=w.dtype)))
    query = torch.zeros(1, 1, dtype=w.dtype)
    keys = torch.zeros(4, 1, dtype=w.dtype)
    values = torch.tensor([[100.0], [4.0], [999.0], [6.0]], dtype=w.dtype)
    valid = torch.tensor([True, True, False, True])
    result = weighted_attention(query, keys, values, weights, valid)
    torch.testing.assert_close(result, torch.tensor([[8 / 3]], dtype=w.dtype))
    pointer_only = weighted_attention(
        query, keys, values, weights, torch.tensor([False, False, False, True])
    )
    empty = weighted_attention(query, keys, values, weights, torch.zeros_like(valid))
    torch.testing.assert_close(pointer_only, torch.tensor([[6.0]], dtype=w.dtype))
    torch.testing.assert_close(empty, torch.zeros_like(query))
    print(json.dumps({
        "source": "synthetic contract example using author-confirmed r3 build v3 gate.g_pre_score",
        "slice_spacing_mm": 8.0,
        "read_displacement_mm": 2.0,
        "interpolated_feature": aligned[0, 0, 1, 1].item(),
        "fully_supported_cutoff_mm": threshold_mm,
        "case_order": ["zero_residual", "cutoff", "just_below_cutoff", "just_above_cutoff",
                       "large_residual", "unacquired_target", "partial_forward_reverse",
                       "partial_support_score_boundary"],
        "distance_mm": distance_mm.tolist(),
        "residual_term": r.tolist(),
        "support": s.tolist(),
        "pre_projection_score": g.tolist(),
        "response": q.tolist(),
        "spatial_value_weights": w.tolist(),
        "attention_output": result.item(),
        "pointer_only_output": pointer_only.item(),
        "empty_bank_residual": empty.item(),
    }, indent=2))


if __name__ == "__main__":
    main()

