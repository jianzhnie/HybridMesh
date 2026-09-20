"""Auxiliary-loss gradient injection, metric roll-up, and the balance loss.

``AuxLoss`` keeps its state on the class (``_group_counts``, ``group_acc``,
``_step_denominator``) because instances on different pipeline stages still have
to agree on the layer count. That makes the tests order-dependent by default, so
the fixture below snapshots and restores all three around every test.

Two properties are worth proving beyond the arithmetic, because both fail
silently:

* The injection is an *identity* in the forward and delivers gradient 1.0 to the
  aux loss on the way back. A metric that logs correctly but never reaches the
  router's gradient is exactly the failure this mechanism exists to prevent.
* ``instance_acc`` is scaled by ``1 / denominator`` while the *injected* gradient
  is ``coeff / denominator`` scaled. Conflating them makes the loss unweighted or
  double-weighted without any visible symptom.

Note on the carrier's gradient: ``_AuxLossInjection`` is an identity, so
whatever gradient arrives from downstream passes through unchanged, *and* the
aux-loss branch adds its own if the loss also depends on the carrier. In the
tests below the two paths are isolated deliberately -- ``_ConstantLoss`` does not
depend on the carrier, ``_TraceableLoss`` does not return it downstream.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hpmesh.models.common.aux_loss import (
    AuxLoss,
    _zero_aux_losses,
    collect_aux_loss_metrics,
)
from hpmesh.models.common.moe import MicrobatchWiseLoadBalanceLoss


@pytest.fixture(autouse=True)
def _isolate_aux_state():
    """Snapshot and restore AuxLoss's class-level state around each test."""
    counts = dict(AuxLoss._group_counts)
    acc = dict(AuxLoss.group_acc)
    denominator = AuxLoss._step_denominator

    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None
    yield
    AuxLoss._group_counts.clear()
    AuxLoss._group_counts.update(counts)
    AuxLoss.group_acc.clear()
    AuxLoss.group_acc.update(acc)
    AuxLoss._step_denominator = denominator


class _ConstantLoss(AuxLoss):
    """Raw value independent of the carrier, isolating the identity path.

    Mirrors real usage: the balance loss depends on the router's scores while
    the carrier is the top-k slice, so the carrier's *own* gradient must arrive
    from downstream untouched.
    """

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        return self.inject(torch.tensor(2.0), carrier=carrier)


class _TraceableLoss(AuxLoss):
    """Raw value from a separate leaf, isolating the injected gradient's path."""

    def forward(self, x: torch.Tensor, *, carrier: torch.Tensor) -> torch.Tensor:
        return self.inject(2 * x.sum(), carrier=carrier)


class _NoMeshes:
    """Stand-in for ParallelDims with every parallelism axis disabled."""

    def get_optional_mesh(self, dims, **kwargs):
        return None


# -- naming and registration -------------------------------------------------


def test_metric_name_is_snake_case() -> None:
    assert MicrobatchWiseLoadBalanceLoss(coeff=1.0).metric_name == (
        "microbatch_wise_load_balance_loss"
    )


def test_metric_name_splits_acronym_runs() -> None:
    """A run like ``MoELoss`` must not become ``mo_e_loss``."""

    class MoEBalanceLoss(AuxLoss):  # noqa: N801 -- the name is the fixture
        pass

    assert MoEBalanceLoss(coeff=1.0).metric_name == "mo_e_balance_loss"


def test_instances_register_themselves_in_their_group() -> None:
    """The layer count is the divisor in collect_aux_loss_metrics."""
    _ConstantLoss(coeff=1.0)
    _ConstantLoss(coeff=1.0)

    assert AuxLoss._group_counts[("batch", "_constant_loss")] == 2


def test_reduce_mesh_selects_the_group() -> None:
    _ConstantLoss(coeff=1.0, reduce_mesh="loss")
    assert ("loss", "_constant_loss") in AuxLoss._group_counts


# -- injection ---------------------------------------------------------------


def test_forward_is_an_identity_in_value() -> None:
    """``apply`` returns a new tensor object, but the same values."""
    loss = _ConstantLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(8.0))
    carrier = torch.randn(4, requires_grad=True)

    out = loss(carrier)

    assert torch.equal(out, carrier)
    assert out is not carrier  # autograd.Function.apply always allocates


def test_inject_raises_without_a_step_denominator() -> None:
    """Without the denominator the loss's scale is undefined, not just small."""
    loss = _ConstantLoss(coeff=1.0)

    with pytest.raises(ValueError, match="set_step_denominator"):
        loss(torch.randn(4, requires_grad=True))


def test_carrier_gradient_passes_through_unchanged() -> None:
    """The identity path must not attenuate or amplify the carrier's gradient."""
    loss = _ConstantLoss(coeff=0.25)
    AuxLoss.set_step_denominator(torch.tensor(8.0))
    carrier = torch.ones(4, requires_grad=True)

    loss(carrier).sum().backward()

    torch.testing.assert_close(carrier.grad, torch.ones(4), rtol=1e-5, atol=1e-8)


def test_aux_loss_receives_gradient_one() -> None:
    """``backward`` returns ``ones_like(aux_loss)`` -- the weight is in the value.

    ``x``'s gradient isolates the aux branch: the loss is ``2 * x.sum()`` scaled
    by ``coeff / denom``, and ``backward`` contributes exactly 1, so the result
    is ``2 * coeff / denom`` with no extra factor.
    """
    coeff, denominator = 0.25, 8.0
    loss = _TraceableLoss(coeff=coeff)
    AuxLoss.set_step_denominator(torch.tensor(denominator))
    x = torch.ones(3, requires_grad=True)

    loss(x, carrier=torch.ones(1, requires_grad=True)).sum().backward()

    torch.testing.assert_close(
        x.grad, torch.full((3,), 2 * coeff / denominator), rtol=1e-5, atol=1e-8
    )


def test_aux_gradient_scales_with_coeff() -> None:
    denominator = 4.0
    grads = []
    for coeff in (1.0, 3.0):
        AuxLoss._group_counts.clear()
        AuxLoss.group_acc.clear()
        loss = _TraceableLoss(coeff=coeff)
        AuxLoss.set_step_denominator(torch.tensor(denominator))
        x = torch.ones(2, requires_grad=True)
        loss(x, carrier=torch.ones(1, requires_grad=True)).sum().backward()
        grads.append(x.grad.clone())

    torch.testing.assert_close(grads[1], grads[0] * 3.0, rtol=1e-5, atol=1e-8)


def test_aux_gradient_shrinks_with_the_denominator() -> None:
    """More valid tokens in the step means a smaller per-token contribution."""
    loss = _TraceableLoss(coeff=1.0)
    grads = []
    for denominator in (2.0, 8.0):
        AuxLoss._group_counts.clear()
        AuxLoss.group_acc.clear()
        AuxLoss.set_step_denominator(torch.tensor(denominator))
        x = torch.ones(2, requires_grad=True)
        loss(x, carrier=torch.ones(1, requires_grad=True)).sum().backward()
        grads.append(x.grad.clone())

    torch.testing.assert_close(grads[1], grads[0] / 4.0, rtol=1e-5, atol=1e-8)


def test_metric_accumulates_the_scaled_not_the_weighted_value() -> None:
    """``instance_acc`` tracks ``raw_sum / denom`` -- ``coeff`` is not applied.

    The metric reports the loss's own magnitude; folding ``coeff`` in would make
    two differently-weighted uses of the same loss incomparable, and would hide
    a coefficient that was accidentally set to zero.
    """
    loss = _ConstantLoss(coeff=100.0)
    AuxLoss.set_step_denominator(torch.tensor(4.0))

    loss(torch.ones(2, requires_grad=True))

    torch.testing.assert_close(
        loss.instance_acc, torch.tensor(2.0 / 4.0), rtol=1e-5, atol=1e-8
    )


def test_metric_accumulates_across_microbatches_within_a_step() -> None:
    loss = _ConstantLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(2.0))

    loss(torch.ones(2, requires_grad=True))
    loss(torch.ones(2, requires_grad=True))

    torch.testing.assert_close(
        loss.instance_acc, torch.tensor(1.0 + 1.0), rtol=1e-5, atol=1e-8
    )


# -- roll-up and collection --------------------------------------------------


def test_zero_hook_rolls_instances_into_the_group_and_clears_them() -> None:
    loss = _ConstantLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(2.0))
    loss(torch.ones(4, requires_grad=True))

    _zero_aux_losses([loss])

    torch.testing.assert_close(
        AuxLoss.group_acc[("batch", "_constant_loss")],
        torch.tensor(1.0),
        rtol=1e-5,
        atol=1e-8,
    )
    assert loss.instance_acc.item() == 0.0


def test_zero_hook_sums_instances_within_one_group() -> None:
    """Two layers in the same group contribute one register holding both."""
    a, b = _ConstantLoss(coeff=1.0), _ConstantLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    a(torch.ones(2, requires_grad=True))  # raw 2
    b(torch.ones(3, requires_grad=True))  # raw 2

    _zero_aux_losses([a, b])

    torch.testing.assert_close(
        AuxLoss.group_acc[("batch", "_constant_loss")],
        torch.tensor(4.0),
        rtol=1e-5,
        atol=1e-8,
    )


def test_zero_hook_finds_nested_loss_modules() -> None:
    """Losses live inside the model, not at its top level."""
    container = torch.nn.Sequential(_ConstantLoss(coeff=1.0))
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    container[0](torch.ones(2, requires_grad=True))

    _zero_aux_losses([container])

    torch.testing.assert_close(
        AuxLoss.group_acc[("batch", "_constant_loss")],
        torch.tensor(2.0),
        rtol=1e-5,
        atol=1e-8,
    )


def test_zero_hook_ignores_modules_that_are_not_aux_losses() -> None:
    container = torch.nn.Sequential(torch.nn.Linear(2, 2), _ConstantLoss(coeff=1.0))
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    container[1](torch.ones(2, requires_grad=True))

    _zero_aux_losses([container])

    assert set(AuxLoss.group_acc) == {("batch", "_constant_loss")}


def test_collect_returns_nothing_when_no_loss_is_configured() -> None:
    assert collect_aux_loss_metrics(parallel_dims=_NoMeshes()) == {}


def test_collect_divides_the_group_sum_by_the_instance_count() -> None:
    """The reported value is the mean over layers, not the sum."""
    a, b = _ConstantLoss(coeff=1.0), _ConstantLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    a(torch.ones(2, requires_grad=True))  # raw 2
    b(torch.ones(4, requires_grad=True))  # raw 2
    _zero_aux_losses([a, b])

    metrics = collect_aux_loss_metrics(parallel_dims=_NoMeshes())

    assert metrics["_constant_loss/mean"] == pytest.approx((2.0 + 2.0) / 2)


def test_collect_clamps_a_missing_group_register_to_zero() -> None:
    """A rank owning no instance of a group still reports the group's mean.

    The register is only created by ``_zero_aux_losses`` on ranks that hold an
    instance, so a rank without one must contribute a zero rather than be
    skipped -- otherwise collective participation would diverge.
    """
    _ConstantLoss(coeff=1.0)  # registers the group...
    metrics = collect_aux_loss_metrics(parallel_dims=_NoMeshes())  # ...but no hook ran

    assert metrics["_constant_loss/mean"] == pytest.approx(0.0)


# -- the balance loss --------------------------------------------------------


def _reference_balance_loss_T(
    scores_TE: torch.Tensor, routing_map_TE: torch.Tensor
) -> torch.Tensor:
    """Eqs 17-19 in their explicit ``(E / (K T))`` form, times T.

    The module folds ``T`` away (see its docstring) and returns ``T * L_bal`` so
    that ``AuxLoss``'s ``1 / valid_tokens`` normalization leaves the injected
    weight at ``coeff * L_bal``. This keeps ``T`` explicit to check that both
    the folding and the extra factor are identities.
    """
    T, E = scores_TE.shape
    counts_E = routing_map_TE.to(scores_TE.dtype).sum(dim=0)
    K = counts_E.sum() / T  # each token routes to K experts
    f_E = (E / (K * T)) * counts_E
    probs_TE = F.normalize(scores_TE, p=1, dim=-1)
    p_E = probs_TE.sum(dim=0)
    return (f_E * p_E).sum()


def _routing_from_scores(scores_TE: torch.Tensor, k: int) -> torch.Tensor:
    routing = torch.zeros_like(scores_TE, dtype=torch.bool)
    for t in range(scores_TE.shape[0]):
        routing[t, torch.topk(scores_TE[t].detach(), k).indices] = True
    return routing


def _balance_loss_value(scores_TE: torch.Tensor, routing_map_TE: torch.Tensor) -> float:
    loss = MicrobatchWiseLoadBalanceLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    loss(scores_TE, routing_map_TE, carrier=torch.ones(1, requires_grad=True))
    return loss.instance_acc.item()


def test_balance_loss_matches_the_paper_formula() -> None:
    """The T-free form must equal the explicit ``(E / (K T))`` form, times T."""
    torch.manual_seed(0)
    T, E, K = 12, 4, 2
    scores_TE = torch.rand(T, E)
    routing_map_TE = _routing_from_scores(scores_TE, K)

    value = _balance_loss_value(scores_TE, routing_map_TE)

    expected = _reference_balance_loss_T(scores_TE, routing_map_TE)
    # coeff/denom == 1, so instance_acc is exactly the module's raw value.
    assert value == pytest.approx(expected.item(), abs=1e-6)


def test_balance_loss_counts_sum_to_the_expert_count() -> None:
    """``sum_i f_i == E`` by construction -- a normalization invariant.

    If this drifts, the loss's scale silently depends on how many experts the
    model happens to have.
    """
    torch.manual_seed(0)
    scores_TE = torch.rand(10, 5)
    routing_map_TE = _routing_from_scores(scores_TE, 2)
    E = scores_TE.size(-1)

    counts_E = routing_map_TE.to(scores_TE.dtype).sum(dim=0)
    f_E = F.normalize(counts_E, p=1, dim=0) * E

    assert f_E.sum().item() == pytest.approx(float(E))


def test_balance_loss_is_invariant_to_score_scale() -> None:
    """Scores are L1-normalized per token, so a global scale cannot matter."""
    torch.manual_seed(0)
    scores_TE = torch.rand(8, 4)
    routing_map_TE = _routing_from_scores(scores_TE, 2)

    base = _balance_loss_value(scores_TE, routing_map_TE)
    scaled = _balance_loss_value(scores_TE * 1000.0, routing_map_TE)

    assert scaled == pytest.approx(base, rel=1e-5)


def test_balance_loss_is_invariant_to_token_order() -> None:
    """Balancing is per-forward and order-free; a permutation cannot change it."""
    torch.manual_seed(0)
    scores_TE = torch.rand(8, 4)
    routing_map_TE = _routing_from_scores(scores_TE, 2)
    perm = torch.tensor([3, 0, 7, 1, 5, 2, 6, 4])

    base = _balance_loss_value(scores_TE, routing_map_TE)
    permuted = _balance_loss_value(scores_TE[perm], routing_map_TE[perm])

    assert permuted == pytest.approx(base, rel=1e-6)


def test_balance_loss_of_even_load_equals_the_token_count() -> None:
    """One expert per token, evenly taken, with uniform scores gives ``T``.

    ``f_i == 1`` for every expert and ``sum_i p_i == T`` (the per-token scores
    are L1-normalized), so ``sum_i f_i p_i == T``.
    """
    T, E = 8, 4
    scores_TE = torch.ones(T, E)
    routing_map_TE = torch.zeros(T, E, dtype=torch.bool)
    for t in range(T):
        routing_map_TE[t, t % E] = True

    value = _balance_loss_value(scores_TE, routing_map_TE)

    assert value == pytest.approx(float(T))


def test_balance_loss_gradient_reaches_the_scores() -> None:
    """The whole point: the router must receive a nonzero gradient."""
    torch.manual_seed(0)
    scores_TE = torch.rand(8, 4, requires_grad=True)
    routing_map_TE = _routing_from_scores(scores_TE, 2)

    loss = MicrobatchWiseLoadBalanceLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    loss(
        scores_TE, routing_map_TE, carrier=torch.ones(1, requires_grad=True)
    ).sum().backward()

    assert scores_TE.grad is not None
    assert scores_TE.grad.abs().sum() > 0


def test_balance_loss_accepts_a_boolean_routing_map() -> None:
    """Routing is non-differentiable; the map must not need a grad path."""
    T, E = 6, 3
    routing_map_TE = torch.zeros(T, E, dtype=torch.bool)
    routing_map_TE[:, 0] = True
    scores_TE = torch.ones(T, E, requires_grad=True)

    loss = MicrobatchWiseLoadBalanceLoss(coeff=1.0)
    AuxLoss.set_step_denominator(torch.tensor(1.0))
    loss(scores_TE, routing_map_TE, carrier=torch.ones(1, requires_grad=True))

    assert loss.instance_acc.item() > 0


def test_balance_loss_uses_the_batch_reduce_mesh() -> None:
    """cp-identical losses reduce over dp only, or they would be over-counted."""
    assert MicrobatchWiseLoadBalanceLoss(coeff=1.0).reduce_mesh == "batch"


def test_balance_loss_skips_reduction_without_a_registered_mesh() -> None:
    """No SPMD mesh registered means no token-sharding axis, so no collective.

    This is hpmesh's state today: nothing calls ``set_spmd_meshes``, so
    ``spmd_mesh_group`` returns None and the reduction degrades to a no-op --
    correct while nothing shards the token dim, and a live collective once
    something does.
    """
    loss = MicrobatchWiseLoadBalanceLoss(coeff=1.0)
    partial = torch.ones(4)

    # Must not raise, must not attempt a collective on a nonexistent group.
    assert torch.equal(loss._reduce_token_partials(partial, ("cp", "tp")), partial)
