"""Train the conditional flow-matching stroke blender (temporal U-Net, endpoint-anchored deviation).

Consumes ``extract_stroke_flow`` output (``checkpoints/flow_data/stroke_flow.npz``: x [M,T,7] time-domain
strokes, c [M,S,7] arc-length paths) and fits :class:`~tiptop.networks.flow_timing.TrajFlow` with
conditional flow matching on the endpoint-anchored deviation ``delta = x - tau*e`` (e = the stroke's net
displacement / endpoint), conditioned on the timing-stripped path ``c`` and the endpoint ``e``. Writes
``--out`` (state_dict + meta with the data scales) for :class:`FlowModel`; omitting it writes
``FlowModel``'s default checkpoint, which is the SHIPPED ``checkpoints/flow_net_t64.pt`` -- pass an
explicit ``--out`` unless you mean to replace it.

The CFM loss does NOT go to zero (the regression target has irreducible variance); judge the model by its
SAMPLES (``eval_flow.py``), not the loss value.

Run in the openpi venv::

    openpi/.venv/bin/python -m tiptop.networks.train_flow --epochs 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_timing import (  # noqa: E402
    DEFAULT_CKPT, DOF, KIND_NULL, N_KINDS, S_ARC, T_TIME, TrajFlow, augment_path, boundary_weights,
    cfm_loss,
)

MERGED = Path(__file__).resolve().parents[1] / "checkpoints" / "flow_data" / "stroke_flow.npz"


def train(args):
    if not MERGED.exists():
        raise FileNotFoundError(f"{MERGED} not found. Run extract_stroke_flow first.")
    blob = np.load(MERGED)
    x = blob["x"].astype(np.float32)  # [M,T,7]  start-centered positions over time
    c = blob["c"].astype(np.float32)  # [M,S,7]  start-centered path over arc length
    if "k" not in blob:
        raise KeyError(f"{MERGED} has no 'k' event-kind labels; re-run extract_stroke_flow.")
    k = blob["k"].astype(np.int64)    # [M,2]    (start, end) gripper-event kinds
    m, T, dof = x.shape
    e = x[:, -1, :]                                   # [M,7] endpoint = net displacement
    tau = np.linspace(0.0, 1.0, T, dtype=np.float32)[None, :, None]
    delta = x - tau * e[:, None, :]                  # delta[:,0]=delta[:,-1]=0 by construction
    delta_std = float(np.sqrt((delta ** 2).mean())) or 1.0
    c_std = float(np.sqrt((c ** 2).mean())) or 1.0
    e_std = float(np.sqrt((e ** 2).mean())) or 1.0
    d_t = torch.as_tensor(delta / delta_std)
    c_t = torch.as_tensor(c / c_std)
    e_t = torch.as_tensor(e / e_std)
    k_t = torch.as_tensor(k)
    c_aug_t = None
    if args.path_aug > 0:
        rng = np.random.default_rng(0)
        aug = np.empty_like(c)
        for i in range(m):
            aug[i] = augment_path(c[i], rng.uniform(0.0, args.aug_sigma),
                                  rng.uniform(0.0, args.aug_lambda))
        c_aug_t = torch.as_tensor(aug / c_std)
        print(f"path augmentation: p={args.path_aug} sigma~U(0,{args.aug_sigma}) "
              f"lambda~U(0,{args.aug_lambda})")
    print(f"Loaded {m} strokes.  delta_std={delta_std:.3f} c_std={c_std:.3f} e_std={e_std:.3f}")
    print("  end-kind counts: " + str({n: int((k[:, 1] == v).sum()) for n, v in
                                       (("none", 0), ("close", 1), ("open", 2))}))

    device = args.device
    n_kinds = 0 if args.no_kinds else N_KINDS
    net = TrajFlow(dof, args.cond_dim, args.t_dim, args.base_ch, n_kinds, args.k_dim).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"TrajFlow U-Net: {n_params/1e6:.2f}M params")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = boundary_weights(T, args.boundary_weight).to(device)
    print(f"boundary loss weight: strength={args.boundary_weight} "
          f"(ends x{float(w[0]) / float(w[T // 2]):.1f} vs mid)")

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(m, generator=g)
    n_val = max(args.batch_size, int(0.05 * m))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    def batches(idx, shuffle):
        order = idx[torch.randperm(len(idx), generator=g)] if shuffle else idx
        for k in range(0, len(order) - 1, args.batch_size):
            yield order[k : k + args.batch_size]

    best = float("inf")
    if not args.out:
        out_path = DEFAULT_CKPT
    else:
        p = Path(args.out)
        out_path = p if p.is_absolute() else (Path(__file__).resolve().parents[1] / p)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        net.train()
        tr = []
        for bi in batches(tr_i, True):
            cb = c_t[bi]
            if c_aug_t is not None:
                # Per-sample coin flip between the real path and its planner-like variant.
                use = torch.rand(len(bi)) < args.path_aug
                cb = torch.where(use[:, None, None], c_aug_t[bi], cb)
            kb = None
            if n_kinds:
                kb = k_t[bi].to(device)
                if args.kind_dropout > 0:
                    # Drop the whole label to the null slot so the net also learns the unconditioned
                    # field, which classifier-free guidance needs at sampling time.
                    drop = torch.rand(len(kb), device=device) < args.kind_dropout
                    kb = torch.where(drop[:, None], torch.full_like(kb, KIND_NULL), kb)
            loss = cfm_loss(net, d_t[bi].to(device), cb.to(device), e_t[bi].to(device), kb, w)
            opt.zero_grad(); loss.backward(); opt.step()
            tr.append(float(loss))
        net.eval()
        with torch.no_grad():
            kv = k_t[val_i].to(device) if n_kinds else None
            va = np.mean([float(cfm_loss(net, d_t[val_i].to(device), c_t[val_i].to(device),
                                         e_t[val_i].to(device), kv, w)) for _ in range(4)])
        tag = ""
        if va < best:
            best = va
            meta = {"t_time": T_TIME, "s_arc": S_ARC, "dof": DOF, "c_std": c_std, "delta_std": delta_std,
                    "e_std": e_std, "cond_dim": args.cond_dim, "t_dim": args.t_dim, "base_ch": args.base_ch,
                    "n_kinds": n_kinds, "k_dim": args.k_dim, "guidance": args.guidance,
                    "kind_dropout": args.kind_dropout, "path_aug": args.path_aug,
                    "aug_sigma": args.aug_sigma, "aug_lambda": args.aug_lambda,
                    "boundary_weight": args.boundary_weight}
            torch.save({"state_dict": net.state_dict(), "meta": meta}, out_path)
            tag = "  *saved"
        if epoch % max(1, args.epochs // 25) == 0 or tag:
            print(f"epoch {epoch:4d}  train {np.mean(tr):.5f}  val {va:.5f}{tag}", flush=True)

    print(f"\nBest val CFM {best:.5f}. Checkpoint -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--cond-dim", type=int, default=128)
    ap.add_argument("--t-dim", type=int, default=64)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--k-dim", type=int, default=16, help="Embedding width per gripper-event kind.")
    ap.add_argument("--boundary-weight", type=float, default=3.0,
                    help="Extra loss weight on the stroke ends (0 = uniform).")
    ap.add_argument("--kind-dropout", type=float, default=0.15,
                    help="Probability of replacing the event label with the null slot (enables CFG).")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="Default CFG scale baked into the checkpoint for FlowModel to use.")
    ap.add_argument("--path-aug", type=float, default=0.5,
                    help="Probability of conditioning on a smoothed/straightened path instead.")
    ap.add_argument("--aug-sigma", type=float, default=4.0, help="Max smoothing sigma (samples).")
    ap.add_argument("--aug-lambda", type=float, default=0.6, help="Max pull toward the chord.")
    ap.add_argument("--no-kinds", action="store_true",
                    help="Ablation: drop the gripper-event conditioning (same data, same schedule).")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None)
    train(ap.parse_args())
