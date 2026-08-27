"""
Physics-informed PySR for the residual (teacher - barycenter) on each of the
4 outputs (x, y, cot_alpha, cot_beta).

Adds, over the vanilla pysr_residuals.py:
  - 24-feature extractor (charge moments, time-resolved centroids, geometric
    cot priors, edge flags, teacher-prior xbary/ybary).
  - Sigma-weighted Huber loss via PySR elementwise_loss (sigma from the
    teacher's predicted marginal std per event).
  - Per-output operator restrictions: positions get polynomial-only,
    angles get tanh+sigmoid+square.
  - Nested-depth + per-operator complexity constraints.
  - Dimensional analysis with X_units / y_units.
  - Optional TemplateExpressionSpec to bake physics topology in.
  - Higher niters/maxsize/populations.

Usage (smoke):
  python pysr_residuals_physics.py --n-events 200 --niters 4 --maxsize 12 \
    --out runs/pysr_results_physics_smoke

Usage (full ~80 min):
  python pysr_residuals_physics.py --n-events 8000 --niters 80 --maxsize 25 \
    --out runs/pysr_results_physics
"""
import argparse, os, sys, json
import numpy as np

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import tensorflow as tf
from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model
from symbolic import PhysicsAnsatz, GEOMETRY

P_X = GEOMETRY['p_x']        # 50 um
P_Y = GEOMETRY['p_y']        # 12.5 um
T_SENSOR = GEOMETRY['T']     # 100 um
N = GEOMETRY['N']            # 16


FEATURE_NAMES = [
    "xbary", "ybary",                                  # 1-2  um
    "wx", "wy",                                        # 3-4  pixel
    "cota_geom", "cotb_geom",                          # 5-6  1
    "q_tot", "q_max", "q_frac_max",                    # 7-9
    "qFx", "qLx", "qFy", "qLy",                        # 10-13 electron
    "rx", "ry",                                        # 14-15 1
    "skew_x", "skew_y", "kurt_x", "kurt_y",            # 16-19 1
    "tasym",                                           # 20    1
    "tasym_x", "tasym_y",                              # 21-22 um
    "edge_x", "edge_y",                                # 23-24 1
]
# Use "C" as a custom unit token for electron count (PySR accepts any string;
# only consistency matters).
FEATURE_UNITS = [
    "um", "um",
    "1", "1",
    "1", "1",
    "C", "C", "1",
    "C", "C", "C", "C",
    "1", "1",
    "1", "1", "1", "1",
    "1",
    "um", "um",
    "1", "1",
]
Y_UNITS_PER_OUTPUT = ["um", "um", "1", "1"]
OUTPUT_NAMES = ["x", "y", "cot_alpha", "cot_beta"]


def _pixel_centers(n, pitch):
    return (np.arange(n, dtype=np.float32) - (n - 1) / 2.0) * pitch


def _profile_moments(profile, centers):
    """Return (centroid, std, skew, kurt) for a batched 1D profile."""
    w = profile.sum(axis=1) + 1e-9
    c = (profile * centers).sum(axis=1) / w
    diff = centers[None, :] - c[:, None]
    var = (profile * diff ** 2).sum(axis=1) / w
    std = np.sqrt(var + 1e-9)
    skew = (profile * diff ** 3).sum(axis=1) / w / (std ** 3 + 1e-9)
    kurt = (profile * diff ** 4).sum(axis=1) / w / (std ** 4 + 1e-9) - 3.0
    return c, std, skew, kurt


def cluster_features_physics(charge):
    """charge: (B, 16, 16, 2). Returns (B, 24) float32."""
    q = charge.sum(axis=-1)                       # (B, 16, 16)
    prof_x = q.sum(axis=2)                        # (B, 16)
    prof_y = q.sum(axis=1)                        # (B, 16)
    active_x = (prof_x > 0).astype(np.float32)
    active_y = (prof_y > 0).astype(np.float32)

    wx = active_x.sum(axis=1)
    wy = active_y.sum(axis=1)
    cota_geom = np.maximum(wx - 1.0, 0.0) * P_X / T_SENSOR
    cotb_geom = np.maximum(wy - 1.0, 0.0) * P_Y / T_SENSOR

    q_tot = q.sum(axis=(1, 2))
    q_max = q.max(axis=(1, 2))
    q_frac_max = q_max / (q_tot + 1e-9)

    def head_tail(p, act):
        first = np.argmax(act, axis=1)
        last = (p.shape[1] - 1) - np.argmax(act[:, ::-1], axis=1)
        qF = p[np.arange(p.shape[0]), first]
        qL = p[np.arange(p.shape[0]), last]
        return qF, qL, first, last
    qFx, qLx, first_x, last_x = head_tail(prof_x, active_x)
    qFy, qLy, first_y, last_y = head_tail(prof_y, active_y)
    rx = (qLx - qFx) / (qLx + qFx + 1e-9)
    ry = (qLy - qFy) / (qLy + qFy + 1e-9)

    x_centers = _pixel_centers(N, P_X)
    y_centers = _pixel_centers(N, P_Y)
    xbary, _, skew_x, kurt_x = _profile_moments(prof_x, x_centers)
    ybary, _, skew_y, kurt_y = _profile_moments(prof_y, y_centers)

    q_t0 = charge[..., 0].sum(axis=(1, 2))
    q_t1 = charge[..., 1].sum(axis=(1, 2))
    tasym = (q_t1 - q_t0) / (q_t1 + q_t0 + 1e-9)

    prof_x_t0 = charge[..., 0].sum(axis=2); prof_x_t1 = charge[..., 1].sum(axis=2)
    prof_y_t0 = charge[..., 0].sum(axis=1); prof_y_t1 = charge[..., 1].sum(axis=1)
    cx_t0 = (prof_x_t0 * x_centers).sum(axis=1) / (prof_x_t0.sum(axis=1) + 1e-9)
    cx_t1 = (prof_x_t1 * x_centers).sum(axis=1) / (prof_x_t1.sum(axis=1) + 1e-9)
    cy_t0 = (prof_y_t0 * y_centers).sum(axis=1) / (prof_y_t0.sum(axis=1) + 1e-9)
    cy_t1 = (prof_y_t1 * y_centers).sum(axis=1) / (prof_y_t1.sum(axis=1) + 1e-9)
    tasym_x = cx_t1 - cx_t0
    tasym_y = cy_t1 - cy_t0

    edge_x = ((first_x == 0) | (last_x == N - 1)).astype(np.float32)
    edge_y = ((first_y == 0) | (last_y == N - 1)).astype(np.float32)

    F = np.stack([
        xbary, ybary,
        wx, wy,
        cota_geom, cotb_geom,
        q_tot, q_max, q_frac_max,
        qFx, qLx, qFy, qLy,
        rx, ry,
        skew_x, skew_y, kurt_x, kurt_y,
        tasym,
        tasym_x, tasym_y,
        edge_x, edge_y,
    ], axis=1).astype(np.float32)
    assert F.shape[1] == len(FEATURE_NAMES) == len(FEATURE_UNITS) == 24
    return F


PHYSICS_LOSS = """
function physics_loss(prediction, target, weight)
    delta = prediction - target
    a = abs(delta)
    return weight * (a < 1.0 ? delta^2 : 2.0*a - 1.0)
end
"""


def per_output_pysr_kwargs(out_name, niters, maxsize, populations,
                           random_state, use_template):
    """Return dict of PySRRegressor kwargs for a given output name."""
    common = dict(
        niterations=niters,
        maxsize=maxsize,
        maxdepth=8,
        populations=populations,
        population_size=33,
        parsimony=0.0032,
        weight_optimize=0.001,
        optimizer_iterations=20,
        progress=False,
        verbosity=0,
        random_state=random_state,
        elementwise_loss=PHYSICS_LOSS,
        dimensional_constraint_penalty=1000.0,
    )
    if out_name in ("x", "y"):
        common.update(
            binary_operators=["+", "-", "*"],
            unary_operators=["square"],
            nested_constraints={"square": {"square": 0}},
            constraints={"*": (8, 8)},
            complexity_of_operators={"*": 2, "square": 3},
        )
    else:  # cot_alpha, cot_beta
        common.update(
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[
                "tanh",
                "sigmoid(x) = 1 / (1 + exp(-x))",
                "square",
            ],
            extra_sympy_mappings={"sigmoid": lambda x: 1 / (1 + np.exp(-x))},
            nested_constraints={
                "tanh": {"tanh": 0, "sigmoid": 0},
                "sigmoid": {"tanh": 0, "sigmoid": 0},
                "/": {"/": 0},
            },
            constraints={"/": (6, 6), "tanh": 8, "sigmoid": 8},
            complexity_of_operators={"/": 3, "tanh": 2, "sigmoid": 2, "square": 3},
        )

    if use_template:
        try:
            from pysr import TemplateExpressionSpec
        except Exception:
            return common
        # Templates per output. Variable_names argument is the FLAT
        # feature list; the `combine` string references variables by name.
        if out_name == "x":
            spec = TemplateExpressionSpec(
                expressions=["base", "lorentz", "edge"],
                variable_names=FEATURE_NAMES,
                combine=("base(rx, skew_x, kurt_x, q_frac_max, wx) "
                         "+ lorentz(tasym, tasym_x, cota_geom) "
                         "+ edge(edge_x, wx)"),
            )
        elif out_name == "y":
            spec = TemplateExpressionSpec(
                expressions=["base", "lorentz", "edge"],
                variable_names=FEATURE_NAMES,
                combine=("base(ry, skew_y, kurt_y, q_frac_max, wy) "
                         "+ lorentz(tasym, tasym_y, cotb_geom) "
                         "+ edge(edge_y, wy)"),
            )
        elif out_name == "cot_alpha":
            spec = TemplateExpressionSpec(
                expressions=["sat", "asym"],
                variable_names=FEATURE_NAMES,
                combine=("sat(cota_geom, wx, q_frac_max) "
                         "+ asym(rx, tasym)"),
            )
        elif out_name == "cot_beta":
            spec = TemplateExpressionSpec(
                expressions=["sat", "asym"],
                variable_names=FEATURE_NAMES,
                combine=("sat(cotb_geom, wy, q_frac_max) "
                         "+ asym(ry, tasym_y)"),
            )
        common["expression_spec"] = spec
    return common


def best_checkpoint(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher-checkpoints',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints")
    p.add_argument('--teacher-model-type', default="ViT_Max")
    p.add_argument('--thresholds-json',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/vit_max_run_1000ep/optimized_thresholds.json")
    p.add_argument('--dataset',
                   default="/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    p.add_argument('--n-events', type=int, default=8000)
    p.add_argument('--niters', type=int, default=80)
    p.add_argument('--maxsize', type=int, default=25)
    p.add_argument('--populations', type=int, default=20)
    p.add_argument('--no-template', action='store_true',
                   help="disable TemplateExpressionSpec (use free-form search)")
    p.add_argument('--no-units', action='store_true',
                   help="disable dimensional analysis")
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    thr = json.load(open(args.thresholds_json))
    thresholds = np.array(thr['thresholds'], dtype=np.float32)
    levels = np.array(thr['levels'], dtype=np.float32)

    _, _, tfr_tr, tfr_val = generate_tfrecords(
        dataset_dir=args.dataset, model_type=args.teacher_model_type,
        train_batch_size=5000, val_batch_size=5000,
        select_contained=False, timeslices=2,
        tfrecords_exist=True, seed=args.seed,
    )
    _, vg = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                           digitize_levels=levels, digitize_thresholds=thresholds,
                           seed=args.seed)

    teacher = create_model(args.teacher_model_type, timeslices=2,
                           soft_quantize_layer=False)
    teacher.load_weights(best_checkpoint(args.teacher_checkpoints))
    teacher.trainable = False
    print(f"teacher loaded: params={teacher.count_params()}", flush=True)

    # Plumb labels_scale into the ansatz so its output is in the same
    # label-normalized units as the teacher's means; otherwise the residual
    # collapses to (label - physical_um) which is nonsense for PySR to fit.
    meta = json.load(open(os.path.join(tfr_tr, 'metadata.json')))
    labels_scale = meta['labels_scale']
    print(f"labels_scale from metadata: {labels_scale}", flush=True)
    ansatz = PhysicsAnsatz(variant='barycenter', labels_scale=labels_scale)

    Fs, Rs, Ws = [], [], []
    total = 0
    for batch in vg:
        x, _ = batch
        x = x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
        t14 = teacher(x, training=False).numpy()
        sym = ansatz(x).numpy()
        s_a = np.sign(t14[:, 4]); s_a[s_a == 0] = 1.0
        s_b = np.sign(t14[:, 6]); s_b[s_b == 0] = 1.0
        sym_signed = sym.copy()
        sym_signed[:, 2] *= s_a
        sym_signed[:, 3] *= s_b

        teacher_means = t14[:, 0:8:2]  # (B, 4)
        residual = teacher_means - sym_signed  # (B, 4)

        # Per-event sigma per output (softplus on raw diag, +1e-9 floor)
        raw_diag = t14[:, 1:8:2]  # (B, 4)
        sigma = np.log1p(np.exp(np.clip(raw_diag, -30, 30))) + 1e-9
        w = 1.0 / (sigma ** 2)

        feats = cluster_features_physics(x)
        Fs.append(feats); Rs.append(residual); Ws.append(w)
        total += x.shape[0]
        if total >= args.n_events:
            break

    F = np.concatenate(Fs, axis=0)[:args.n_events]
    R = np.concatenate(Rs, axis=0)[:args.n_events]
    W = np.concatenate(Ws, axis=0)[:args.n_events]

    # Clip per-output weights to [1st, 99th] percentile
    for o in range(4):
        lo, hi = np.percentile(W[:, o], [1, 99])
        W[:, o] = np.clip(W[:, o], lo, hi)

    print(f"collected {F.shape[0]} events", flush=True)
    print(f"feature stats (min, mean, max):", flush=True)
    for i, nm in enumerate(FEATURE_NAMES):
        col = F[:, i]
        print(f"  {nm:12s}  {col.min():+.3e}  {col.mean():+.3e}  {col.max():+.3e}", flush=True)
    print(f"residual |max| per output: {np.abs(R).max(axis=0).tolist()}", flush=True)

    assert np.isfinite(F).all(), "non-finite feature"
    assert np.isfinite(R).all(), "non-finite residual"
    assert np.isfinite(W).all() and (W > 0).all(), "bad sigma weights"

    np.savez(os.path.join(args.out, 'features_residuals.npz'),
             features=F, residuals=R, weights=W,
             feature_names=np.array(FEATURE_NAMES, dtype=object),
             feature_units=np.array(FEATURE_UNITS, dtype=object),
             output_names=np.array(OUTPUT_NAMES, dtype=object))

    # PySR imports are lazy (Julia precompile happens on first PySRRegressor())
    from pysr import PySRRegressor

    discovered = {}
    pareto_lines = ["# PySR physics-informed: top-3 per output\n"]
    for i, out_name in enumerate(OUTPUT_NAMES):
        sub_out = os.path.join(args.out, out_name)
        os.makedirs(sub_out, exist_ok=True)
        print(f"\n=== PySR for {out_name} ===", flush=True)
        kwargs = per_output_pysr_kwargs(
            out_name, niters=args.niters, maxsize=args.maxsize,
            populations=args.populations, random_state=args.seed,
            use_template=not args.no_template,
        )
        kwargs['output_directory'] = sub_out
        if not args.no_units:
            kwargs['dimensional_constraint_penalty'] = 1000.0
        else:
            kwargs.pop('dimensional_constraint_penalty', None)
        try:
            model = PySRRegressor(**kwargs)
        except TypeError as e:
            print(f"  PySR kwargs rejected ({e}); falling back to no-template/no-units", flush=True)
            kwargs.pop('expression_spec', None)
            kwargs.pop('dimensional_constraint_penalty', None)
            model = PySRRegressor(**kwargs)
        fit_kwargs = dict(variable_names=FEATURE_NAMES, weights=W[:, i])
        if not args.no_units:
            fit_kwargs['X_units'] = FEATURE_UNITS
            fit_kwargs['y_units'] = Y_UNITS_PER_OUTPUT[i]
        try:
            model.fit(F, R[:, i], **fit_kwargs)
        except Exception as e:
            print(f"  fit with units failed ({e}); retrying without units", flush=True)
            fit_kwargs.pop('X_units', None); fit_kwargs.pop('y_units', None)
            model.fit(F, R[:, i], **fit_kwargs)
        eq = model.get_best()
        discovered[out_name] = {
            'equation': str(eq['equation']),
            'loss': float(eq['loss']),
            'complexity': int(eq['complexity']),
        }
        print(f"  best: {discovered[out_name]['equation']}  "
              f"loss={discovered[out_name]['loss']:.4g}  "
              f"complexity={discovered[out_name]['complexity']}", flush=True)

        # Pareto top-3 by score = loss * exp(0.01 * complexity)
        try:
            hof = model.equations_
            hof = hof.sort_values('loss').head(3)
            pareto_lines.append(f"\n## {out_name}\n")
            for _, row in hof.iterrows():
                pareto_lines.append(
                    f"- loss={row['loss']:.4g}  complexity={row['complexity']}  "
                    f"`{row['equation']}`\n")
        except Exception:
            pass

    json.dump(discovered, open(os.path.join(args.out, 'discovered_equations.json'), 'w'),
              indent=1)
    open(os.path.join(args.out, 'pareto_summary.md'), 'w').writelines(pareto_lines)
    print("\nDONE. Discovered:")
    for k, v in discovered.items():
        print(f"  {k}: {v['equation']}  (loss={v['loss']:.4g})")


if __name__ == '__main__':
    main()
