"""
Train the slimmable / in-place-distillation MoE super-net. One weight-shared
net; the big width is the in-net teacher (data NLL), narrower widths distill
in-place from it (NLL + forward Gaussian-KL). Yields the best model at every
width AND the NLL-vs-capacity curve in one run.

Usage:
  python train_slimmable.py --out runs/slimmable_moe
"""
import argparse, os, sys, json, time, traceback

THIS = os.path.dirname(os.path.abspath(__file__))
HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model
from slimmable_moe import SlimmableMoE, DEFAULT_WIDTHS, slice_params
from fair_fit import AbortOnStuckData

TEACHER_PER_EVENT = -8.7525


def best_checkpoint(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher-model-type', default='ViT_Max')   # only for the TFRecord pipeline
    p.add_argument('--thresholds-json',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/vit_max_run_1000ep/optimized_thresholds.json")
    p.add_argument('--dataset',
                   default="/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lam-kd', type=float, default=1.0)
    p.add_argument('--clipnorm', type=float, default=1.0)
    p.add_argument('--attach-vit', action='store_true',
                   help='attach frozen ViT_Max as the apex teacher distilling into the big slice')
    p.add_argument('--teacher-checkpoints',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints")
    p.add_argument('--lam-vit', type=float, default=0.3)
    p.add_argument('--vit-warmup', type=int, default=200)
    p.add_argument('--vit-tau', type=float, default=2.0, help='soften ViT covariance (tau^2) for KD stability')
    p.add_argument('--widths-preset', default='full', choices=['full', 'small', 'big'],
                   help='full=1K..152K (over-constrains), small=1K/2.3K/4.8K, big=4.8K/23K/68K')
    p.add_argument('--quantize', action='store_true',
                   help='8-bit QAT (quantized_bits(8,0) weights, quantized_tanh(8) acts) -- deployable')
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'train.log'), 'a', buffering=1)
    stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()
    PRESETS = {
        'full':  DEFAULT_WIDTHS,
        'small': [(2, 8, 8, 6), (3, 12, 12, 8), (4, 16, 16, 16)],          # 1K, 2.3K, 4.8K
        'big':   [(4, 16, 16, 16), (8, 32, 32, 16), (12, 48, 48, 16)],     # 4.8K, 23K, 68K
    }
    widths = PRESETS[args.widths_preset]

    try:
        tf.random.set_seed(args.seed); np.random.seed(args.seed)
        stamp(f"slimmable MoE: widths={widths} lam_kd={args.lam_kd} clipnorm={args.clipnorm}")
        thr = json.load(open(args.thresholds_json))
        thresholds = np.array(thr['thresholds'], dtype=np.float32)
        levels = np.array(thr['levels'], dtype=np.float32)
        _, _, tfr_tr, tfr_val = generate_tfrecords(
            dataset_dir=args.dataset, model_type=args.teacher_model_type,
            train_batch_size=5000, val_batch_size=5000,
            select_contained=False, timeslices=2, tfrecords_exist=True, seed=args.seed)
        tg, vg = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                                digitize_levels=levels, digitize_thresholds=thresholds, seed=args.seed)
        stamp("TFRecords loaded")

        teacher = None
        if args.attach_vit:
            teacher = create_model('ViT_Max', timeslices=2, soft_quantize_layer=False)
            teacher.load_weights(best_checkpoint(args.teacher_checkpoints))
            teacher.trainable = False
            stamp(f"apex ViT_Max attached: params={teacher.count_params()} lam_vit={args.lam_vit} warmup={args.vit_warmup}")

        model = SlimmableMoE(widths=widths, lam_kd=args.lam_kd,
                             teacher=teacher, lam_vit=args.lam_vit, vit_warmup_steps=args.vit_warmup,
                             vit_tau=args.vit_tau, quantize=args.quantize)
        stamp(f"quantize(8-bit QAT)={args.quantize}")
        model.build((None, 16, 16, 2))
        stamp(f"super-net params={model.count_params()}  per-width eff params={[slice_params(*w) for w in widths]}")
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=args.lr, clipnorm=args.clipnorm))

        # checkpoint every epoch + auto-resume, so a GPU preemption doesn't lose
        # progress (these long runs keep getting preempted on the shared GPU).
        ckpt = os.path.join(args.out, 'ckpt.weights.h5')
        resumed = os.path.exists(ckpt)
        if resumed:
            model.load_weights(ckpt)
            stamp(f"RESUMED from checkpoint {ckpt}")
        callbacks = [
            tf.keras.callbacks.CSVLogger(os.path.join(args.out, 'history.csv'), append=resumed),
            tf.keras.callbacks.TerminateOnNaN(),
            AbortOnStuckData(threshold=10.0, patience=8),
            tf.keras.callbacks.ModelCheckpoint(ckpt, save_weights_only=True, save_freq='epoch'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss_data', patience=args.patience,
                                             restore_best_weights=True, verbose=1),
        ]
        stamp(f"starting fit: epochs={args.epochs} patience={args.patience}")
        h = model.fit(tg, validation_data=vg, epochs=args.epochs,
                      callbacks=callbacks, shuffle=False, verbose=1)
        stamp("fit complete")

        json.dump({k: [float(x) for x in v] for k, v in h.history.items()},
                  open(os.path.join(args.out, 'history.json'), 'w'), indent=1)
        model.save_weights(os.path.join(args.out, 'slimmable.weights.h5'))
        curve = {}
        for i, w in enumerate(widths):
            key = f'val_nll_w{i}'
            best = float(min(h.history.get(key, [float('inf')])))
            curve[f'w{i}'] = {'config': w, 'eff_params': slice_params(*w),
                              'best_val_per_event': best, 'best_val_per_batch': best * 5000.0,
                              'gap_to_teacher': best - TEACHER_PER_EVENT}
        summary = {
            'method': 'slimmable_inplace_distillation', 'widths': widths,
            'lam_kd': args.lam_kd, 'super_net_params': int(model.count_params()),
            'epochs_run': len(h.history.get('nll_w0', [])),
            'capacity_curve': curve,
            'deploy_slice_best_per_event': curve['w0']['best_val_per_event'],
            'standalone_tiny_per_event': -6.163,
            'teacher_per_event': TEACHER_PER_EVENT,
        }
        json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
        stamp(f"summary: {json.dumps(summary)}")

    except Exception:
        traceback.print_exc()
        with open(os.path.join(args.out, 'FAILED.txt'), 'w') as f:
            f.write(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
