"""
Warm-start KL refinement: load a CONVERGED standalone student (good means +
sane covariance), then fine-tune with a gentle, gradient-CLIPPED forward-KL
toward the teacher. This is the safe way to actually test distillation here --
the from-scratch forward-KL exploded (1e25) because a random student's
covariance is near-singular; a warm-started student cannot explode.

  total = data_NLL + beta * KL[teacher || student]   (beta small, clipnorm on)

Honest test: does pulling a well-trained student toward the teacher's full
distribution lower its NLL below the standalone it started from?

Usage:
  python train_distill_warmstart.py --student-model-type QMlp_MoE_M \\
    --init-weights runs/standalone_moe_M/student_final.weights.h5 \\
    --fixed-beta 0.1 --clipnorm 1.0 --out runs/warmstart_moe_M_b0p1
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
from distill14 import Distiller14
from fair_fit import AbortOnStuckData

TEACHER_PER_EVENT = -8.7525


def best_checkpoint(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--student-model-type', required=True,
                   choices=['QMlp_MoE_Max', 'QMlp_MoE_M', 'QMlp_MoE_L', 'QMlp_MoE_XL', 'QMlp_MoE_ConvStem'])
    p.add_argument('--init-weights', required=True, help='converged standalone student weights to warm-start from')
    p.add_argument('--teacher-checkpoints',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints")
    p.add_argument('--teacher-model-type', default='ViT_Max')
    p.add_argument('--thresholds-json',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/vit_max_run_1000ep/optimized_thresholds.json")
    p.add_argument('--dataset',
                   default="/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=3e-4, help='lower LR for fine-tuning a converged model')
    p.add_argument('--fixed-beta', type=float, default=0.1)
    p.add_argument('--clipnorm', type=float, default=1.0, help='tames KL spikes; >0 strongly recommended')
    p.add_argument('--kl-reverse', action='store_true')
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'train.log'), 'a', buffering=1)
    stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()

    try:
        tf.random.set_seed(args.seed); np.random.seed(args.seed)
        stamp(f"warmstart: student={args.student_model_type}  beta={args.fixed_beta}  "
              f"clipnorm={args.clipnorm}  reverse={args.kl_reverse}  lr={args.lr}")
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

        teacher = create_model(args.teacher_model_type, timeslices=2, soft_quantize_layer=False)
        teacher.load_weights(best_checkpoint(args.teacher_checkpoints))
        teacher.trainable = False

        student = create_model(args.student_model_type, timeslices=2, soft_quantize_layer=False)
        student.load_weights(args.init_weights)
        stamp(f"warm-started student params={student.count_params()} from {args.init_weights}")

        distiller = Distiller14(student, teacher, fixed_beta=args.fixed_beta,
                                warmup_steps=0, kl_reverse=args.kl_reverse)
        opt = tf.keras.optimizers.Nadam(learning_rate=args.lr, clipnorm=args.clipnorm)
        distiller.compile(optimizer=opt)

        callbacks = [
            tf.keras.callbacks.CSVLogger(os.path.join(args.out, 'history.csv'), append=False),
            tf.keras.callbacks.TerminateOnNaN(),
            AbortOnStuckData(threshold=10.0, patience=5),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss_data', patience=args.patience,
                                             restore_best_weights=True, verbose=1),
        ]
        stamp(f"starting warm-start fit: epochs={args.epochs} patience={args.patience}")
        h = distiller.fit(tg, validation_data=vg, epochs=args.epochs,
                          callbacks=callbacks, shuffle=False, verbose=1)
        stamp("fit complete")

        json.dump({k: [float(x) for x in v] for k, v in h.history.items()},
                  open(os.path.join(args.out, 'history.json'), 'w'), indent=1)
        student.save_weights(os.path.join(args.out, 'student_final.weights.h5'))
        best_val = float(min(h.history.get('val_loss_data', [float('inf')])))
        init_baseline = None
        try:
            init_dir = os.path.dirname(args.init_weights)
            init_baseline = float(json.load(open(os.path.join(init_dir, 'summary.json')))['best_val_loss_data'])
        except Exception:
            pass
        summary = {
            'method': 'warmstart_forward_KL' if not args.kl_reverse else 'warmstart_reverse_KL',
            'student_model_type': args.student_model_type,
            'init_weights': args.init_weights,
            'fixed_beta': args.fixed_beta, 'clipnorm': args.clipnorm, 'lr': args.lr,
            'student_params': int(student.count_params()),
            'epochs_run': len(h.history['loss_data']),
            'best_val_loss_data': best_val,
            'best_val_per_batch': best_val * 5000.0,
            'standalone_init_per_event': init_baseline,
            'gain_vs_standalone_per_event': (init_baseline - best_val) if init_baseline is not None else None,
            'best_val_kl': float(min(h.history.get('val_kl', [float('inf')]))),
            'teacher_target_per_event': TEACHER_PER_EVENT,
            'gap_to_teacher_per_event': best_val - TEACHER_PER_EVENT,
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
