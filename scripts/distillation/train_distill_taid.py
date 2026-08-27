"""
Distill a QConv2D_Max (or Conv2D_Max) student from the extracted ViT_Max
teacher under the same `L_data + lambda * KL[T||S]` objective with MDMM.

Skips Part 1 (thresholds already fixed from the 1000-epoch teacher run).
Uses the Part-2-style data pipeline: hard-digitized 2-bit input.

Usage:
  python train_distill_qconv2d.py --student-model-type QConv2D_Max \\
    --out runs/distill_qconv2d_max
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
from distill14_taid import Distiller14TAID


def best_checkpoint(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--student-model-type', default='QConv2D_Max',
                   choices=['QConv2D_Max', 'Conv2D_Max', 'QMlp_Max', 'QMlp_MoE_Max', 'QConv1D_Full', 'QMlp_Full'])
    p.add_argument('--teacher-checkpoints',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints")
    p.add_argument('--teacher-model-type', default='ViT_Max')
    p.add_argument('--thresholds-json',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/vit_max_run_1000ep/optimized_thresholds.json")
    p.add_argument('--dataset',
                   default="/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1.0, help='weight on the TAID KD term')
    p.add_argument('--anneal-steps', type=float, default=20000, help='steps to anneal alpha 0->1')
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'train.log'), 'a', buffering=1)
    stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()

    try:
        stamp(f"student={args.student_model_type}  teacher={args.teacher_model_type}")
        thr = json.load(open(args.thresholds_json))
        thresholds = np.array(thr['thresholds'], dtype=np.float32)
        levels = np.array(thr['levels'], dtype=np.float32)

        _, _, tfr_tr, tfr_val = generate_tfrecords(
            dataset_dir=args.dataset, model_type=args.teacher_model_type,
            train_batch_size=5000, val_batch_size=5000,
            select_contained=False, timeslices=2,
            tfrecords_exist=True, seed=args.seed,
        )
        tg, vg = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                                digitize_levels=levels, digitize_thresholds=thresholds,
                                seed=args.seed)
        stamp("TFRecords loaded")

        teacher = create_model(args.teacher_model_type, timeslices=2, soft_quantize_layer=False)
        teacher.load_weights(best_checkpoint(args.teacher_checkpoints))
        teacher.trainable = False
        stamp(f"teacher params={teacher.count_params()}")

        student = create_model(args.student_model_type, timeslices=2, soft_quantize_layer=False)
        stamp(f"student params={student.count_params()}")

        distiller = Distiller14TAID(student, teacher,
                                    beta=args.beta, anneal_steps=args.anneal_steps)
        distiller.compile(optimizer=tf.keras.optimizers.Adam(args.lr, clipnorm=1.0))

        callbacks = [
            tf.keras.callbacks.CSVLogger(os.path.join(args.out, 'history.csv'), append=False),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss_data', patience=args.patience,
                restore_best_weights=True, verbose=1),
        ]
        stamp(f"starting fit: epochs={args.epochs}  patience={args.patience}")
        h = distiller.fit(tg, validation_data=vg, epochs=args.epochs,
                          callbacks=callbacks, shuffle=False, verbose=1)
        stamp("fit complete")

        # Save artifacts
        json.dump({k: [float(x) for x in v] for k, v in h.history.items()},
                  open(os.path.join(args.out, 'history.json'), 'w'), indent=1)
        student.save_weights(os.path.join(args.out, 'student_final.weights.h5'))
        summary = {
            'student_model_type': args.student_model_type,
            'teacher_model_type': args.teacher_model_type,
            'teacher_checkpoints': args.teacher_checkpoints,
            'student_params': int(student.count_params()),
            'teacher_params': int(teacher.count_params()),
            'epochs_run': len(h.history['loss_data']),
            'final_train_loss_data': float(h.history['loss_data'][-1]),
            'best_val_loss_data': float(min(h.history.get('val_loss_data', [float('inf')]))),
            'final_train_kl': float(h.history['kl'][-1]),
            'best_val_kl': float(min(h.history.get('val_kl', [float('inf')]))),
            'beta': float(args.beta),
            'anneal_steps': float(args.anneal_steps),
            'method': 'TAID interpolated target',
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
