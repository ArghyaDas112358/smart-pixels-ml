"""
E2: DeiT-style separate-head distillation of QMlp_MoE_Max_Distill (18 outputs)
from the extracted ViT_Max teacher. Main head fits ground truth (means + cov)
via NLL; a separate distill head fits the teacher means via MSE; the deployed
prediction averages the two mean heads. See distill14_deit.py.

Usage:
  python train_distill_deit.py --w 0.5 --out runs/distill_qmlp_moe_deit_w0p5
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
from distill14_deit import Distiller14DeiT
from fair_fit import fit_with_retry


def best_checkpoint(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--student-model-type', default='QMlp_MoE_Max_Distill')
    p.add_argument('--teacher-checkpoints',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints")
    p.add_argument('--teacher-model-type', default='ViT_Max')
    p.add_argument('--thresholds-json',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/vit_max_run_1000ep/optimized_thresholds.json")
    p.add_argument('--dataset',
                   default="/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--optimizer', default='nadam', choices=['nadam', 'adam'],
                   help='nadam (matches the standalone -30,526 trainer) or adam')
    p.add_argument('--clipnorm', type=float, default=0.0,
                   help='gradient clipnorm; 0 = none (the standalone trainer uses none)')
    p.add_argument('--w', type=float, default=0.5, help='fixed weight on the distill-head MSE term')
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'train.log'), 'a', buffering=1)
    stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()

    try:
        stamp(f"E2 DeiT: student={args.student_model_type}  teacher={args.teacher_model_type}  w={args.w}")
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

        def make_distiller(seed):
            student = create_model(args.student_model_type, timeslices=2, soft_quantize_layer=False)
            return Distiller14DeiT(student, teacher, w=args.w)

        stamp(f"starting fair fit (Nadam, retry-on-stuck): epochs={args.epochs}  patience={args.patience}")
        distiller, h, seed_used = fit_with_retry(
            make_distiller, tg, vg, args.out, stamp,
            epochs=args.epochs, patience=args.patience, lr=args.lr, base_seed=args.seed)
        student = distiller.student
        stamp(f"fit complete (seed_used={seed_used})")

        json.dump({k: [float(x) for x in v] for k, v in h.history.items()},
                  open(os.path.join(args.out, 'history.json'), 'w'), indent=1)
        student.save_weights(os.path.join(args.out, 'student_final.weights.h5'))
        best_deployed = float(min(h.history.get('val_loss_data', [float('inf')])))
        best_main = float(min(h.history.get('val_main_only', [float('inf')])))
        summary = {
            'method': 'DeiT_separate_head',
            'student_model_type': args.student_model_type,
            'teacher_model_type': args.teacher_model_type,
            'w': args.w, 'seed_used': seed_used,
            'student_params': int(student.count_params()),
            'teacher_params': int(teacher.count_params()),
            'epochs_run': len(h.history['loss_data']),
            'best_val_deployed_per_event': best_deployed,
            'best_val_deployed_per_batch': best_deployed * 5000.0,
            'best_val_main_only_per_event': best_main,
            'best_val_main_only_per_batch': best_main * 5000.0,
            'standalone_baseline_per_batch': -30525.97,
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
