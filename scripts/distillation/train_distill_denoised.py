"""
Warm-start DENOISED-MEAN distillation (Agent-1 main lever):
  L = NLL(y) + gamma*Huber(mu_S-mu_T) + beta*KL[T(tau)||S]
Load converged standalone weights, then refine. Nadam + clipnorm (KL safe).

Usage:
  python train_distill_denoised.py --student-model-type QMlp_MoE_ConvStem \\
    --init-weights runs/standalone_moe_convstem/student_final.weights.h5 \\
    --gamma 2.0 --beta 0.3 --tau 2.0 --out runs/denoised_convstem
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
from distill14_denoised import Distiller14Denoised
from fair_fit import AbortOnStuckData

TEACHER_PER_EVENT = -8.7525


def best_checkpoint(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--student-model-type', required=True)
    p.add_argument('--init-weights', required=True)
    p.add_argument('--teacher-checkpoints',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints")
    p.add_argument('--teacher-model-type', default='ViT_Max')
    p.add_argument('--thresholds-json',
                   default="/work/users/das214/SmartPixels/smart-pixels-ml/runs/vit_max_run_1000ep/optimized_thresholds.json")
    p.add_argument('--dataset',
                   default="/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=2.0, help='denoised-mean Huber weight')
    p.add_argument('--beta', type=float, default=0.3, help='temperature-softened forward-KL weight')
    p.add_argument('--tau', type=float, default=2.0, help='teacher covariance temperature (Sigma_T->tau^2 Sigma_T)')
    p.add_argument('--huber-delta', type=float, default=1.0)
    p.add_argument('--teacher-bounded', action='store_true')
    p.add_argument('--clipnorm', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'train.log'), 'a', buffering=1)
    stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()

    try:
        tf.random.set_seed(args.seed); np.random.seed(args.seed)
        stamp(f"denoised: student={args.student_model_type} gamma={args.gamma} beta={args.beta} "
              f"tau={args.tau} tbounded={args.teacher_bounded} clipnorm={args.clipnorm} lr={args.lr}")
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

        distiller = Distiller14Denoised(student, teacher, gamma=args.gamma, beta=args.beta,
                                        tau=args.tau, huber_delta=args.huber_delta,
                                        teacher_bounded=args.teacher_bounded)
        distiller.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=args.lr, clipnorm=args.clipnorm))

        callbacks = [
            tf.keras.callbacks.CSVLogger(os.path.join(args.out, 'history.csv'), append=False),
            tf.keras.callbacks.TerminateOnNaN(),
            AbortOnStuckData(threshold=10.0, patience=5),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss_data', patience=args.patience,
                                             restore_best_weights=True, verbose=1),
        ]
        stamp(f"starting denoised fit: epochs={args.epochs} patience={args.patience}")
        h = distiller.fit(tg, validation_data=vg, epochs=args.epochs,
                          callbacks=callbacks, shuffle=False, verbose=1)
        stamp("fit complete")

        json.dump({k: [float(x) for x in v] for k, v in h.history.items()},
                  open(os.path.join(args.out, 'history.json'), 'w'), indent=1)
        student.save_weights(os.path.join(args.out, 'student_final.weights.h5'))
        best_val = float(min(h.history.get('val_loss_data', [float('inf')])))
        init_baseline = None
        try:
            init_baseline = float(json.load(open(os.path.join(os.path.dirname(args.init_weights),
                                                               'summary.json')))['best_val_loss_data'])
        except Exception:
            pass
        summary = {
            'method': 'warmstart_denoised_mean',
            'student_model_type': args.student_model_type, 'init_weights': args.init_weights,
            'gamma': args.gamma, 'beta': args.beta, 'tau': args.tau,
            'teacher_bounded': args.teacher_bounded, 'lr': args.lr, 'clipnorm': args.clipnorm,
            'student_params': int(student.count_params()),
            'epochs_run': len(h.history['loss_data']),
            'best_val_loss_data': best_val, 'best_val_per_batch': best_val * 5000.0,
            'standalone_init_per_event': init_baseline,
            'gain_vs_standalone_per_event': (init_baseline - best_val) if init_baseline is not None else None,
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
