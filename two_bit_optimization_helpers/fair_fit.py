"""
Fair training harness for the distillation experiments, matching the regime
that produced the standalone -30,526 baseline.

Two confounds were found in the earlier distillation comparison:
  1. The distill driver used Adam + clipnorm=1.0 with EarlyStopping(patience=50),
     which stopped ~epoch 128 and converged to a worse optimum (~-23k). The
     standalone used Nadam, no clipnorm, and ran to ~epoch 420 to reach -30,526.
  2. Nadam (the better optimizer here) can get pinned at the NLL clip ceiling
     (val_loss_data ~ +20.7/event) on a bad random init -- the known "Part-2
     stuck at ~1e5" mode. The standalone escaped it with an AbortOnStuck +
     retry-with-new-seed loop.

This module gives every distillation run the SAME safety net: Nadam, no clipnorm,
abort-on-stuck, and retry with a fresh seed until the student escapes the
ceiling. Then w=0 reproduces the standalone and TBR/DeiT become honest tests.
"""
import numpy as np
import tensorflow as tf


class AbortOnStuckData(tf.keras.callbacks.Callback):
    """Abort if val_loss_data (per-event NLL) stays above `threshold` for
    `patience` consecutive epochs -- the student is pinned at the clip ceiling
    (~+20.7/event) and this init will not recover."""

    def __init__(self, threshold=10.0, patience=5):
        super().__init__()
        self.thr = threshold
        self.pat = patience
        self.bad = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        v = logs.get("val_loss_data", np.inf)
        if v > self.thr or not np.isfinite(v):
            self.bad += 1
            if self.bad >= self.pat:
                print(f"[AbortOnStuck] val_loss_data {v:.2f} > {self.thr} for "
                      f"{self.pat} epochs -- bad init, aborting to retry.")
                self.model.stop_training = True
        else:
            self.bad = 0


def fit_with_retry(make_distiller, tg, vg, out_dir, stamp,
                   epochs=1000, patience=50, lr=1e-3,
                   base_seed=42, max_attempts=6,
                   stuck_threshold=10.0, stuck_patience=5):
    """Build + fit a distiller with Nadam (no clipnorm) and retry on stuck init.

    make_distiller(seed) -> a fresh, UNCOMPILED tf.keras.Model distiller whose
    test_step emits 'val_loss_data'. This function compiles it (Nadam) and fits.
    Returns (distiller, history, seed_used). Raises if all attempts get stuck.
    """
    import os, tensorflow as tf
    for attempt in range(max_attempts):
        seed = base_seed + 1000 * attempt
        tf.random.set_seed(seed)
        np.random.seed(seed)
        import random as _random
        _random.seed(seed)
        stamp(f"[attempt {attempt}] seed={seed}: building + compiling (Nadam, no clipnorm)")
        distiller = make_distiller(seed)
        distiller.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=lr))
        callbacks = [
            tf.keras.callbacks.CSVLogger(os.path.join(out_dir, 'history.csv'), append=False),
            tf.keras.callbacks.TerminateOnNaN(),
            AbortOnStuckData(threshold=stuck_threshold, patience=stuck_patience),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss_data', patience=patience,
                restore_best_weights=True, verbose=1),
        ]
        h = distiller.fit(tg, validation_data=vg, epochs=epochs,
                          callbacks=callbacks, shuffle=False, verbose=1)
        best = float(min(h.history.get('val_loss_data', [np.inf])))
        if best < 0.0:                      # escaped the ceiling and learned
            stamp(f"[attempt {attempt}] escaped: best val_loss_data={best:.4f}")
            return distiller, h, seed
        stamp(f"[attempt {attempt}] STUCK (best={best:.2f}); retrying with new seed")
    raise RuntimeError(f"all {max_attempts} attempts got stuck at the clip ceiling")
