"""Make ARM=c (the UCB bandit) a no-op in the cold sbatch.

User decision 2026-08-20: only the pair-lattice is being pursued. The bandit
arm was holding roughly a third of the smallgpu per-user submit limit
(QOSMaxSubmitJobPerUserLimit), which starved the lattice chains -- three of
them died silently in one night because their resubmit was refused. Exiting
immediately frees the slot and stops the arm re-chaining.

Reversible: delete the guard block to resume the bandit.
"""
import sys

p = "scripts/gautschi/submit_o21v2_cold.sbatch"
s = open(p).read()
if "BANDIT RETIRED" in s:
    print("  already retired"); sys.exit(0)
anchor = 'SEED=${SEED:?set SEED=<n> when submitting}'
if anchor not in s:
    print("  anchor not found -- NOT modified"); sys.exit(1)
guard = anchor + '''

# BANDIT RETIRED (user decision 2026-08-20): only the pair-lattice arm is being
# pursued. Arm c was consuming a third of the smallgpu per-user submit limit,
# starving the lattice chains (three died silently in one night when their
# resubmit was refused). Exit at once so any queued arm-c chunk frees its slot
# and stops re-chaining. Delete this block to resume the bandit.
if [ "$ARM" = "c" ]; then
    echo "arm c (bandit) retired -- exiting without training"
    exit 0
fi'''
open(p, "w").write(s.replace(anchor, guard, 1))
print("  guard installed")
