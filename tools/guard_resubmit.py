"""Install a RESUBMIT GUARD in the self-resubmitting Gautschi sbatch chains.

Failure this fixes (observed twice, seeds 22042 and 24142): the chunk prints
"resubmitting ..." and calls sbatch, but sbatch is REFUSED (QOS job cap) and the
script ignores its exit status -- so the chain silently ends and the seed sits
dead for hours while its siblings advance. Retry a few times, and if it still
fails, say so loudly in the log instead of dying quietly.
"""
import re, sys, os

for p in sys.argv[1:]:
    if not os.path.exists(p):
        print(f"  {p}: missing -- skipped"); continue
    s = open(p).read()
    if "RESUBMIT GUARD" in s:
        print(f"  {p}: already guarded"); continue
    m = re.search(r'^([ \t]*)cd "\$REPO" && ((?:\w+=\S+ )*)sbatch (\S+)\s*$', s, re.M)
    if not m:
        print(f"  {p}: resubmit line not matched -- skipped"); continue
    ind, vars_, script = m.group(1), m.group(2), m.group(3)
    new = (
        f'{ind}# RESUBMIT GUARD: sbatch can be REFUSED (QOS job cap). The old line\n'
        f'{ind}# ignored its exit status, so the chain ended silently -- seeds 22042\n'
        f'{ind}# and 24142 each sat dead for hours that way. Retry, then shout.\n'
        f'{ind}for _try in 1 2 3 4 5; do\n'
        f'{ind}    if cd "$REPO" && {vars_}sbatch {script}; then\n'
        f'{ind}        echo "resubmit ok (attempt $_try)"; break\n'
        f'{ind}    fi\n'
        f'{ind}    echo "RESUBMIT FAILED (attempt $_try) -- retrying in 120s"\n'
        f'{ind}    if [ "$_try" = 5 ]; then echo "RESUBMIT GUARD: GIVING UP -- CHAIN DEAD"; else sleep 120; fi\n'
        f'{ind}done'
    )
    open(p, "w").write(s.replace(m.group(0), new))
    print(f"  {p}: guard installed")
