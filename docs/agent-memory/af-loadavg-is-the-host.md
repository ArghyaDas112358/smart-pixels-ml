---
name: af-loadavg-is-the-host
description: On the Purdue AF, /proc/loadavg reports the physical Geddes node, not your pod — use cgroup cpu.stat to judge contention
metadata:
  type: reference
---

Inside a Purdue AF session, `/proc/loadavg` and `uptime` report the **physical
Geddes host**, shared with every other tenant's pod — not your container. A load
average of ~390 on a 128-core session is normal and says nothing about whether
your jobs are starved.

The authoritative check is the cgroup:

```
cat /sys/fs/cgroup/cpu.stat        # nr_throttled / throttled_time
cat /sys/fs/cgroup/cpu.max         # your quota
ps -eo user,%cpu --no-headers | awk '{s[$1]+=$2} END {for (u in s) print u, s[u]}'
```

On 2026-08-29 this read `nr_throttled 0, throttled_time 0` over 8,156,992
periods while `uptime` showed 391 — i.e. never throttled, ever. Summed process
CPU was ~1200% (12 of 128 cores). Two other tells: `/proc/loadavg` field 4
showed 33,635 system-wide processes (far more than any one pod), and the 1/5/15
minute figures sat at 391/393/393 for hours — real contention fluctuates, a
host-wide average seen from inside a small pod does not.

**Why it matters:** I spent several messages telling the user their trainers were
being starved by a peer session's 96-worker job, and a peer session
independently blamed itself for an uncapped-process mistake. Neither was true.
The actual slowdown (26 s -> 38.5 s per epoch) was **two trainers sharing one
A100** — a self-inflicted, deliberate choice. GPU sharing is real on this box;
CPU contention essentially is not, since sessions get a hard core quota.

**How to apply:** never cite `uptime`/loadavg as evidence of contention on the
AF. Check `cpu.stat` for CPU and count your own concurrent GPU jobs for GPU.
Related: [[smartpixels-fresh-env]], [[act-on-ops-decisions]].
