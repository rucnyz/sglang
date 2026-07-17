# archive_path_a — historical eval docs from the VMM-remap actuator era

These three documents were written when the inter-pool actuator was
the VMM `cuMemUnmap` / `cuMemMap` remap path (now called path A in
`dev/v1_logical_partition/design.md`). After the path-B pivot (logical
partition with boot-time max allocation), most of the content here is
historical:

- `SETTINGS.md` — the paper-eval plan tied to path-A's expected
  behavior. Settings 1-5 + 6 ablations assume the VMM actuator works.
- `RESULTS.md` — append-only log of eval runs, including the L2
  debugging chain (mobile-soft bug, drain protocol crashes, gate-retune
  sweeps). All of this is path-A artefacts.
- `BLOCKERS.md` — append-only log of blockers. Most are resolved or
  superseded by path B.

Kept here for paper-narrative reference and to preserve the search
history of what was tried. Not authoritative for the current
direction — see `dev/v1_logical_partition/` for that.

Current eval setup will be written fresh after the path-B actuator
integration; this archive is purely historical from this point.
