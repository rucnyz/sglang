# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Teacher-forcing token-override bookkeeping.

Splits the pure, model-free half of token forcing out of the scheduler so it is
unit-testable without booting a worker. The GPU scatter that consumes these
positions lives in `Scheduler._apply_forced_tokens`.
"""

from typing import List, Tuple


def forced_override_positions(reqs) -> List[Tuple[int, int]]:
    """For each req that commits a token THIS batch (not finished, not retracted,
    `inflight_middle_chunks <= 0`, mirroring the commit-time filter in the
    batch-result processor) and carries `custom_params["forced_output_ids"]`
    with `forced_dispatched < len(forced)`, return `(batch_index, forced_token)`
    and advance that req's `forced_dispatched` counter.

    Pure bookkeeping over req state (no tensor ops). The counter advances at
    DISPATCH, not at commit, because under `--enable-overlap` the commit
    (`output_ids.append`) lags ~1 step behind dispatch, so `len(req.output_ids)`
    would index the wrong forced position.
    """
    out: List[Tuple[int, int]] = []
    for i, req in enumerate(reqs):
        if req.finished() or req.is_retracted or req.inflight_middle_chunks > 0:
            continue
        cp = req.sampling_params.custom_params
        forced = cp.get("forced_output_ids") if cp else None
        if not forced:
            continue
        pos = req.forced_dispatched
        if pos < len(forced):
            out.append((i, int(forced[pos])))
            req.forced_dispatched = pos + 1
    return out
