# Task 143 — Golden vectors for the vendored simulation contract

> Written by `plan_dsrc_rec`: every decision below was taken by recommendation,
> without putting the question to the user. Section 3's decisions table says what
> was chosen and why; **none of it is user-approved**. Section 11 holds every
> point a question would have exposed, including the places where the code
> contradicted the brief. This plan is the fixed target a validator audits the
> implementation against.

> **Tree state.** `HEAD` is `975b7a2` ("Record tasks 142-145: four claims the
> repository does not support"), working tree clean, `sha256(git diff)` is the
> empty-input digest `e3b0c44298fc1c14`. `HEAD` moved once while this plan was
> being written (from `98032dc`); every line number below was re-resolved at
> `975b7a2` after the move. The plan itself wrote nothing to the repository
> outside this file. Re-resolve line numbers before trusting them; nothing in
> this plan depends on one, and every count in it was measured by running code.

## The short version

**The claim.** The paper plan (`plans/paper_deploying_self_regulating_cars.md:536`)
names contract vendoring as "how the device cannot drift from what was trained".
`deployment/jetson/policy/sim_contract.py` carries a numpy copy of the sim's
observation encoder, scales, action heads and bin decoders, taken from sim commit
`d477dba`. The file that is supposed to check the copy against the original is
`deployment/jetson/tests/test_sim_contract.py`.

**The defect, measured.** That file opens with
`pytest.importorskip("src.rl.encoders")` (`:18`). `src/rl/encoders.py`,
`actions.py` and `models.py` were deleted in `6b538f2`, so the import fails and
the whole module is skipped at collection:

    .venv/bin/python -m pytest deployment/jetson/tests/test_sim_contract.py -q -p no:cacheprovider
    1 skipped in 0.03s

**How much is dormant.** 25 tests. Extracting the sim at `d477dba` into a
directory with `git archive` and putting it ahead of the working tree on
`sys.path` makes the same file run: **25 passed in 0.80 s, none failed.** So the
vendored contract is correct today; what is missing is anything that would notice
if it stopped being correct.

**The measurement that decides the approach.** The reference is not gone. Commit
`d477dba` is reachable from both `main` and `origin/main`; `git archive d477dba src`
yields 47 Python files (328 KB); and every module the test needs —
`src.rl.encoders`, `src.rl.actions`, `src.envs.wrappers`, `src.rl.models`,
`src.sensing.local` — imports cleanly under the current `.venv` (torch 2.13.0,
numpy 2.5.2, highway_env 1.12.1 all present). The reference is therefore available
to a **generator run once**, without restoring a single file to the working tree.

**What gets built.** Three artifacts, following the idiom the repository already
uses for the same problem across Python and Kotlin
(`specs/transport_golden_frames.json`):

1. `specs/sim_contract_golden_vectors.json` — frozen vectors: 14 encoded
   observations recorded slot by slot with the derivation of each number, the
   ordered slot-name list, `FIELD_SCALES`, the action heads and values, the two
   bin decoders over a grid, `bin_index` over a grid, the neutral fallbacks, the
   actor state-dict layout, and the contract fingerprint.
2. `scripts/generate_sim_contract_golden_vectors.py` — derives every recorded
   value **twice**, once from the `d477dba` reference and once from
   `sim_contract.py`, and refuses to write when the two disagree.
3. A rewritten `deployment/jetson/tests/test_sim_contract.py` that reads the
   frozen file and imports no simulation module, so the check runs on every
   machine instead of no machine.

**The rule that stops the file from being blessed.** A golden file whose
generator is run whenever the test fails proves nothing. Three separate
mechanisms, in Section 5: the generator writes nothing unless the vendored
contract agrees with the `d477dba` reference; the file will not be overwritten
without `--force`; and the fingerprint is pinned as a string literal typed in the
test source, which a regeneration cannot rewrite.

**The gate.** Section 7. The new check is not trusted until it has been seen to
fail against six named mutations, each caught by a named test, with a negative
control. Two of the six exist specifically to prove the golden vectors cover
ground `contract_fingerprint()` does not.

### Scope boundary — what this task does not do

- **Changing the contract.** `sim_contract.py`'s values are frozen as they are.
  If the generator reports a disagreement with `d477dba`, that is a finding to
  report, not a value to edit.
- **Re-exporting policy bundles.** No `.ts` or `.json` bundle is produced,
  re-exported or moved. `export_policy.py` is read, not modified.
- **Which policy the device runs.** That is task 145. This task says nothing
  about trained versus random weights, checkpoint selection or bundle provenance.
- **Restoring the deleted sim modules to the working tree.** The `git archive`
  reference lives in a temporary directory and is deleted after use.
- **`specs/observation_schema.md` and `specs/action_schema.md`.** Read as the
  source for the neutral fallbacks; not edited.

### Open decisions, taken by recommendation and flagged (full text in Section 11)

| # | What a question would have asked | Taken |
|---|---|---|
| O1 | Delete the 25 sim-import tests, or re-point them at the archive reference | Delete; the generator absorbs them |
| O2 | Keep a torch-requiring golden test for the actor state-dict layout | Keep, isolated to one test |
| O3 | `FIELD_SCALES` membership is checked in one direction only | Fix as part of this task |
| O4 | `remutate.py`'s `EXPECTED_PYTHON_TESTCASES` is stale (2060 against a measured 2272), so every Python mutation returns INCONCLUSIVE | Correct it here, because it blocks this task's gate |
| O5 | ARCHITECTURE §6 and the test docstring both say the action tests need `highway_env`; at `d477dba` they do not | Correct the text |
| O6 | The golden file records `SIM_COMMIT`, which the fingerprint deliberately excludes | Record it as metadata, assert it, do not hash it |

---

## Section 1 — What is unenforced today

Every number in this section was produced by running the command shown.

| Measurement | Command | Result |
|---|---|---|
| The contract check | `pytest deployment/jetson/tests/test_sim_contract.py -q -p no:cacheprovider` | `1 skipped in 0.03s` |
| Tests in that file | `pytest --co` against a copy with the reference on `sys.path` | 25 collected |
| Those tests against the reference | same, run | `25 passed in 0.80s`, 0 failed |
| Whole Jetson suite | `pytest -q deployment/jetson/tests/ -p no:cacheprovider` | `2247 passed, 25 skipped` in 76.84 s |
| JUnit testcases in that run | parse of `results.xml` | 2272 |
| Breakdown of the 25 skips | parse of `results.xml` | 24 are USB-device-gated (`test_transport_acceptor_contract`, `test_transport_backend_contract`, reason "no USB device attached"); **1 is `test_sim_contract.py`'s module-level collection skip** |

So the sim-contract module skip is the only skip in the Jetson suite that is not
explained by absent hardware. After this task the suite's skip count should be 24
on a machine with no handset attached.

**What the module-level `importorskip` costs beyond the sim comparisons.** The
skip is at module scope (`:18`), so it removes eight tests that need no
simulation module at all: `test_inf_encoding_known_values` (`:136`), the three in
`TestTheBundleGuardSeesMoreThanTheDimension` (`:148`) and the four in
`TestEncodedSlotNames` (`:187`). Those eight already run entirely against
`policy.sim_contract` and are dormant only because of where the skip sits.

**What `contract_fingerprint` covers, and what it does not.**
`actor_runtime.py:95-101` reads `contract_fingerprint` from the bundle manifest
and raises when it differs from `sim_contract.contract_fingerprint()`.
`sim_contract.py:253-277` builds that hash from exactly two things: the ordered
list `LOCAL_OBS_FIELDS`, and `FIELD_SCALES` sorted by key. Measured value today:
`918ec57cf2f2e1db`.

It therefore moves on a reordered field and on a changed scale — the two failures
its docstring names. It does **not** move on any of the following, each of which
changes what the device computes:

| Change | Fingerprint moves? | Currently caught by anything? |
|---|---|---|
| `LOCAL_OBS_FIELDS` reordered | yes | yes, `actor_runtime` refuses the bundle |
| a `FIELD_SCALES` value changed | yes | yes |
| `COOPERATION_FIELDS` reordered | **no** | no |
| `LANE_DISTRIBUTION_LANES` reordered or extended | **no** | no |
| `_number`'s `inf` clamp constant (200.0) changed | **no** | no |
| `_number`'s bool branch removed | **no** | no |
| `SPEED_BIN_OFFSETS_MPS` or `HEADWAY_BIN_S` values changed | **no** | no |
| `ACTION_VALUES` order changed within a head | **no** | no |
| `bin_index`'s `>=` changed to `>` | **no** | no |

Two further limits of the existing guard, both read directly from the code:

- `actor_runtime.py:95` uses `self.manifest.get("contract_fingerprint")` and
  accepts `None`. A bundle exported before the field existed passes
  unconditionally. The comment at `:91-94` states this is deliberate; it is
  recorded here because it means the fingerprint constrains only bundles that
  carry one.
- The fingerprint compares a bundle against the contract. It cannot compare the
  contract against the simulation, which is the thing task 143 is about.

## Section 2 — The reference still exists

This is the measurement that settles Decision D1, so it is stated with the
commands that produced it.

```
git cat-file -t d477dba                  -> commit
git branch -a --contains d477dba         -> main, mainz-src-port, origin/main, origin/HEAD
git archive d477dba src | tar -x -C DIR  -> 47 .py files, 328 KB
```

With `DIR` inserted at `sys.path[0]` and the working tree root never added, all
of the following import and run under `.venv/bin/python`:

| Module at `d477dba` | Third-party imports it pulls | Imports? |
|---|---|---|
| `src.rl.encoders` | `torch` | yes |
| `src.rl.actions` | via `src.envs.wrappers` → `src.envs.base_ctde_env`, which is **stdlib only** | yes |
| `src.envs.wrappers` | stdlib only | yes |
| `src.rl.models` | `torch` | yes |
| `src.sensing.local` (`_bin`, `:436-437`) | — | yes |

Agreement between the reference and `sim_contract.py`, measured directly:

| Axis | Result |
|---|---|
| `LOCAL_OBS_FIELDS`, `COOPERATION_FIELDS`, `LANE_DISTRIBUTION_LANES` | equal, in order |
| `local_obs_dim()` | 39 on both |
| `FIELD_SCALES` values, over the vendored keys | equal |
| `FIELD_SCALES` key sets | **not equal**: the reference has 54 keys, the vendored copy 32. The 22 extra reference keys (`density`, `inflow`, `queue_length`, `time`, …) are global-state fields no local-observation field reads. `is_active` is a `LOCAL_OBS_FIELD` with no scale entry and therefore takes `_field_number`'s 1.0 default, on both sides. |
| `ACTION_HEADS`, `ACTION_VALUES`, `FORCED_ACTIONS` | equal |
| `decode_headway_bin`, `decode_speed_bin` (3 bins × 3 free-flow speeds) | equal |
| `MultiCategoricalActor(39)` vs `VendoredActor(39)` state-dict key→shape map | equal, 14 entries |
| 12 edge-case encodings, `atol=1e-6`, `rtol=0` | 0 mismatches |
| `contract_fingerprint()` | `918ec57cf2f2e1db` |

**One `sys.path` constraint the implementer must respect.** Setting
`PYTHONPATH=<archive>` does **not** work when running under the Jetson test
suite. `deployment/jetson/tests/conftest.py:8` does `sys.path.insert(0, REPO_ROOT)`,
the working tree has `src/__init__.py`, and the working-tree package therefore
wins. Measured: the same pytest invocation with `PYTHONPATH` set still reports
`1 skipped`. The generator must run as its own process, inserting the archive
directory at `sys.path[0]` and never adding the repository root.

**Representation is exact.** The encoder returns `float32`. Every value across
the edge cases is finite (the `inf` branch clamps to ±5·scale before dividing).
`float(np.float32) -> json.dumps -> json.loads -> np.float32` round-trips
bit-exactly, checked over a full 39-slot vector. JSON numbers are therefore a
lossless representation and no hex or base64 encoding is needed.

## Section 3 — Decisions taken (by recommendation — not signed off by the user)

| ID | Question | Options | Taken | Why |
|---|---|---|---|---|
| D1 | How the contract is checked with the sim modules gone | (a) restore `encoders.py`, `actions.py`, `models.py` to the working tree; (b) freeze golden vectors and test against the file; (c) have the test itself `git archive` the reference on every run | **(b)**, with (c) used inside the generator only | (a) reverses `6b538f2`, whose commit message states the deletion was computed from the paper's import closure, and it reintroduces a `torch` + `highway_env` import path into a tree the ARCHITECTURE requires the device never to touch — a working-tree change made to serve a test. (c) keeps a true equality check but makes every test run depend on git history being present, on the archive extraction succeeding, and on `torch`; it also breaks on the `sys.path` collision in Section 2 unless each run spawns a subprocess. (b) is the idiom already in the repository for this exact problem: `specs/transport_golden_frames.json` freezes the wire format and the Python and Kotlin codecs test against the file rather than against each other (`deployment/jetson/tests/test_transport_golden.py:8-12`). Using (c) inside the generator recovers what (b) alone would lose — the file is derived from the original sim, not from the copy it is meant to check. |
| D2 | What the frozen file contains | see Section 4 | encoded vectors **plus their derivations**, slot names in order, scales, action constants, both bin decoders, `bin_index`, neutral fallbacks, actor layout, fingerprint | The brief's minimum, extended by two things the measurements above showed are unguarded: `bin_index`'s comparison direction and the two decoder tables, neither of which moves the fingerprint. Order is pinned by recording an ordered array of `{slot, …}` objects rather than a name→value map, so a reorder is a diff on every subsequent entry rather than no diff at all. |
| D3 | File format and location | (a) `specs/sim_contract_golden_vectors.json`; (b) a Python module of literals; (c) `deployment/jetson/policy/` | **(a)** | Matches `specs/transport_golden_frames.json` exactly: `specs/` is where frozen agreements live, JSON is inert data that cannot import anything, and a `.py` file of literals is code that a refactor tool will happily rewrite. `specs/` is tracked (`.gitignore` excludes `outputs/`, `build/` and `phone/*/build/`, not `specs/`). |
| D4 | How regeneration is prevented from blessing a broken contract | see Section 5 | triple derivation in the generator; `--force` required to overwrite; the fingerprint pinned as a literal in the test source | Explained in full in Section 5. |
| D5 | The existing `importorskip` tests | (a) delete; (b) keep, permanently skipped; (c) convert to archive-backed | **(a)** | (b) leaves a file reporting `1 skipped` forever, which reads as coverage and is the defect this task exists to close. (c) is D1(c). Deletion is safe only because the generator re-derives every one of those assertions from the same reference before writing — the coverage moves, it is not dropped. Section 6.3 maps each deleted test to what replaces it. |
| D6 | ARCHITECTURE §6's maintenance procedure | (a) delete the paragraph; (b) rewrite it | **(b)** | The paragraph is the only written answer to "the sim contract changed, now what". Deleting it leaves the question unanswered. Text in Section 8. |
| D7 | Whether the mutation proof is a one-time exercise or a standing pin | (a) run the mutations by hand once and record the result in the plan; (b) add entries to `scripts/remutate.py` | **(b)** | `scripts/remutate.py:1-11` exists because a pin lapsed silently: a test written for one route kept passing after an unrelated fix closed the route, and mutating the teardown away left 51 instrumented tests green. A mutation proves something on the day it is run and not after. |
| D8 | Where the new tests live | (a) rewrite `test_sim_contract.py`; (b) add `test_sim_contract_golden.py` beside it | **(a)** | The transport precedent has two files because both still do work. Here, removing the module-level `importorskip` is required anyway to recover the eight dormant non-sim tests (Section 1), and a second file would leave the first one either empty or still skipping. One file named for the guarantee, enforcing the guarantee. |
| D9 | How a 39-value vector is recorded | (a) 39 JSON numbers; (b) numbers plus a SHA-256 over a canonical serialization | **(b)** | The numbers are what a reviewer reads; the digest is what makes "this vector changed" one line in a diff instead of a hunt through 39. Mirrors `frame_sha256` in the transport file. Measured: exact round-trip, so the digest is stable. |
| D10 | Whether each slot records only its value | (a) value only; (b) value plus `raw`, `scale` and the branch of `_number` that produced it | **(b)** | This is what makes a diff reasoned about rather than accepted. `leader_gap: 1.3333333` says nothing; `{raw: Infinity, scale: 150.0, rule: "inf_clamp", value: 1.3333333}` says the `inf` branch produced 200.0, the ±5·scale clamp did not bind at 750.0, and 200/150 followed. It also enables the independent third account in Section 5.3. |
| D11 | How the fingerprint is pinned | (a) in the golden JSON only; (b) in the JSON **and** as a string literal in the test source | **(b)** | (a) is rewritten by any regeneration. The literal in the test source is not, and the existing file already uses this idiom with a justification: `test_additive_the_fingerprint_does_not_depend_on_it` asserts `before == "918ec57cf2f2e1db"` precisely because "two calls to the same pure function agree with each other no matter what either one returns" (`:198-206`). |
| D12 | Whether `EXPECTED_PYTHON_TESTCASES` is corrected here | (a) out of scope, raise separately; (b) correct it in this task | **(b)** | Measured: the constant is 2060 (`scripts/remutate.py:3178`) and the suite yields 2272 JUnit testcases. `run()` at `:3233-3234` returns `INCONCLUSIVE` whenever the two differ, so **every** Python mutation entry currently returns INCONCLUSIVE rather than a verdict. This task's own pass/fail gate runs through that code path. The gate cannot be satisfied while the constant is wrong, so correcting it is inside the task whether or not it looks like it. |
| D13 | Number of encoded cases frozen | (a) the 12 the current tests use; (b) 12 plus additions | **(b)**, 14 | The 12 existing cases are kept unchanged, in order, under their existing meanings. Two are added for gaps the measurements exposed: a case exercising `COOPERATION_FIELDS` with three *distinct* values, and one exercising `nearby_av_lane_distribution` with three distinct values. Both are needed by mutation M7 (Section 7). |

## Section 4 — What the golden file contains, exactly

`specs/sim_contract_golden_vectors.json`, one JSON object.

### 4.1 Top-level metadata

| Key | Value | Asserted by |
|---|---|---|
| `sim_commit` | `"d477dba"` | equals `sim_contract.SIM_COMMIT` |
| `frozen` | `true` | asserted literally |
| `generated_from` | `"git archive d477dba src"` | documentation |
| `contract_fingerprint` | `"918ec57cf2f2e1db"` | equals `sim_contract.contract_fingerprint()` **and** the literal in the test source |
| `local_obs_dim` | `39` | equals `sim_contract.local_obs_dim()` |
| `note` | the freeze rule, in prose: what a change to a recorded value means, and that adding a case is not such a change | asserted to contain the sentence the test names |

### 4.2 `slot_names`

An **ordered array** of 39 strings: `sim_contract.encoded_slot_names()`. Recorded
as an array, never as an object, so position is part of the data. Asserted
element-by-element against `encoded_slot_names()`, and its first 33 entries
against `LOCAL_OBS_FIELDS` in order.

### 4.3 `field_scales`

An ordered array of `{"field": str, "scale": float}` in `LOCAL_OBS_FIELDS` order,
followed by any scale key not in that list. The test asserts set equality in
**both** directions against `sim_contract.FIELD_SCALES` (see O3) and per-field
value equality.

### 4.4 `cases` — 14 encoded observations

Each entry:

```
{
  "name": "full_obs",
  "why":  "the ordinary case: every field present and numeric",
  "obs":  { ... the input observation, with inf written as the string
             "Infinity" and NaN as "NaN", decoded by the test ... },
  "vector_sha256": "f237016d0f48caad",
  "slots": [
    {"slot": "is_active",  "raw": true,       "scale": 1.0,   "rule": "bool",       "value": 1.0},
    {"slot": "ego_speed",  "raw": 23.4,       "scale": 40.0,  "rule": "plain",      "value": 0.585},
    {"slot": "leader_gap", "raw": "Infinity", "scale": 150.0, "rule": "inf_clamp",  "value": 1.3333333},
    ...39 entries, in encoder order...
  ]
}
```

`rule` is one of exactly five values, one per branch of `sim_contract._number`
(`:150-168`): `bool`, `none`, `parse_fail`, `inf_clamp`, `plain`.

JSON has no literal for infinity or NaN, and `json.dumps` would emit the
non-standard tokens `Infinity` and `NaN`. The file therefore writes them as the
strings `"Infinity"`, `"-Infinity"`, `"NaN"`, and a test asserts the file's raw
text contains neither a bare `Infinity` nor a bare `NaN` token — the same rule
`test_no_message_case_contains_a_nan_token` enforces on the transport file.

The 14 cases: the 12 from `test_sim_contract.py:64-77`, unchanged and in the same
order, named `empty`, `all_none`, `full_obs`, `inf_leader_gap_and_headway`,
`neg_inf_relative_speed`, `nan_ego_speed`, `bools_flipped`, `numeric_string`,
`junk_string`, `cooperation_not_a_mapping`, `lane_distribution_not_a_mapping`,
`inf_time_since_lane_change`; plus `cooperation_distinct_values` and
`lane_distribution_distinct_values` (D13).

### 4.5 `actions`

`heads` as an ordered array; `values` as an ordered array of
`{"head": str, "values": [str, ...]}` preserving within-head order;
`forced` as an ordered array of `{"head": str, "value": str}`; `profiles` as an
ordered array of `{"profile": str, "heads": [str, ...]}`; and `default_indices`
as an ordered array of `{"head": str, "index": int}`.

Order within a head matters and is not covered by the fingerprint: the runtime
converts an argmax index to a string through `ACTION_VALUES[head][idx]`
(`actor_runtime.py:155` calling `sim_contract.indices_to_action`, `sim_contract.py:210-216`).
Swapping `"slow"` and `"fast"` changes what the driver is told while changing
nothing the fingerprint hashes.

### 4.6 `decoders`

- `headway_bin_s`: ordered array of `{"bin": str, "seconds": float}` for all 3 bins.
- `speed_bin_mps`: ordered array of `{"bin": str, "free_flow_mps": float, "min_contextual_mps": float, "value": float}` over the 3 bins × free-flow speeds `(13.0, 20.0, 22.0, 30.0, 40.0)`. The set covers all three regimes of the `max()` in `decode_speed_bin` (`:224-229`), measured: at 13.0 the `min_contextual_speed_mps` floor binds for two bins (`slow`, `nominal`); at 20.0 it binds for one (`slow`, at `10.0 < 12.0`); at 22.0 it binds for one **at exact equality** (`22.0 - 10.0 = 12.0`), which is where a `>` written as `>=` would show; at 30.0 and 40.0 it binds for none.
- `bin_index`: ordered array of `{"value": float, "edges": [float, ...], "index": int}` over edges `(0.5, 1.0, 2.0)` and values `(0.0, 0.4999, 0.5, 0.9999, 1.0, 1.5, 2.0, 2.5)`. The values on the edges are the point of the case: `_bin` counts `value >= edge` (`src/sensing/local.py:436-437` at `d477dba`), so 0.5 gives 1 and not 0.

### 4.7 `neutral_cooperation`

Ordered array of `{"free_flow_mps": float, "values": [{"field": str, "value": float}, ...]}`
for free-flow speeds `(13.0, 30.0)`, with `values` in `COOPERATION_FIELDS` order.
Cross-checked in the test against `specs/observation_schema.md:94-96`, which is
the source `sim_contract.py:239-246` cites and which is still in the tree.

### 4.8 `actor_state_dict_layout`

Ordered array of `{"key": str, "shape": [int, ...]}`, 14 entries, from
`src.rl.models.MultiCategoricalActor(39)` at `d477dba`. Measured today:
`backbone.{0,2,4}.{weight,bias}` with shapes `[128,39]`, `[128]`, `[128,128]`,
`[128]`, `[128,128]`, `[128]`, and four heads each `[3,128]` / `[3]`.

## Section 5 — The generator, and the rule that stops a regeneration

`scripts/generate_sim_contract_golden_vectors.py`.

A golden file regenerated whenever the test fails is a self-portrait. Three
mechanisms stand between this file and that, and they fail in different ways on
purpose.

### 5.1 The generator derives every value twice and refuses to write on disagreement

The generator extracts `d477dba` with `git archive` into a temporary directory,
inserts it at `sys.path[0]`, and computes every recorded quantity **from the
reference**. It separately computes the same quantity from
`deployment/jetson/policy/sim_contract.py`. If any pair disagrees it prints the
disagreement and exits non-zero, having written nothing.

The consequence is the one that matters: a person who edits `sim_contract.py`
incorrectly and then runs the generator to make the test pass does not get a
file. They get a non-zero exit naming the field that moved. The only way to
change a recorded value is to change `SIM_COMMIT` to a commit whose reference
actually produces the new value — which is a contract change, is visible in the
diff, and is out of this task's scope.

Default mode is `--check`: derive everything three ways (Section 5.3), compare
against the recorded file, write nothing, exit non-zero on any difference.
`--write` is required to write, and refuses an existing file without `--force`,
matching `generate_transport_golden_frames.py:415-419`.

The temporary directory is removed in a `finally`. Nothing is added to the
working tree.

### 5.2 The file will not be silently overwritten

`--force` is required, and the file declares `"frozen": true`. A test asserts
that flag and asserts the `note` text so a reader of a diff knows which kind of
change they are looking at. Following the transport precedent
(`test_transport_golden.py:255-278`), a test regenerates into a temporary
directory and asserts that every **pre-existing** case is byte-identical, so
adding a case cannot quietly move an existing one.

### 5.3 Three independent accounts of every number

- **Account A**: the simulation at `d477dba`, imported from the archive.
- **Account B**: `sim_contract.py`'s numpy encoder, as it is in the tree.
- **Account C**: arithmetic restated in the test file, from `raw`, `scale` and
  `rule`, without calling either encoder:

      bool       -> float(raw)
      none       -> 0.0
      parse_fail -> 0.0
      inf_clamp  -> sign * min(200.0, 5.0 * max(scale, 1e-9)) / max(scale, 1e-9),
                    where sign = +1.0 if raw > 0 else -1.0
      plain      -> float(raw) / max(scale, 1e-9)

  and, separately, a check that `rule` is the branch the `raw` value actually
  selects — a `raw` of `2.0` recorded as `inf_clamp` is a failure even when the
  arithmetic happens to agree.

  **`sign` must be written as the comparison `raw > 0`, not as `math.copysign`.**
  `_number` (`sim_contract.py:161-162`) takes the negative branch for anything
  that is not strictly positive, and `NaN > 0` is False, so NaN encodes negative.
  Measured: `ego_speed = NaN` encodes to `-5.0`, while
  `math.copysign(5.0, float("nan"))` returns `+5.0`. An Account C written with
  `copysign` disagrees with both other accounts on the `nan_ego_speed` case, and
  a reviewer would read that as a contract defect rather than a restatement
  error.

A and B are compared inside the generator. B and C are compared inside the test,
which runs everywhere and imports no simulation module. A single wrong value must
survive all three to reach the device.

Account C is why D10 records the derivation rather than the number alone. Without
it, the test could only compare the file against `sim_contract.py` — which is the
circularity the golden file exists to break.

### 5.4 What the fingerprint is for, and what the vectors are for

They must not be allowed to cover for each other. The split is explicit and is
enforced by the gate:

- `contract_fingerprint()` covers field order and scales, and is the only one of
  the two that the **device** consults at bundle load (`actor_runtime.py:95-101`).
- The golden vectors cover everything else, and the gate requires mutations M3
  through M7 (Section 7) to be caught by a test that does **not** read the
  fingerprint. If a fingerprint test is the one that catches them, the vectors
  have added nothing and the gate fails.

## Section 6 — The tests

`deployment/jetson/tests/test_sim_contract.py`, rewritten. The module-level
`importorskip` at `:18` is removed; the file imports `json`, `math`, `pathlib`,
`numpy` and `policy.sim_contract`, and nothing from `src`.

### 6.1 Test groups

| Group | What it asserts | Needs torch? |
|---|---|---|
| the file itself | `frozen` is true; `sim_commit` equals `sim_contract.SIM_COMMIT`; `local_obs_dim` equals 39; case names unique; the `note` carries the freeze sentence; the raw text carries no bare `NaN` or `Infinity` token | no |
| slot order | `slot_names` equals `encoded_slot_names()` element-wise; first 33 equal `LOCAL_OBS_FIELDS` in order; the two nested blocks are dotted and in the recorded order | no |
| scales | both-direction key-set equality and per-field value equality against `FIELD_SCALES` | no |
| encoding (14 parametrized) | `encode_local_observation(obs)` equals the recorded `value` list, `atol=1e-6`, `rtol=0`; shape is `(39,)`; the SHA-256 of the canonical serialization equals `vector_sha256` | no |
| derivation (14 parametrized) | Account C: every slot's `value` recomputed from `raw`/`scale`/`rule`, and `rule` is the branch `raw` actually selects | no |
| actions | heads, per-head value order, forced defaults, profiles and `default_indices()` against the recorded arrays | no |
| decoders | `decode_headway_bin`, `decode_speed_bin` over the recorded grid, `bin_index` over the recorded grid | no |
| neutral fallbacks | `neutral_cooperation` against the recorded values, and against `specs/observation_schema.md:94-96` | no |
| fingerprint | `contract_fingerprint()` equals the file's value **and** equals the string literal `"918ec57cf2f2e1db"` typed in the test source | no |
| regeneration | regenerate into a temp dir; every pre-existing case byte-identical | no (spawns the generator, which needs torch and git) |
| actor layout | `VendoredActor(39).state_dict()` key→shape map equals the recorded array | **yes** |
| retained | the eight existing non-sim tests (`test_inf_encoding_known_values`, `TestTheBundleGuardSeesMoreThanTheDimension`, `TestEncodedSlotNames`), unchanged | no |

### 6.2 Torch, and where the check can run

Only two items need torch: the actor-layout test, and the regeneration test via
the generator subprocess. Everything else runs on a bare interpreter with numpy.

This is not a new constraint on the device: `policy/actor_runtime.py:26` imports
torch unconditionally, so any machine that can run the runtime can run these two.
It is recorded because it bounds where the check is complete, and the ARCHITECTURE
text in Section 8 states it. The regeneration test additionally needs `git` and
the `d477dba` object; both tests skip with a stated reason if their dependency is
missing, and the plan's gate requires them to run and pass on the development
machine before sign-off. **These are the only two skips the new file may
introduce, and neither may be skipping on the machine where the gate is run.**

### 6.3 What replaces each deleted test

| Deleted (`test_sim_contract.py` at `975b7a2`) | Replaced by |
|---|---|
| `test_field_lists_match_sim` `:79` | generator Account A comparison + `slot_names`/`field_scales` tests |
| `test_field_scales_match_sim` `:86` | generator Account A comparison + the scales test, now both-direction (O3) |
| `test_encoding_matches_sim[0..11]` `:92` | generator Account A comparison + 14 encoding tests + 14 derivation tests |
| `test_action_constants_match_sim` `:100` | generator Account A comparison + the actions tests, now covering within-head order |
| `test_decoders_match_sim_wrappers` `:110` | generator Account A comparison + the decoders tests, now covering `bin_index` and the `min_contextual` boundary |
| `test_actor_state_dict_layout_matches_sim` `:123` | generator Account A comparison + the actor-layout test |

Every row's left-hand side ran only when the sim imported, which is never. Every
row's right-hand side runs on every invocation of the suite, except the two noted
in 6.2.

## Section 7 — The mutation gate

**This is a pass/fail gate. The new check is not trusted, and this task is not
finished, until every row below has been observed to behave as the table says.**

Each mutation is applied to the stated file at the stated anchor, the Jetson
suite is run, and the verdict is recorded. The harness is `scripts/remutate.py`,
which already reports which test caught a mutation rather than only that
something failed (`:3194-3212`), and which distinguishes SURVIVED from
"did not build/import" and from INCONCLUSIVE.

### 7.1 The mutations

| # | Name | File | Anchor → replacement | Must be caught by |
|---|---|---|---|---|
| M1 | **field order** — two fields swapped, dimension unchanged | `deployment/jetson/policy/sim_contract.py` | in `LOCAL_OBS_FIELDS`: `    "leader_gap",\n    "leader_relative_speed",` → `    "leader_relative_speed",\n    "leader_gap",` | a golden test (slot order and/or encoding). The fingerprint test may also fire; it may not be the only one. |
| M2 | **scale changed** | same | `    "leader_gap": 150.0,` → `    "leader_gap": 120.0,` | a golden test (scales and/or encoding). Same rule as M1. |
| M3 | **bin edge changed** | same | `HEADWAY_BIN_S: dict[str, float] = {"normal": 1.6, "larger": 2.2, "largest": 3.0}` → `... "larger": 2.3 ...` | a golden **decoders** test. **A fingerprint test catching this is a gate failure**, since the fingerprint does not hash `HEADWAY_BIN_S`. |
| M4 | **bin boundary direction** | same | in `bin_index`: `sum(float(value) >= edge for edge in edges)` → `sum(float(value) > edge for edge in edges)` | a golden `bin_index` test. Not the fingerprint. |
| M5 | **inf clamp constant** | same | in `_number`: `result = 200.0 if result > 0 else -200.0` → `result = 100.0 if result > 0 else -100.0` | a golden encoding test **and** a derivation test. Not the fingerprint. |
| M6 | **action value order within a head** | same | `"desired_speed_bin": ("slow", "nominal", "fast"),` → `"desired_speed_bin": ("fast", "nominal", "slow"),` | a golden actions test. Not the fingerprint. |
| M7 | **cooperation block order** | same | in `COOPERATION_FIELDS`: `    "segment_target_speed",\n    "merge_pressure",` → `    "merge_pressure",\n    "segment_target_speed",` | a golden slot-order test **and** the `cooperation_distinct_values` encoding case (D13). Not the fingerprint. |

M1, M2 and M3 are the three the task brief requires. M4 through M7 are added
because they are the changes the measurements in Section 1 showed nothing
currently sees, and because M3 through M7 are the evidence that the golden
vectors cover ground the fingerprint does not.

### 7.2 The negative controls

Three, because a guard that has only been seen to pass has not been seen.

- **C1 — clean run.** With no mutation applied, the full Jetson suite passes and
  the new file contributes no skips on the development machine (Section 6.2).
- **C2 — the catching test is named and is the right one.** `remutate.py` prints
  `by {failing_test}`. For each of M1–M7 that name is recorded in the sign-off
  record. A mutation "caught" by an unrelated failure is a gate failure, and for
  M3–M7 a mutation caught only by a fingerprint test is a gate failure.
- **C3 — the generator refuses.** With M2 applied, run the generator in
  `--check` mode. It must exit non-zero and name `leader_gap`. With M2 applied,
  run it with `--write --force`. It must **still** exit non-zero and write
  nothing, because the reference disagrees. This is the direct test of Section
  5.1, and it is the one that proves the file cannot be regenerated into
  agreement with a broken contract.

### 7.3 Preconditions

`remutate.py` `run("python")` returns `INCONCLUSIVE` when the JUnit testcase
count differs from `EXPECTED_PYTHON_TESTCASES` (`:3233-3234`). Measured: the
constant is 2060 and the suite produces 2272. **Every Python mutation therefore
returns INCONCLUSIVE today, including these seven.** The constant must be
re-measured with `pytest --junit-xml` after the new tests land and set to the
observed count, before the gate is run. If the gate is run against a stale
constant it reports nothing and proves nothing.

`remutate.py:16-19` also warns that it edits files in place: the tree must be
committed before it runs, and it must not be pointed at a tree a validator is
reading.

## Section 8 — The correction to `ARCHITECTURE.md` section 6

The paragraph at `deployment/jetson/ARCHITECTURE.md:149-153` instructs running
`test_sim_contract.py` "on a machine where the sim imports (encoder tests run
everywhere torch exists; action/wrapper tests need highway_env)". No such machine
exists: the modules were deleted. The parenthesis is also wrong on its own terms —
at `d477dba`, `src.envs.wrappers` imports only `src.envs.base_ctde_env`, which is
stdlib-only, so the action and wrapper comparisons never needed `highway_env`.

Replacement text for `:149-157`:

> **When the sim contract changes:** update `sim_contract.py` (and `SIM_COMMIT`),
> then regenerate the frozen vectors with
> `python3 scripts/generate_sim_contract_golden_vectors.py --write --force`. The
> generator extracts the sim at `SIM_COMMIT` with `git archive` into a temporary
> directory and derives every recorded value twice, once from that reference and
> once from `sim_contract.py`; it writes nothing if the two disagree, so it
> cannot be used to make a failing test pass. Then run
> `python3 -m pytest deployment/jetson/tests/test_sim_contract.py`, which reads
> `specs/sim_contract_golden_vectors.json` and imports no simulation module, so
> it runs on any machine — including the Jetson. Then re-export the policy bundle
> (`export_policy.py` refuses dim mismatches, and stamps the manifest with
> `contract_fingerprint`, which `actor_runtime` refuses to load against a
> different contract).
>
> Two of those tests need more than numpy: the actor state-dict layout check
> needs `torch` (which `actor_runtime.py` imports anyway), and the regeneration
> check spawns the generator, which needs `torch` and a `git` repository holding
> the `SIM_COMMIT` object. Both skip with a stated reason where their dependency
> is absent; run them on the development machine.
>
> The frozen file records the observation vectors slot by slot with the
> derivation of each number, the slot order, the scales, the action heads and
> their per-head value order, both bin decoders, `bin_index`, the neutral
> fallbacks and the contract fingerprint. Reordering two fields leaves the
> dimension at 39 and puts every value in the wrong slot, so order is recorded as
> ordered arrays rather than as name→value maps. The mutations that prove the
> check fails when the contract moves are registered in `scripts/remutate.py`
> under the names beginning `sim contract:`.
>
> The actor architecture (`backbone.{0,2,4}` + `heads.<name>` state-dict layout)
> is mirrored in `export_policy.VendoredActor` and frozen in the same file.

## Section 9 — The work

| # | Step | Produces | Done when |
|---|---|---|---|
| 1 | Branch and worktree for the task, off `main` | — | branch exists, tree clean |
| 2 | Re-run the three baseline measurements: the one-skip result, the 25-pass result against the archive reference, and the full-suite JUnit count | numbers in the sign-off record | all three reproduce Section 1 |
| 3 | Write `scripts/generate_sim_contract_golden_vectors.py`: `git archive` extraction with `finally` cleanup, `sys.path[0]` insertion, Account A and Account B derivation of every quantity in Section 4, disagreement report, `--check` default, `--write` / `--force` | the generator | `--check` runs against no file and reports the file is absent, exit non-zero |
| 4 | Run `--write` to produce `specs/sim_contract_golden_vectors.json` | the frozen file | file exists; `--check` passes; a second `--write` without `--force` refuses |
| 5 | Read the generated file end to end and confirm each of the 14 cases' 39 slots against Section 4.4's five rules by hand-checking at least the `inf_clamp`, `bool`, `parse_fail` and `none` slots | a reviewed file | reviewer can state, for each rule, one slot that exercises it |
| 6 | Rewrite `deployment/jetson/tests/test_sim_contract.py` per Section 6: remove the module `importorskip`, delete the six sim-import tests, keep the eight existing non-sim tests unchanged, add the groups in 6.1 | the test file | file collects with no skips on the development machine |
| 7 | Run `pytest deployment/jetson/tests/test_sim_contract.py -q -p no:cacheprovider` | a count | passes; 0 skipped |
| 8 | Run the full Jetson suite with `--junit-xml`; count JUnit testcases | a number | passes; skip count is 24 with no handset attached |
| 9 | Set `EXPECTED_PYTHON_TESTCASES` in `scripts/remutate.py` to the count from step 8 | corrected constant | constant equals the measured count |
| 10 | Add M1–M7 to `MUTATIONS` in `scripts/remutate.py`, names prefixed `sim contract:`, kind `python` | seven entries | each anchor appears exactly once in the file (`remutate.py` reports ANCHOR AMBIGUOUS otherwise) |
| 11 | Commit everything, because `remutate.py` edits files in place and restores from its own copy | a commit | `git status` clean |
| 12 | Run `python3 scripts/remutate.py python --name="sim contract:"` | seven verdicts | all seven CAUGHT, none SURVIVED, none INCONCLUSIVE, none "did not build/import" |
| 13 | Record the catching test name for each of M1–M7 and check them against Section 7.1's "must be caught by" column | the C2 record | no M3–M7 row is caught only by a fingerprint test |
| 14 | Run control C3: apply M2 by hand, run the generator in `--check` and in `--write --force`, confirm both exit non-zero and write nothing, restore M2 | the C3 record | file unchanged on disk after both runs (compare SHA-256 before and after) |
| 15 | Rewrite `ARCHITECTURE.md:149-157` with Section 8's text | corrected doc | no sentence in §6 describes a machine or a dependency that does not exist |
| 16 | Update `plans/implementation_records.md` entry 143 from "Open" to a record of what was built and what the gate measured | the record | entry states the seven mutation verdicts |
| 17 | Full Jetson suite once more, after every edit | a final count | passes; matches the constant set in step 9 |

## Section 10 — Risks

**R1 — a golden file that is regenerated rather than reasoned about.** The named
failure mode for this whole approach: a test fails, someone runs the generator,
the test passes, and nothing was checked. Countered three ways in Section 5, the
strongest of which is that the generator refuses to write when `sim_contract.py`
disagrees with `d477dba` — so regeneration cannot produce agreement with a broken
contract, only with a correct one. Control C3 is the direct test of that claim.
**Residual risk**: someone changes `SIM_COMMIT` to a commit whose reference
matches their edit. That is a contract change, it is visible in the diff as a
changed commit id, and it is out of this task's scope — but nothing in this task
mechanically prevents it.

**R2 — the fingerprint and the golden vectors masking each other.** Both move on
a reordered field and a changed scale. If a fingerprint test is what catches M3
through M7, the golden vectors have added nothing and the reader of a green suite
would conclude otherwise. Countered by making that a stated gate failure
(Section 7.1, control C2), which requires reading the catching test's name rather
than only the verdict. `remutate.py:3234-3237` already prints it.

**R3 — torch constrains where the check runs.** Two of the new tests need torch,
and one of those also needs git and the `d477dba` object. They skip with a stated
reason elsewhere, which means a machine without torch runs an incomplete check
and still reports green. Countered by bounding it: torch is already an
unconditional import in `policy/actor_runtime.py:26`, so no machine that runs the
device runtime is affected; by requiring both tests to run and pass on the
development machine before sign-off; and by stating the limit in ARCHITECTURE §6.
**Residual risk**: a CI configuration without torch would report a pass on a
check that did not run. Nothing in this task adds such a configuration.

**R4 — the reference is a git object, not a file.** The generator depends on
`d477dba` remaining reachable. Measured: it is reachable from `main`,
`origin/main` and `mainz-src-port`, so it is not a candidate for garbage
collection and it exists on the remote. **Residual risk**: a shallow clone, or a
history rewrite, removes it. The failure is loud — `git archive` exits non-zero
and the generator cannot run — but the *test suite* would still pass, because the
tests read the frozen file and not the reference. That is the intended split, and
it is also the reason the frozen file, not the generator, is the artifact of
record.

**R5 — the file freezes agreement as of `d477dba`, not agreement in general.**
After this task, the check enforces that `sim_contract.py` still produces the
values the sim produced at `d477dba`. It does not and cannot enforce agreement
with a *future* simulation, because the device deliberately does not import one.
Stated flatly because the paper's claim is about what the device runs versus what
was trained, and what was trained is a fixed past artifact — which is the case
this construction covers.

**R6 — the mutation harness cannot currently return a verdict.** Measured:
`EXPECTED_PYTHON_TESTCASES` is 2060 against a suite producing 2272 JUnit
testcases, so `run()` returns `INCONCLUSIVE` for every Python entry
(`remutate.py:3233-3234`). The gate in Section 7 runs through that code path.
Addressed as step 9, and it must be done before step 12 or the gate measures
nothing. This is a pre-existing condition, not one this task introduces, and it
means the repository's Python mutation pins have been returning no verdict since
the suite last grew past 2060.

**R7 — the archive's `src` losing to the working tree's `src`.** Measured:
`PYTHONPATH=<archive>` has no effect under the Jetson suite, because
`tests/conftest.py:8` inserts the repository root at position 0 and the working
tree has `src/__init__.py`. A generator that gets this wrong imports the
*current* `src` — which holds only `src_q.py` — and fails with
`ModuleNotFoundError`, or worse, on a future tree, imports a different module
under the same name. Countered by running the generator as its own process that
inserts the archive at `sys.path[0]` and never adds the repository root, and by
asserting inside the generator that `src.rl.encoders.__file__` is under the
temporary directory before any value is read from it.

**R8 — the frozen cases stop covering the contract as it grows.** Adding a field
to `LOCAL_OBS_FIELDS` changes the dimension, which the fingerprint and
`local_obs_dim` both catch. Adding a `COOPERATION_FIELDS` entry does not move the
fingerprint, and would be caught only by the slot-order test. Recorded rather than
countered: the slot-order test covers it, and D13 adds the two cases that give the
nested blocks distinct per-slot values so a reorder inside them changes numbers
and not only names.

## Section 11 — Open items flagged for the user

Each of these is a point where `plan_dsrc` would have stopped and asked.
`plan_dsrc_rec` took the recommended option and carried on; the disclosure is
here.

**O1 — deleting the 25 sim-import tests is a coverage claim.** They pass today
against the reference (measured). Deleting them is defensible only because the
generator re-derives every one of their assertions from the same reference before
it writes, so the assertion moves from test time to generation time. That is a
real weakening: the sim comparison then runs when someone regenerates, not on
every suite run. The alternative, D1(c), keeps the comparison on every run at the
cost of making the test suite depend on git history, a subprocess and torch.
Taken: delete. Section 6.3 maps every deleted test to its replacement.

**O2 — the actor state-dict layout test needs torch, and is the one golden test
that is not pure data.** It could be dropped, leaving the layout unchecked, or
kept as the only torch-requiring test in the file. Taken: keep. The layout is what
makes `VendoredActor` load a bundle trained by `MultiCategoricalActor` at all, and
torch is already an unconditional import in the runtime it guards.

**O3 — `FIELD_SCALES` is checked in one direction only, today.**
`test_field_scales_match_sim` (`:86-89`) iterates over the *vendored* dict, so a
scale the sim has and the vendored copy lacks is invisible. Measured: the sim has
54 keys, the vendored copy 32, and the 22 extras are global-state fields no local
observation reads — so the gap is currently benign. It would stop being benign the
moment a field moved from the global encoder to the local one. Taken: the new test
asserts key-set equality in both directions against the recorded array, and the
generator asserts that the vendored key set equals the reference's key set
restricted to the fields the local encoder reads. **This is a behaviour change to
an existing assertion and is the one place this task tightens something the brief
did not ask about.**

**O4 — `EXPECTED_PYTHON_TESTCASES` is stale and that is not nominally this task's
business.** Measured: 2060 against 2272. Every Python mutation entry in
`remutate.py` returns INCONCLUSIVE. Taken: correct it here, because this task's
own gate cannot produce a verdict otherwise. The side effect is that the
repository's other Python mutation pins start returning verdicts again, and some
of them may report SURVIVED. **That is out of this task's scope to fix, and it may
surface findings** — a lapsed pin elsewhere is exactly what `remutate.py` exists
to detect. It should be reported, not repaired inside task 143.

**O5 — two documents state a dependency that does not exist.** ARCHITECTURE
`:151-152` and `test_sim_contract.py:3-5` both say the action and wrapper
comparisons need `highway_env`. At `d477dba` they do not:
`src.envs.base_ctde_env` imports only `abc` and `typing`. (`highway_env` 1.12.1 is
in fact installed in `.venv`, so the claim was never tested.) Taken: correct the
ARCHITECTURE text; the test docstring is rewritten wholesale by step 6.

**O6 — `SIM_COMMIT` in the golden file creates a second, weaker link.**
`contract_fingerprint` deliberately excludes `SIM_COMMIT`, with a stated reason
(`sim_contract.py:263-266`): the sim moves for reasons that do not touch the
contract, and a guard firing on every unrelated commit gets turned off. The golden
file records `sim_commit` and a test asserts it equals `sim_contract.SIM_COMMIT`.
That assertion fires when someone bumps `SIM_COMMIT` without regenerating — which
is the correct behaviour — but it also means a `SIM_COMMIT` bump now requires a
regeneration even when no contract value moved. Taken: record and assert it, do
not hash it. The regeneration in that case is cheap and its diff is one line.

**O7 — nothing in this task checks any exported bundle.** `models/*.json`
manifests carry a `contract_fingerprint` written by `export_policy.py:133`, and
`actor_runtime` accepts a manifest with no fingerprint at all
(`actor_runtime.py:95`, deliberately). Whether any bundle currently on the device
or in the tree carries a fingerprint matching `918ec57cf2f2e1db` is not asked
here. It is task 145's ground, and it is named in the scope boundary.

## Section 12 — Sign-off

The task is finished when every line below can be answered with a measured
number or a named test, not with a description.

| # | Sign-off item | Evidence required |
|---|---|---|
| S1 | `test_sim_contract.py` reports 0 skipped | the pytest line |
| S2 | The full Jetson suite passes, with 24 skips and no non-hardware skip | the pytest line and the JUnit skip breakdown |
| S3 | `EXPECTED_PYTHON_TESTCASES` equals the measured JUnit count | both numbers |
| S4 | M1 CAUGHT | the catching test's name |
| S5 | M2 CAUGHT | the catching test's name |
| S6 | M3 CAUGHT, and not by a fingerprint test | the catching test's name |
| S7 | M4 CAUGHT, and not by a fingerprint test | the catching test's name |
| S8 | M5 CAUGHT, and not by a fingerprint test | the catching test's name |
| S9 | M6 CAUGHT, and not by a fingerprint test | the catching test's name |
| S10 | M7 CAUGHT, and not by a fingerprint test | the catching test's name |
| S11 | Control C1: clean run green | the pytest line |
| S12 | Control C3: the generator refuses to write under M2, in both `--check` and `--write --force` | both exit codes, and the file's SHA-256 before and after |
| S13 | The golden file's 14 cases each exercise a named rule, and each of the five `_number` rules is exercised by at least one recorded slot | the slot names, one per rule |
| S14 | ARCHITECTURE §6 names no machine or dependency that does not exist | the rewritten text |
| S15 | `implementation_records.md` entry 143 records the seven verdicts | the entry |
| S16 | No file outside the list in Section 9 was modified | `git diff --stat` |
