# Reviewer and blend verification — 2026-10-25

Scope: restore experiment review under autopilot and fix the controller's automatic blend. Preparation, research, API configuration, fallback policy and submission resource defaults are unchanged.

## Changes

- Every experiment that passes smoke now calls the existing semantic reviewer, including controller baselines and AutoML references under autopilot. `revise` enters the existing bounded implementer repair loop; modified code must pass smoke and review again before full execution. The review cache still requires identical source bytes and contract.
- The misleading `--no-autopilot` CLI help was corrected.
- Both validation selection and final blending pass the preparation's ID columns and reference rows. IDs are copied, never averaged/voted. Mismatched columns, indexes, row counts, IDs, missing/nonfinite values and invalid weights reject that blend.
- Numeric predictions/probabilities are averaged; class labels use weighted per-row voting, including floating, boolean and string class labels. Ties favor the first member. Voting complexity is O(rows × members²), with no matrix sized by the total number of unique labels.
- Output prediction semantics are read from extraction, then known task/metric semantics; integer probability endpoints 0/1 are not mistaken for labels. Unsupported/ambiguous semantics or batch predictions skip automatic blending and retain the best individually validated model.
- Prediction frames may omit IDs: their positional contract is enforced by the existing worker and equal indexes/lengths. With no IDs, arbitrary silently reordered values cannot be proven aligned by combine alone; existing frozen-predictor row-order smoke probes remain in place.

## Offline validation

- Full local suite: 216 test executions passed in 190.397 seconds. This includes inherited/repeated test cases already present in discovery. Interpreter shutdown emitted ignored multiprocessing ResourceTracker cleanup errors after the successful test result; test process exit was 0.
- Added integration coverage subsequently passed with the focused suite: 10 blend tests, including validation selection and final writing with fit-only and refitted member predictions.
- The same 10 blend tests passed on Linux against the flat archive modules.
- All 15 Python files in the archive imported independently in fresh Linux processes.
- Twelve initial focused tests passed, covering autopilot review, review-triggered repairs after smoke, and blend edge cases.

## Real reviewer repair probe

A separate diagnostic node intentionally received `.median()` where the immutable plan explicitly required `.mean()`. Only its initial code was seeded; all reviewer decisions and the implementer's repair came from the real official API.

- 3 actual API requests; 35.37 seconds; exit 0.
- First reviewer: revise.
- Implementer changed median to mean.
- Second reviewer: pass.
- Repeated smoke and full execution passed; node completed.

## Previous six-node predictions

On the same 18,308 validation rows where the old combine hit its 300-second timeout:

- Corrected greedy blend selection: 2.257 seconds.
- Final combination/writing: 1.525 seconds.
- Selected four members; validation RMSE 43.1372 versus best single 45.0432. These are local validation scores, not platform scores.
- Output: 22,449 rows, correct ID order, no nulls.

## Archive

Candidate SHA256: `b03898d0b0c9bf546af3fa21c110a396bb96ed3b1908eb0664f259fab990c400`.
81,511 bytes; 15 root Python modules and the unchanged official requirements.txt.
Fallback remains disabled. Submission defaults remain 160 minutes total, six executors, one library thread, eight concurrent LLM calls, reasoning low and two researchers.

## Ten-node live run

Diagnostic overrides: max_nodes=10, total agent budget=1770 seconds, outer wall limit=1800 seconds, experiment timeout=300 seconds, API timeout=120 seconds, maximum 90 actual network requests. These are test-only limits; they do not change the ZIP defaults.


Completed result:

- Exit 0, ready/complete, stopped at the 10-node limit; elapsed 913.89 seconds (15m14s).
- Ten model nodes completed; zero failed. Twelve researchers completed; zero failed/cancelled.
- 67 real API requests in the main run, plus 3 in the isolated repair probe. Model reviewer produced 15 valid verdicts and retried one invalid quote response. All 10 final review hashes match the code actually executed.
- Four nodes were repaired after reviewer objections (005, 006, 007, 008). Runtime/smoke repair paths also executed successfully. The final successful node 006 had previously hit a training timeout, then a LightGBM argument error; both were recovered within the existing node request budget.
- Best single: node_006, validation RMSE 42.871348. Selected blend: node_006 × 2, node_005 × 1, node_003 × 1; validation RMSE 41.311067.
- All three selected members refitted on the full training table. Final output passed validated_blend/final_submission_validated: 22,449 rows, columns id/site_eui, exact test ID order, no nulls.
- finalization.json is absent for the successful blend path: completion evidence is report.checks plus blend.json sources=refit_all_train for every member and the final CSV checks.
- Network/LLM failures in this run were recoverable schema-validation errors; this is not a reproduction of every possible platform cancellation/proxy failure. Fault-injection lifecycle tests also passed in the full local suite.

## Deliverables and reproducibility

- ZIP: `submission_review_blend_20261025.zip` (copied byte-for-byte from the server-tested candidate, not rebuilt).
- Changed archive files versus submission_research_20261025.zip: alfa_runner.py, alfa_worker.py, alfa_cli.py only. All other entries and the official requirements.txt are byte-identical.
- All archive entries were compared byte-for-byte with a fresh export of the final source tree; exact match.
- Evidence: `artifacts/reviewer-blend-20261025/`, including main run report, reviewer responses, generated sources, repair trace, smoke/review/finish hashes, Linux test logs and benchmark outputs. No original CSV rows, saved model binaries or API keys are included.
- Server run: `/mnt/20tb/pam-edelkin/experiments/codex_review_blend_20261025`; all computation was launched through tmux. No platform submission was made.
