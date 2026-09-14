# Checkpoints used in the thesis

This folder holds exactly the checkpoints and logs the thesis's results chapter reports
numbers from. Unlike `runs/` (gitignored, and full of intermediate/abandoned attempts),
everything here is tracked so the reported results stay reproducible from the repo alone.

- `vae-phase1/` -- `context_mode="vae"` run, Phase 1 stopped at iteration 5,500 of a planned
  20,000. Referenced in the thesis as the checkpoint behind Section "A First Completed Run"
  and the "Empirical Results" section.
- `vae-phase2/` -- the detector trained on top of that checkpoint, stopped at iteration 121
  of a planned 300 (only the final checkpoint was kept; there is no earlier one).
- `bruno-phase1/` -- `context_mode="bruno"` (the adopted posterior) run, Phase 1 checkpoint
  at iteration 19,500 of its planned 20,000 (the checkpoint the Phase-2 run below was
  actually trained against; a later 19,999 checkpoint also exists in the original run but
  isn't what the reported detector was trained on). Referenced as "A Second Completed Run"
  and "A Second, Fully-Trained Run".
- `bruno-phase2/` -- the detector trained on top of `bruno-phase1`, completed all 100
  planned iterations.

Each folder has the model checkpoint(s), `args.json` (exact training configuration), and the
per-iteration training log (`phase1_log.csv` / `phase2_log.csv`, plus `console.log` where
available) used to produce the training-curve figures in `svg-inkscape/`.
