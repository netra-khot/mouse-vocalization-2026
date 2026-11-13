-rc# Context: Jump tracker / MFT extraction work (from 11/12 meeting)

## Project background
- Goal: an end-to-end model (brainstem circuit → muscle model → jet length/speed → USV) that reproduces a dataset of mouse USVs. Works roughly like an autoencoder: spectrogram in, reconstructed spectrogram out.
- USV frequency depends on 3 things: **mode number**, **jet length** (tracheal muscle), and **jet speed** (respiratory muscle). The frequency trajectory is extractable, but the three factors are confounded.
- Jumps happen on a shorter time scale than muscle control, so the model needs a **stochastic jump model**. Smooth frequency changes = coordinated changes in velocity and jet length (under muscle control).
- USVs are about 60–100 ms long and embedded in exhalations. Velocity over an exhale usually rises, then falls; a USV can fall on the rising part, the falling part, or the middle.
- Spectrogram color = acoustic power. It's probably not calibrated across recordings (mouse moves around), but variation within one USV may correlate with flow speed (non-linearly).
- Placeholder assumptions for now: **constant velocity** plus a **fixed typical mode number**.

## Known issue
The current tracker follows the *strongest frequency in each bin*. When two ridges coexist, it flips between them based on small fluctuations, which **inflates the jump count**. The model trains on tracker output as ground truth, so tracker errors propagate into the model.

## Tasks
1. Redo jump statistics with **bottom-ridge** and **top-ridge** trackers.
2. Implement a **dual-ridge tracker**: take the max peak per time bin, plus any peak ≥ ~0.5× its amplitude, as simultaneous frequencies. Describe changes as ridge **appearance/disappearance** events rather than jumps. Require min separation (~3 kHz) and min persistence (2–3 frames).
3. **Mode-number estimate**: n ≈ f_lower / |Δf| (assumes a one-mode jump). It should be close to an integer. Find the typical n (expected 4–5) and use it as a placeholder. With dual ridges, f_low / (f_high − f_low) gives n without needing a jump in time.
4. **Non-integer ratios**: check the spectrogram first to confirm tracking accuracy. A possible cause is a skipped mode (for example 3→5).
5. **Save ridge amplitude** along each track, for a later test of amplitude vs. flow speed.
6. Make overlay plots: spectrogram + all three tracker outputs (+ model prediction). Send them to the adviser.
7. (Model side) Find which jumps the model fails on, and check how failures correlate with muscle activity. Is the model "right for the wrong reason," for example reading a jump as the exhale slowing down?
8. Check the source dataset paper for recording setup and calibration. Email Andrea (cc adviser) a detailed project summary.

## Stats to compute per tracker (same calls for all three)
Jumps/events per call · jump-size distribution · % up vs. down · histogram of f_lower/|Δf| · % within ±0.15 of an integer · most common n · jump rate by starting frequency band (the adviser noted more frequent jumps from extreme frequencies) · jump timing within the call.

## Open questions
- Adviser: Is a one-mode jump assumption OK? Should non-integer ratios be rounded, discarded, or treated as skips? Is 0.5× the right dual-ridge threshold? Does he want stats on jumps, on appearance/disappearance events, or both?
- Andrea (fluids): Are some mode numbers physically more likely or stable (a prior)? How should amplitude relate to jet speed and jet length?

## Time-boxed plan (1 hr)
0–10 shared jump-stats function · 10–25 dual-ridge tracker · 25–40 run all 3 + summary table · 40–50 visual check of ~10 calls (including non-integer cases) · 50–60 write-up/email.
