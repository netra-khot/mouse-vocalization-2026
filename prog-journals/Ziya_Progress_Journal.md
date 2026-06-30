# Ziya's Progress Journal

## June 29, 2026
**Duration:** 45 min

- More breakdown of contents in Dryad dataset
    - USV recordings: 65
    - Pleth files: 44
    - Opto files: 3
    - EMG files: 4
    - Other: 1
- Organized repo w/ folders for each model
- Added utils.py
    - Added get_spectrogram -> Use to return the stats (times, freqs, magnitudes) needed to create spectrgram
    - Added get_main_freq_traj
- Tested the get_main_freq_traj() method in model.ipynb (in freq-to-usv model)
    - Next step: need to improve function
        - If there's no sound in a certain area of the spectrogram, it should be balnk on the main-freq traj graph
            - Currently, I think it's hallucinating a pattern when there's really just no vocalization in the clip

## June 27, 2026
**Duration:**  1 h

- Downloaded the dryad dataset
    - Added it to this folder (+ sent ZIP to netra)
    - Replaced old dryad files (the ones I got from the API) which fixed some of the issues I had opening them
- Opened up some of the Dryad files
    - Made some metadata (file type splits)
    - Found out there's no syllable annotations

## June 22, 2026
**Time:** 4:30pm-5:30pm

**Duration:**  1 h

- Verified that we're working with the audio files (not LFS pointers)
- Played around with more EDA stuff
    - Generated random training data file's spectrogram & waveform
        - Load w/ librosa & plots an STFT spectrogram
    - Analyzed data @ TRAIN_PATH to write eda_train_inventory.csv w/ path, size_bytes, duration, and sr
    - Analyzed CSV w/ stats for size_bytes, duration, sr
    - Plotted duration & sr histograms w/ seaborn & matplotlib
    - Generated 4x4 grid of random usv's spectrograms

## June 18, 2026
**Time:** 2:00pm–4:45pm

**Duration:** 2h 45m
 
- Set up Git/GitHub workflow (cloning, gitignore, branches, tried working w/ LFS for large files)
- Resolved dataset access by pulling real audio from BiWaveGAN GitLab repo via Git LFS
- Set up local config file system so each collaborator can use their own data path without pushing it to GitHub
- Began EDA: confirmed sample rate (250kHz) and call duration (~40ms) on a single file
- Computed dataset-wide call duration distribution
- Checked RMS energy across files to screen for bad detections
- Confirmed train/test split is 90/10 and well-stratified by strain
- Identified strain imbalance (DBA: 20,265 train files vs. C57: 8,064 train files)
- Decided to filter dataset to DBA only for training + testing