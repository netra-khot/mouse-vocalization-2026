# Ziya's Progress Journal

## Jul 20, 2026
**Duration:** 1h 45m
- Ported 3 more files
    - airflow, USVfreq, jet_speed
- made two separate functions for jet_speed
    - source code had like completely separate logic depending on if it was pressure or airflow based
- replaced redundant code w/ function calls
- fixed bug relating to elementwise ops vs matrix ops
    - I did edit the Hakanssons MATLAB code for this but nothing that would change the actual values

## Jul 19, 2026
**Duration:** 1h 45m

- Oopsies I have not updated this in a while
- Ported 3 files in the QMC_mouseUSV repo
    - subglottal_pressure, glottal_area, and impingement_length
- Installed Octave to run the MATLAB files
    - Took me a while to figure out how to use + navigate this
- Organized the repo w/ folders
    - Turned the pytorch folder into an importable package
- Took notes on the purpose + processes of each the three ported files on google docs
- Verified all the ported files against their MATLAB counterparts

## Jul 1, 2026
**Duration:** 1h

- Worked with the forked QMC_ratUSV repo
- Trying to go from MATLAB -> PyTorch
    - Started off with the subglottal_pressure.m file
    - I think I now have a decent enough grasp of the math going on to move forward
        - Made notes on a Google Doc
        - In short, they're using linear interpolation to calculate the subglottal pressure from the RM_activity
- Ran into an issue: all of the physiological constants are for rats, not mice!!
    - Identified six variables which we'd need to find the values for in terms of mice
        - pressure_max, area_cart_max, area_memb_max, length_max, length_min, tracheal_area
        - (also have these on Google Doc w/ better description + notes)
    - I think I found the values for mice for three of these? -> Most were in the Mahrt et al. 2016 paper
        - Still need to triple check this, b/c these values are really important for the model
    - Not sure where I'm going to find the other values
        - Will email Dr. Tripp by tmrw if it looks like a dead end

## June 29, 2026
**Duration:** 3 h

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