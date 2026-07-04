# Netra's Progress Journal #
## ENTRY 7 &rarr; 07/04/2026 (1.5 hours)
### 2:30pm - 4:00pm
- TODO need to reorganize code in audio_preprocess_01.ipynb and work out folder organization for utils.py
- moved utils.py into dataset_preparation folder for easier usage
- The extracted main frequency trajectory from DBA_9858_542.WAV  (chosen randomly) had a bad jump at the start
- Wou   
## ENTRY 6 &rarr; 07/02/2026 (2.5 hours)
### 3:30pm - 6:00pm
- Does the data need filtering --> need to verify dataset integrity
    - I compared the raw spectrograms across the train/test samples
        - sample rate indicates consistent 250 kHz
        - Noise floor is broadband, meaning that it is present across the full spectrum rather than concentrated outside the USV range
            - filtering will help remove noise below 25 kHz/above 110 kHz but won't fully clean noise overlapping the USV band itself
- Started the main freq traj extraction --> pull out the dominant frequency over time from each USV
    - Used ziya's get_main_freq_traj from freq-to-usv-model/utils.py
## ENTRY 5 &rarr; 07/01/2026 (1 hour)
### 9:30pm - 10:30pm
- Started data preparation
- Working on constructing the gitlab dataset
    - Specifically worked on audio selection and preprocessing
- First few kernels are imports and some data analysis for future processing
    - loading data inventory ~22,000 filel w/ train and test
- Printing sample rate, duration, and the channel audit
- loaded data file and plotted the raw waveform
- Restructured files to have data preparation as separate folder
## ENTRY 4 &rarr; 06/30/2026 (1 hour)
### 9:30pm - 10:30pm
- Started data 
- Working on constructing the gitlab dataset
    - Specifically worked on audio selection and preprocessing
- First few kernels are imports and some data analysis for future processing
    - loading data inventory ~22,000 filel w/ train and test
- Printing sample rate, duration, and the channel audit
- loaded data file and plotted the raw waveform
- Restructured files to have data preparation as separate folder
## ENTRY 3 &rarr; 06/29/2026 (2.5 hour)
### 4:00pm - 6:30pm
- Tentatively choosing state-space models
    - Explored neural SSMs and linear SSMs for an interpretable baseline
- Changed file name to neural_ssm.testing.ipynb
- Added configs 
## ENTRY 2 &rarr; 06/28/2026 (1 hour)
### 4:45 - 5:45pm
- Worked through different model 1 architectures and families
    - Narrowed down to hierarchical conditional model and latent dynamical system
    - Hierarchical systems model the brainstem well and latent systems treat muscle data as an evolving state
        - Learning towards latent dynamical system
- Added latent_dynamical_testing.ipynb
- Added initial imports and packages needed in general

## ENTRY 1 &rarr; 06/18/2026 (3.5 hours)
### 2:00 - 4:45pm
- Set up VSCode and GitHub repository (added journal folders, data folders, set up .gitignore, etc.)
- Looked through data &rarr; imbalance of C57 and DBA strain (DBA in favor)
    - Decided to remove all C57 files to avoid bias
- Performed EDA on data
    - inspected raw data
    - checked dataset stucture
    - visualized waveform with time domain
### 10:45 - 11:30pm
- More EDA on data
    - Duration distribution
    - Converted waveform to spectrogram
