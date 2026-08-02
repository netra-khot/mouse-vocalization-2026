# Netra's Progress Journal #
## ENTRY 20 &rarr; 07/31/2026 (1 hour)
### 9:00pm - 10:00pm
- saved all contours to a pickle file
- finalized the mft algorithm by removing all gaps in the contour (candidate should never be NaN or 0)
- removed vocalizations with no visible main frequency trajectory (cleaned_file_df contains 21861 files whereas original file_df contains)
- checked that pickle file contained correct file type, plotting, # of files
## ENTRY 19 &rarr; 07/27/2026 (3 hours)
### 9:30pm - 12:30am
- maybe let's try clustering on our own! it will be way faster bc its running on vscode and not chrome...
- looking at clustering algorithms, k-means and PCA will be good to cluster and reduce dimensionality for vector quantities of mfts
- pipeline kinda goes use get main freq traj method, cluster using above methods, form the cluster groups (lots of extra steps in between obv)
- run 20 files, 100 files, 1000 files, and clusters are pretty even (270 - 330)
    - looks like the common clusters are upward, flat, chevron, u-shaped, complex?, and down (separate but similar to flat)
- this algorithm works decently well (compared the cluster average shape with the files it clustered inside that group, the files i checked in clusters have the same features)
- may want to try deepsqueak again bc that like actually works and is validated properly but talk to Dr. Tripp first
## ENTRY 18 &rarr; 07/25/2026 (1 hour)
### 5:30 - 6:30pm
- started working on detection of the rest of the files
    - deepsqueak will end up taking a few days to process all 22k files, also acknowledging that the online session must be active
- install computer vision tool
## ENTRY 17 &rarr; 07/23/2026 (2 hours)
### 10:00 - 12:00am
- Loaded the online MATLAB with the DeepSqueak zip (after many failures)
    - GUI successfully opened
- copied our data, imported into the DeepSqueak audio folder
- started processing ~15 audio files (first 15 in train)
    - grouped the flat syllables together (clustering without formal label) still need to check other files
## ENTRY 16 &rarr; 07/21/2026 (1.5 hours)
### 3:00 - 4:30pm
- Evaluated VocalMat vs. DeepSqueak, going with DeepSqueak
   - DeepSqueak adjusts sample rate based on recording
   - Accepts raw .wav files, saves a lot of the rescaling work for us
   - Will be easier to set up contour detection validation to evaluate MFT algorithm
- The classifier automatically clusters the data based on shape, but we have to physically assign specific syllable names
   - Make sure to run the labels by Dr. Tripp before finalizing
## ENTRY 15 &rarr; 07/20/2026 (1.5 hours)
### 3:00 - 4:00pm
- Added method for exporting the MFTs (export_mft)
   - currently going with .csv files, but may be subject to change if needed? After some research it seems they're a good option for Python or MATLAB
- Evaluated best approaches for syllable classification, VocalMat offers 12 syllable types including noise, so it may be better for long-term usage
### 11:30pm - 12:00am
- Ran a couple files through VocalMat, but because our data is scaled very differently and is in a different format, we'd need to jump through more hoops to get it to work?
- Will run files through DeepSqueak tomorrow because even though it only has 5 syllables available, it is trained on .wav --> spectrogram files and our data may be easier to rescale to DeepSqueak rather than VocalMat
## ENTRY 14 &rarr; 07/19/2026 (2.5 hours)
### 3:00 - 5:00pm
- Replaced someo of the original greedy main frequency trajectory tracker with a Viterbi-style ridge tracker to improve continuity between competing frequency ridges
- Did some tuning spectral entropy thresholds
    - entropy_threshold, jump_penalty, hop_length, silence_value, and more
        - found that jump_penalty at 0.08 follows frequency jumps better than smaller values
        - entropy_threshold is doing well at the 0.75-0.8 range, currently at 0.76
- Added some exception handling if files are deleted issues for future usage (thinking ahead i guess!)
## ENTRY 13 &rarr; 07/18/2026 (45 mins)
### 5:00pm - 5:45pm
- note that files like test --> DBA_3245_821.WAV shouldn't be producing an mft because the signal is very faint
    - this one specifically may be 2 different events instead of a single vocalization, so it is better if the algorithm rejects the recording for now until the algorithm becomes stronger (?)
## ENTRY 12 &rarr; 07/15/2026 (30 mins)
### 2:30pm - 3:00pm
- causes of hallucinations:
    - continuity penalty for freq jumps is too strong
    - entropy detector is probably activating a few frames too early, so the tracker latches onto noise
- reducing jump penalty to allow bigger jumps when the first ridge fades out
    - jump_pentalty started at 1e-7, changed around to 1e-8, 1e-9, 1e-10   
## ENTRY 11 &rarr; 07/13/2026 (1 hour)
### 9:30pm - 10:30pm
- changed entropy threshold to ranges of 0.4-0.8 (hundreths place) 
    - by changing the raw value itself, it seems that 0.74-0.78 yield the greatest improvments in extracting the full MFT in weak onsets
- changing entropy alone may not be enough to see complete mfts for cases like weak onsets, so we can try adding morphological opening, closing, and dilation
    - Goal is to remove isolated detections, fill small gaps, and extend detected vocalization regions
        - so the extracted MFT matches the spectrogram better
## ENTRY 10 &rarr; 07/10/2026 (1.5 hours)
### 12:30pm - 2:00pm
- Mapped out full algorithm pipeline &rarr;
    - Convert each .WAV recording into a spectrogram.
        - Apply a bandpass filter to isolate USV frequency range of 30-110 kHz
        - Detect which time bins contain vocalization using Shannon spectral entropy
        - Use a tfridge-inspired peak frequency tracker 
        - Save the strongest appearances (falling back to maximum-amplitude frequency bin if no local peaks are found).
        - Track frequency ridge by scoring each appearance based on amplitude and its continuity with the previously selected frequency
        - Save the extracted MFT
- for validation, not many options and may be clunky but something is better than nothing at this stage
    - use contour detection (like mft) through DeepSqueak and do a visual comparison (or somehow extract the math function)
    - visually draw over spectrograms then overlap the drawn vs calculated mfts
## ENTRY 9 &rarr; 07/08/2026 (2.25 hour)
### 2:30pm - 4:45pm
- After brainstem group call, started working on getting in depth understanding of algorithm
- What are the two functions used in the code? 
    - tfridge: a built-in MATLAB function used for signal processing. It extracts the "ridges", or paths of maximum energy, from a time-frequency representation, such as a spectrogram
    - Shannon spectral entropy: outputs entropy from time bins &rarr;
        - Low entropy indicates that a USV is there
        - High entropy means that there may be silence or broadband noise
- Forked DeepSqueaks github repo and looked around for syllable labeling functions
## ENTRY 8 &rarr; 07/06/2026 (2 hours)
### 12:30pm - 2:30pm
- Started writing main-frequency trajectory algorithm for Task 1
- Goal is to produce an algorithm that is interpretable by the Hakansson et al. model and accounts for all types of USVs
    - USVs to look out for: weak onsets, multiple peaks, frequency jumps, complex syllables
- Rewrote existing get_main_freq_traj to use functions implemented in Hakansson model (tfridge and Shannon entropy function)
    - TODO need to gain a better understanding of how the algorithm pipeline actually works rather than knowledge of input --> algorithm --> output
- tfridge-inspired peak tracker is used for getting the candidate peaks
    - track_ridge_tfridge_like method (should probably workshop this name)
## ENTRY 7 &rarr; 07/04/2026 (1.5 hours)
### 2:30pm - 4:00pm
- TODO need to reorganize code in audio_preprocess_01.ipynb and work out folder organization for utils.py
- moved utils.py into dataset_preparation folder for easier usage
- The extracted main frequency trajectory from DBA_9858_542.WAV  (chosen randomly) had a bad jump at the start
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
