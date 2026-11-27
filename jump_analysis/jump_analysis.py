import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from scipy.ndimage import binary_opening, binary_closing, binary_dilation
from utils import get_spectrogram, track_ridge_tfridge_like
from config import DATA_PATH


def extract_raw_ridge(path):
    times, freqs, mag = get_spectrogram(path, n_fft=1024, hop_length=128)
    mask = (freqs >= 20000) & (freqs <= 100000)
    freqs, mag = freqs[mask], mag[mask]
    if mag.size == 0 or mag.shape[0] < 2:
        return times, freqs, mag, np.full(len(times), np.nan), np.zeros(len(times), bool)
    power = mag ** 2
    p = power / (power.sum(axis=0, keepdims=True) + 1e-12)
    entropy = -(p * np.log2(p + 1e-12)).sum(axis=0) / np.log2(len(freqs))
    if len(entropy) >= 3:
        entropy = np.convolve(entropy, np.ones(3) / 3, mode='same')
    active = binary_opening(entropy < .85, structure=np.ones(2))
    active = binary_closing(active, structure=np.ones(2))
    active = binary_dilation(active, structure=np.ones(5))
    ridge, amplitude = track_ridge_tfridge_like(mag, freqs, active, top_k=12,
                                                jump_penalty=.03, max_jump_hz=None)
    return times, freqs, mag, ridge, active


def inspect_file(path, threshold_khz, min_frames, out_dir):
    times, freqs, mag, ridge, active = extract_raw_ridge(path)
    good = np.isfinite(ridge) & active
    # Only adjacent frames within the same contiguous active segment.
    adjacent = good[1:] & good[:-1]
    changes = np.diff(ridge) / 1000
    candidates = np.flatnonzero(adjacent & (np.abs(changes) >= threshold_khz)) + 1
    rows = []
    for j in candidates:
        # Require stable ridge on either side to reduce one-frame tracking spikes.
        pre = ridge[max(0, j-min_frames):j]
        post = ridge[j:min(len(ridge), j+min_frames)]
        if len(pre) < min_frames or len(post) < min_frames or not (np.isfinite(pre).all() and np.isfinite(post).all()):
            continue
        pre_khz = float(np.median(pre) / 1000)
        post_khz = float(np.median(post) / 1000)
        if abs(post_khz - pre_khz) < threshold_khz:
            continue
        rows.append(dict(file=str(path), frame=int(j), time_ms=float(times[j]*1000),
                         pre_khz=pre_khz, post_khz=post_khz,
                         signed_jump_khz=post_khz-pre_khz,
                         jump_khz=abs(post_khz-pre_khz),
                         review_status='unreviewed'))
    # Plot only recordings with candidates; figures are for manual verification.
    if rows:
        fig, ax = plt.subplots(figsize=(11, 4))
        db = librosa.amplitude_to_db(mag, ref=np.max)
        ax.pcolormesh(times*1000, freqs/1000, db, shading='auto', cmap='magma', vmin=-60, vmax=0)
        ax.plot(times*1000, ridge/1000, color='cyan', linewidth=.8, label='Raw tracked ridge')
        for row in rows:
            ax.axvline(row['time_ms'], color='lime', alpha=.7, linewidth=1)
        ax.set(xlabel='Time (ms)', ylabel='Frequency (kHz)', title=path.name)
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(out_dir / f'{path.parent.name}_{path.stem}_review.png', dpi=140)
        plt.close(fig)
    return rows, dict(file=str(path), active_frames=int(good.sum()), candidate_jumps=len(rows),
                      status='included' if good.sum() >= 10 else 'low_signal_or_no_detected_usv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold-khz', type=float, default=10)
    parser.add_argument('--min-frames', type=int, default=2)
    parser.add_argument('--out', type=Path, default=Path('jump_analysis_output'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(set(Path(DATA_PATH).glob('train/**/*.WAV')) | set(Path(DATA_PATH).glob('test/**/*.WAV')))
    if not files:
        raise SystemExit(f'No WAV files found under {DATA_PATH}/train or test')
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(files))
    all_rows, audits = [], []
    for idx in shuffled:
        if sum(a['status'] == 'included' for a in audits) >= args.n:
            break
        path = files[int(idx)]
        try:
            rows, audit = inspect_file(path, args.threshold_khz, args.min_frames, args.out)
            audits.append(audit)
            if audit['status'] == 'included':
                all_rows.extend(rows)
        except Exception as exc:
            audits.append(dict(file=str(path), status='error', error=str(exc), active_frames=0, candidate_jumps=0))
    cols = ['file','frame','time_ms','pre_khz','post_khz','signed_jump_khz','jump_khz','review_status']
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(args.out / 'jump_candidates.csv', index=False)
    pd.DataFrame(audits).to_csv(args.out / 'recording_audit.csv', index=False)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df['pre_khz'], df['jump_khz'], alpha=.7)
        ax.set(xlabel='Pre-jump frequency (kHz)', ylabel='Jump magnitude (kHz)',
               title='Candidate frequency jumps (NOT YET VERIFIED)')
        fig.tight_layout()
        fig.savefig(args.out / 'jump_scatter.png', dpi=170)
        plt.close(fig)
    print(f"Screened {len(audits)} recordings; included {sum(a['status']=='included' for a in audits)}; "
          f"found {len(df)} unverified candidates.")
    print(f'Outputs: {args.out.resolve()}')
    print('Inspect review images and edit review_status to confirmed/artifact/uncertain before drawing conclusions.')


if __name__ == '__main__':
    main()