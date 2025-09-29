import os, argparse, numpy as np, sounddevice as sd, soundfile as sf, time


def record_clips(device_name: str, count: int, seconds: float, sr: int = 16000):
    outdir = os.path.join("data", device_name); os.makedirs(outdir, exist_ok=True)
    print(f"Recording {count} clips x {seconds}s to {outdir} @ {sr}Hz. Ctrl-C to stop.")
    try:
        for i in range(1, count+1):
            print(f"[{i}/{count}] Starts in 1s… speak normally."); time.sleep(1)
            audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait()
            x = audio.squeeze(); rms = float(np.sqrt(np.mean(x**2)) + 1e-8); print(f"RMS {rms:.4f}")
            sf.write(os.path.join(outdir, f"clip_{i:02d}.wav"), x, sr)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True); ap.add_argument("--count", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args(); record_clips(args.device, args.count, args.seconds)
