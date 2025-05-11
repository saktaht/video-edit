import os
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from glob import glob
from itertools import chain
# threshold: 音の大きさ(値が小さいほど小さい音を拾う) 推奨: 0.008~0.015
# min_duration:  音がこの時間以上続いたら音と認識する 推奨: 0.25~0.4 
# PADDING: カットの前後の余白 推奨: 0.15~0.25

def remove_silence(video_path, output_path, threshold=0.009, min_duration=0.3):
    video = VideoFileClip(video_path)
    audio = video.audio
    fps = audio.fps
    # --- MoviePy + NumPy 2.x workaround ---
    # audio.iter_frames() returns a generator; convert to list before stacking
    frames = list(audio.iter_frames())
    samples = np.vstack(frames)
    volume = np.mean(np.abs(samples), axis=1)

    # 平滑化（0.1秒単位）
    window_size = int(fps * 0.1)
    vol_smoothed = np.convolve(
        volume, np.ones(window_size) / window_size, mode="valid"
    )
    mask = vol_smoothed > threshold
    # 各スライディングウィンドウの中心時刻（秒）
    times = (np.arange(len(vol_smoothed)) + window_size // 2) / fps

    # 音がある区間を検出
    intervals = []
    start = None
    for i, is_loud in enumerate(mask):
        if is_loud and start is None:
            start = times[i]
        elif not is_loud and start is not None:
            if times[i] - start > min_duration:
                intervals.append((start, times[i]))
            start = None
    if start is not None:
        intervals.append((start, times[-1]))

    # --- 追加: 前後に余白を入れてブツ切り感を緩和 ---
    PADDING = 0.2  # 秒
    padded = []
    for s, e in intervals:
        s = max(0, s - PADDING)
        e = min(video.duration, e + PADDING)
        # 直前の区間と重なっていればマージ
        if padded and s <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(padded[-1][1], e))
        else:
            padded.append((s, e))
    intervals = padded

    if not intervals:
        print(f"❌ 無音すぎてスキップ: {os.path.basename(video_path)}")
        return

    clips = [video.subclip(s, e) for s, e in intervals]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # files = glob(os.path.join(input_dir, "*.mov"))
    patterns = ["*.mov", "*.MOV", "*.mp4", "*.MP4"]
    files = list(chain.from_iterable(glob(os.path.join(input_dir, p)) for p in patterns))
    for f in files:
        name = os.path.basename(f)
        output_path = os.path.join(output_dir, name)
        print(f"▶️ 処理中: {name}")
        remove_silence(f, output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("使い方: python movie.py 入力ディレクトリ 出力ディレクトリ")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    batch_process(input_dir, output_dir)
