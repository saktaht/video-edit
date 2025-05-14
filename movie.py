import os
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from glob import glob
from itertools import chain
# threshold: 音の大きさ None なら自動推定 (20% quantile)
# min_duration:  音がこの時間以上続いたら音と認識する 推奨: 0.25~0.4 
# PADDING: カットの前後の余白 推奨: 0.15~0.25

def estimate_threshold_quantile(volumes: np.ndarray, q: float = 0.20) -> float:
    """
    下位 q 分位点をしきい値として返す簡易推定器。
    """
    return float(np.quantile(volumes, q))


def estimate_threshold_kmeans(volumes: np.ndarray) -> float:
    """
    k-means (k=2) で静音クラスタと有音クラスタに分け、
    静音クラスタの中心＋αをしきい値にする。
    scikit-learn がインストールされていない場合は呼び出さないこと。
    """
    try:
        from sklearn.cluster import KMeans
    except ModuleNotFoundError:
        # fallback – quantile 20% 相当を返す
        return estimate_threshold_quantile(volumes, q=0.15)

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(volumes.reshape(-1, 1))
    quiet, loud = sorted(km.cluster_centers_.flatten())
    return quiet + 0.15 * (loud - quiet)

# def remove_silence(video_path, output_path, threshold=None, min_duration=0.3, quantile_q=0.20, use_kmeans=False):
def remove_silence(
    video_path,
    output_path,
    threshold: float | None = None,
    min_duration: float = 0.3,
    quantile_q: float = 0.20,
    use_kmeans: bool = True,
):
    video = VideoFileClip(video_path)
    audio = video.audio
    fps = audio.fps
    # --- MoviePy + NumPy 2.x workaround ---
    # audio.iter_frames() returns a generator; convert to list before stacking
    frames = list(audio.iter_frames())
    samples = np.vstack(frames)
    volume = np.mean(np.abs(samples), axis=1)

    # 動画ごとに自動でしきい値を推定
    if threshold is None:
        if use_kmeans:
            threshold = estimate_threshold_kmeans(volume)
        else:
            threshold = estimate_threshold_quantile(volume, q=quantile_q)
    print(f"[{os.path.basename(video_path)}] threshold = {threshold:.5f}")

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
    PADDING = 1.0  # 秒
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