import os
import shutil
from pathlib import Path
import librosa
import soundfile as sf

def process_audio(input_path: str, output_path: str, speed: float, threshold_sec: float):
    """
    如果音檔長度超過 threshold_sec，則以 speed 倍速加速，否則直接複製檔案。
    回傳 (handled: bool, orig_sec: float)。
    """
    # 1. 取得音檔原始長度（秒）
    orig_sec = librosa.get_duration(filename=input_path)
    if orig_sec > threshold_sec:
        # 2. 載入完整音訊
        y, sr = librosa.load(input_path, sr=None)
        # 3. 時間伸縮
        y_fast = librosa.effects.time_stretch(y, rate=speed)
        # 4. 輸出
        sf.write(output_path, y_fast, sr)
        return True, orig_sec
    else:
        # 直接複製
        shutil.copy2(input_path, output_path)
        return False, orig_sec

def batch_process_folder(input_dir: str, output_dir: str, speed: float = 1.2, threshold_sec: float = 130.0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mp3_file in input_dir.glob("*.mp3"):
        out_file = output_dir / mp3_file.name
        handled, orig_sec = process_audio(str(mp3_file), str(out_file), speed, threshold_sec)
        if handled:
            new_sec = orig_sec / speed
            print(f"✓ {mp3_file.name}: {orig_sec:.1f}s → {new_sec:.1f}s (×{speed})")
        else:
            print(f"⧗ {mp3_file.name}: {orig_sec:.1f}s (無需加速)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批次處理：超過 2 分 10 秒的 MP3 加速 1.2 倍")
    parser.add_argument("input_folder", help="來源 MP3 資料夾")
    parser.add_argument("output_folder", help="輸出 MP3 資料夾")
    args = parser.parse_args()

    # 門檻 2 分 10 秒 = 130 秒，速度 1.2 倍
    batch_process_folder(args.input_folder, args.output_folder, speed=2, threshold_sec=130.0)
