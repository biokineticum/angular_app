# angular_app — Angular velocity / pose (Kivy)

**Prototype Android-oriented / desktop Kivy app** for **joint angle analysis from video** using a TensorFlow Lite pose model.

For physios and students exploring on-device pose angles (elbows, knees) without a full lab stack.

## What it does

Single script: `realtime_pose_kivy 4.py`

- Loads a `.tflite` pose model and an `.mp4` video via Kivy file choosers
- Estimates keypoints with TFLite Interpreter
- Computes angles for: prawy/lewy łokieć, prawe/lewe kolano
- Overlays pose and plots angles (OpenCV + Matplotlib + Kivy)

## Requirements

- Python 3 with: `kivy`, `opencv-python` (`cv2`), `numpy`, `matplotlib`, `tensorflow` (TFLite Interpreter)

## Quick start

```bash
git clone https://github.com/biokineticum/angular_app.git
cd angular_app
pip install kivy opencv-python numpy matplotlib tensorflow
python "realtime_pose_kivy 4.py"
```

Select a `.tflite` model and `.mp4` video in the UI, choose a joint, then run analysis.

> Note: filename contains a space — quote it on the command line as above.

---

## Author

**Dariusz Mosler** (Biokineticum) — physiotherapy, biomechanics, sports science  
🌐 [biokineticum.com](https://biokineticum.com) · 📰 [Substack](https://biokineticum.substack.com) · 𝕏 [@Biokineticum](https://x.com/Biokineticum)

