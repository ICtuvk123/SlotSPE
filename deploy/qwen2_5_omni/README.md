# Local Qwen2.5-Omni Deployment

This directory contains a local deployment scaffold for Qwen2.5-Omni.

Current machine check from this workspace:

- `nvidia-smi` cannot communicate with an NVIDIA driver.
- `ffmpeg` is not on `PATH`.
- `docker` is not on `PATH`.
- The filesystem has about 49 GB free.

That means the model can be prepared here, but practical local inference needs a working NVIDIA driver and enough GPU memory. The scripts default to `Qwen/Qwen2.5-Omni-3B` because it is the better fit for this disk budget. Use `Qwen/Qwen2.5-Omni-7B` only if you have enough disk and GPU memory.

## 1. Create the Environment

From the repository root:

```bash
bash deploy/qwen2_5_omni/setup_env.sh
```

This creates a separate `qwen-omni` conda environment, installs PyTorch CUDA 12.1 wheels, Qwen's compatible Transformers build, Qwen Omni utilities, Gradio, and `ffmpeg`.

If your system uses a different CUDA runtime, edit `setup_env.sh` before running it.

## 2. Download the Model

```bash
conda activate qwen-omni
bash deploy/qwen2_5_omni/download_model.sh Qwen/Qwen2.5-Omni-3B
```

For the 7B model:

```bash
bash deploy/qwen2_5_omni/download_model.sh Qwen/Qwen2.5-Omni-7B
```

Downloaded files are stored under `deploy/qwen2_5_omni/models/` by default.

## 3. Run a Smoke Test

```bash
conda activate qwen-omni
python deploy/qwen2_5_omni/smoke_test.py \
  --model-path deploy/qwen2_5_omni/models/Qwen2.5-Omni-3B \
  --prompt "Say hello in one sentence." \
  --no-audio
```

The test intentionally supports text-only input first. Add `--image`, `--audio`, or `--video` after the base deployment works.

## 4. Launch the Local Web UI

```bash
conda activate qwen-omni
bash deploy/qwen2_5_omni/web_demo.sh deploy/qwen2_5_omni/models/Qwen2.5-Omni-3B
```

Then open:

```text
http://127.0.0.1:7860
```

## Notes

- Qwen's official model card recommends installing a Qwen-compatible Transformers build to avoid `KeyError: 'qwen2_5_omni'`.
- Qwen's official memory table lists BF16 minimums around 18.38 GB for Qwen2.5-Omni-3B on 15 second video and 31.11 GB for Qwen2.5-Omni-7B on 15 second video, with practical usage usually higher.
- Audio/video input requires `ffmpeg`.
- FlashAttention 2 is optional but recommended on supported NVIDIA GPUs. Install it manually inside `qwen-omni` if your CUDA/PyTorch build supports it:

```bash
pip install flash-attn --no-build-isolation
```

