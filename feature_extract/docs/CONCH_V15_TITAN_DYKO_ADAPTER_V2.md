# CONCH v1.5 + TITAN + DyKo Adapter 的 SlotSPE-v2 流程

这套流程位于 Git 分支 `CONCHv1.5`，用于 TCGA-KIRC 的 494 张 WSI。

```text
WSI → 512×512 @ 20x → resize 448 → CONCH v1.5 → patch tokens [N,768]
文本事件 → TITAN text encoder → event tokens [22,768]

patch tokens → Path Adapter ┐
                              ├→ v2 event evidence / event gate → SlotSPE
event tokens → Text Adapter ─┘
```

CONCH v1.5 patch token 与 TITAN text token 在预处理结束后只保证维度相同，不能直接视为同一语义空间。训练阶段使用两个独立的 `768→192→768` Adapter 和事件分布 KL 损失学习任务级可比较空间。这里不使用 TITAN slide encoder，也不加入 DyKo 的 FAISS KMeans、PromptLearner 或分类 cross-attention。

## 1. 创建独立预处理环境

现有 `titan` 环境存在 NumPy/h5py ABI 冲突，`slotspe` 环境的 Transformers 5 又不兼容 TITAN。新环境放到 `/data0`，避免继续占用几乎已满的 home 分区。

```bash
cd /home/liufangyi/proj/SLotSPE/SlotSPE
source /home/liufangyi/anaconda3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/data0/lfy_data/conda_pkgs
conda env create \
  --prefix /data0/lfy_data/conda_envs/slotspe-titan-v15 \
  --file feature_extract/envs/conch_v15_titan.yml

conda activate /data0/lfy_data/conda_envs/slotspe-titan-v15
```

作用：安装 TITAN 官方版本依赖，并提供 OpenSlide、h5py 和 OpenCV 所需的 WSI 环境。不会修改原来的 `slotspe` 或 `titan` 环境。

## 2. 检查本地模型与编码器

先用 `nvidia-smi` 选择一张至少约有 14 GiB 空闲显存的 GPU。下面以物理 GPU 0 为例：

```bash
cd /home/liufangyi/proj/SLotSPE/SlotSPE
conda activate /data0/lfy_data/conda_envs/slotspe-titan-v15

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/smoke_test_titan_local.py \
  --model /data0/lfy_data/Pathology/checkpoints/TITAN \
  --device cuda
```

作用：从本地目录加载 tokenizer、TITAN text encoder 和 CONCH v1.5 patch encoder，分别生成一个 `[1,768]` 单位向量。该命令不访问 Hugging Face。

## 3. 单张 WSI 预处理 smoke test

下载流程默认使用 `GDC_DIRECT=1`：调用 `gdc-client` 时会清除继承的
`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`（以及小写形式），并在首次下载前
直连 `https://api.gdc.cancer.gov/status`。因此可以在关闭 VPN 后运行，且不会
悄悄复用已经失效的代理端口。服务器已在 2026-07-20 以无代理方式实测该接口
返回 HTTP 200。

每张 WSI 最多尝试下载 5 次，失败间隔 30 秒。GDC 的部分下载文件不会被删除，
重新执行相同命令即可继续；已经生成并验证通过的 `.pt` 会自动跳过。

```bash
cd /home/liufangyi/proj/SLotSPE/SlotSPE
conda activate /data0/lfy_data/conda_envs/slotspe-titan-v15

START_BATCH=1 END_BATCH=1 \
GDC_DIRECT=1 GDC_DOWNLOAD_RETRIES=5 GDC_RETRY_SECONDS=30 \
GPU_WAIT_CANDIDATES=0,1,2,3,4 GPU_MIN_FREE_MIB=14000 \
bash scripts/run_kirc_conch_v15_preprocess.sh
```

作用：下载 manifest 中第一张 WSI，生成 `512×512 @ 20x` 非重叠 patch 坐标，将图像缩放到 448 后提取 CONCH v1.5 特征，验证 `[N,768]`、float16、有限值和单位范数，再删除已验证的原始 WSI。

查看结果：

```bash
find /data0/lfy_data/Pathology/CONCH_v1.5/kirc/pt_files \
  -maxdepth 1 -type f -name '*.pt' | wc -l

tail -20 \
  /data0/lfy_data/Pathology/Slides/kirc_conch_v15_work/logs/batch_status.tsv
```

## 4. 流式处理全部 494 张 WSI

当前 UNI 任务在等待 GPU 5/6/7，因此这里默认只等待 GPU 0–4，避免两个任务同时抢到同一张卡。

```bash
cd /home/liufangyi/proj/SLotSPE/SlotSPE
source /home/liufangyi/anaconda3/etc/profile.d/conda.sh

tmux new-session -d -s kirc_conch_v15 \
  "cd /home/liufangyi/proj/SLotSPE/SlotSPE && \
   source /home/liufangyi/anaconda3/etc/profile.d/conda.sh && \
   conda activate /data0/lfy_data/conda_envs/slotspe-titan-v15 && \
   GDC_DIRECT=1 GDC_DOWNLOAD_RETRIES=5 GDC_RETRY_SECONDS=30 \
   GPU_WAIT_CANDIDATES=0,1,2,3,4 GPU_MIN_FREE_MIB=14000 \
   bash scripts/run_kirc_conch_v15_preprocess.sh 2>&1 | \
   tee -a /data0/lfy_data/Pathology/Slides/kirc_conch_v15_work/logs/run.log"
```

作用：逐张执行“下载 → 分割/坐标 → 特征 → 验证 → 清理”。已经存在且验证通过的 `.pt` 会自动跳过，前面的 smoke test 不会重复计算。

监控：

```bash
tmux attach -t kirc_conch_v15

tail -f \
  /data0/lfy_data/Pathology/Slides/kirc_conch_v15_work/logs/run.log

watch -n 30 "find /data0/lfy_data/Pathology/CONCH_v1.5/kirc/pt_files \
  -maxdepth 1 -type f -name '*.pt' | wc -l"
```

从指定 batch 恢复，例如从第 37 张开始：

```bash
START_BATCH=37 \
GDC_DIRECT=1 GDC_DOWNLOAD_RETRIES=5 GDC_RETRY_SECONDS=30 \
GPU_WAIT_CANDIDATES=0,1,2,3,4 GPU_MIN_FREE_MIB=14000 \
bash scripts/run_kirc_conch_v15_preprocess.sh
```

若日志出现 `cannot reach GDC directly`，这表示主机本身的 DNS 或防火墙无法直连
GDC，而不是 CONCH/TITAN 的问题。修复网络后重新执行同一命令即可，不需要重新
处理已经完成的 WSI。只有明确希望继承当前代理设置时才使用 `GDC_DIRECT=0`。

最终严格验证：

```bash
python tools/validate_conch_features.py \
  --feature-dir /data0/lfy_data/Pathology/CONCH_v1.5/kirc/pt_files \
  --manifest feature_extract/tools/gdc/gdc_manifest_kirc_slotspe_494.txt \
  --expected-dim 768 \
  --expected-dtype float16 \
  --allow-missing 0 \
  --report /data0/lfy_data/Pathology/CONCH_v1.5/kirc/validation_report.json
```

验收条件是 `expected_slides=494`、`valid_slides=494`、无 missing、无 invalid。若某张分割失败，原始数据和中间文件会保留，并记录在 `failed_slides.tsv`，处理完失败样本后再执行严格验证。

## 5. 构建 TITAN v2 文本事件库

选择一张空闲 GPU；以下仍以物理 GPU 0 为例：

```bash
cd /home/liufangyi/proj/SLotSPE/SlotSPE
conda activate /data0/lfy_data/conda_envs/slotspe-titan-v15

PYTHON=/data0/lfy_data/conda_envs/slotspe-titan-v15/bin/python \
TITAN_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
bash scripts/build_tcga_kirc_event_bank_v2_conch_v15.sh
```

作用：使用 TITAN text encoder 对 `events_patch_core.json` 中的 22 个 v2 事件进行 prompt ensemble，生成：

```text
assets/event_bank/tcga_kirc/v2/titan_text_event_bank_v2.pt
assets/event_bank/tcga_kirc/v2/titan_text_event_bank_v2_summary.json
```

事件库是 `[22,768]`、float32、L2 归一化，并标记为 `titan_text_768`。当前事件集合仍处于 `machine_curated_pending_pathologist_review`，程序会保留这一警告。

## 6. 运行代码测试

模型训练仍使用原来的 `slotspe` 环境，因为训练只读取预计算 token，不加载 TITAN。

```bash
cd /home/liufangyi/proj/SLotSPE/SlotSPE
conda activate slotspe

python -m unittest \
  tests.test_vl_adapter \
  tests.test_event_gated_slot_attention \
  tests.test_conch_evidence \
  tests.test_conch_event_bank \
  -v

python -m unittest discover -s tests -v
```

作用：验证 Adapter 的维度、归一化、padding、双分支梯度，KL 对齐损失，v2 forward/backward，以及旧 SlotSPE/CONCH v1 的兼容性。

## 7. 启动 Adapter-v2 训练

先用一个 fold 和一个 epoch 做真实数据 smoke test：

```bash
cd /home/liufangyi/proj/SLotSPE/SlotSPE
conda activate slotspe

GPU=0 MAX_EPOCHS=1 BATCH_SIZE=2 \
SPECIFIC_SIMPLE=conch_v15_titan_adapter_v2_smoke \
bash scripts/run_kirc_conch_v15_titan_adapter_v2.sh \
  --k_start 0 --k_end 1
```

完整 5-fold 训练：

```bash
GPU=0 MAX_EPOCHS=30 BATCH_SIZE=32 SEED=3 \
LAMBDA_VL_ALIGNMENT=0.1 \
SPECIFIC_SIMPLE=conch_v15_titan_dyko_adapter_v2 \
bash scripts/run_kirc_conch_v15_titan_adapter_v2.sh
```

训练附加损失为：

```text
L_aux = λ_decoder L_decoder
      + λ_recon L_reconstruction
      + 0.1 KL(p_semantic || p_visual)
```

`p_semantic` 与 `p_visual` 都不 detach，因此 Path Adapter 和 Text Adapter 都会更新；CONCH v1.5、TITAN 和原始事件库不会更新。

## 8. 消融命令

关闭 KL、仅依靠生存损失和 v2 门控训练 Adapter：

```bash
GPU=0 LAMBDA_VL_ALIGNMENT=0 \
SPECIFIC_SIMPLE=conch_v15_titan_adapter_no_kl \
bash scripts/run_kirc_conch_v15_titan_adapter_v2.sh
```

测试不同 KL 权重时，只使用验证 fold 选择参数：

```bash
for weight in 0.01 0.1 1.0; do
  GPU=0 LAMBDA_VL_ALIGNMENT="${weight}" \
  SPECIFIC_SIMPLE="conch_v15_titan_adapter_kl_${weight}" \
  bash scripts/run_kirc_conch_v15_titan_adapter_v2.sh
done
```

不要直接使用 `vl_adapter_type=none` 比较 CONCH v1.5 patch 与 TITAN 文本；模型会主动报错。无事件门控的 CONCH v1.5 原始 SlotSPE 可以作为视觉表征基线，但它不做 WSI–文本比较。

## 9. 主要输出位置

```text
WSI patch features:
/data0/lfy_data/Pathology/CONCH_v1.5/kirc/pt_files

预处理日志与状态：
/data0/lfy_data/Pathology/Slides/kirc_conch_v15_work/logs

TITAN text event bank:
assets/event_bank/tcga_kirc/v2/titan_text_event_bank_v2.pt

训练结果与 checkpoint:
results_train/kirc/SlotSPE/
```
