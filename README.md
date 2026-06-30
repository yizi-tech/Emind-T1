<div align="center">
  <img src="/logo.png" alt="Emind-T1 AI Logo" width="120">
  <h1>Emind-T1 AI v3.0</h1>
  <h3>亦梓·智脑 — 面向 423B 大模型的生产级训练框架</h3>
  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.0%2B-orange" alt="PyTorch 2.0+"></a>
    <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-3.0.0-green" alt="V3.0.0"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-个人研究免费-lightgrey" alt="License"></a>
  </p>
  <p><strong>架构</strong>:GQA+SWA · head_dim 解耦 · MoE(fine-grained+shared+aux-loss-free) · MTP · 128K 上下文</p>
  <p>
    <a href="https://modelscope.cn/models/fuzhen/emind-1.5b">ModelScope</a> ·
    <a href="docs/V3_TRAINING_GUIDE.md">训练指南</a> ·
    <a href="CHANGELOG.md">更新日志</a> ·
    <a href="CONTRIBUTING.md">贡献指南</a>
  </p>
</div>

---

## 概述

**Emind-T1 v3.0** 是面向 **423B 级大模型**的生产级训练框架,默认配置对标 **DeepSeek-V3**(MoE 256 专家 + SWA + head_dim 解耦 + MTP),同时完整支持 **0.5B 到 423B** 全规模训练。覆盖 **数据蒸馏 → 模型训练 → 量化部署 → 推理服务** 全链路。

### V3.0 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **默认 423B MoE** | DSV3 风格:d_model=7168 / 61层 / 256专家×Top-8 + 1共享 / 128K 上下文 |
| 🎯 **SOTA 架构** | GQA+SWA 层间交替 · head_dim 解耦 · per-head QK-Norm · Dynamic NTK · MTP |
| 🔤 **128K 分词器** | HF tokenizers Rust + byte-level BPE,CJK/emoji 无损 roundtrip |
| 🔗 **HF 互操作** | `from_pretrained` 加载 Qwen2.5/Llama3,`save_pretrained` 输出标准格式 |
| 📊 **全规模预设** | 0.5B / 1.5B / 4B / 7B Dense + 4B MoE + 423B,一键切换 |
| 🔬 **zhengliu 蒸馏** | 多 Teacher(GPT-4/Claude/Gemini)+ KTO/ORPO + 行业垂类 |
| 🚀 **vLLM 集成** | Prefix Caching / Speculative Decoding / FP8 / 多 LoRA |

---

## ⚡ 快速开始

```bash
git clone https://github.com/fuzhen563-bot/Emind.git && cd Emind-T1

# 仅核心(训练 + 本地推理)
pip install -r requirements.txt

# 全量(推荐开发者)
pip install -e ".[all]"

# 蒸馏工具箱(独立包)
cd zhengliu && pip install -e ".[all]" && cd ..

# 验证
python -m pytest tests/test_v3_0_arch.py -v
```

---

## 🎓 小模型训练配置教程

> **重要**:V3.0 默认配置面向 **423B 大模型**。训练小模型(0.5B~7B)**务必使用预设**,不要用裸默认值。

### 选择模型规模

| 预设 | 参数量 | 显存(全参 BF16) | 显存(LoRA) | 适用场景 |
|------|--------|-----------------|-----------|---------|
| `emind_0_5b()` | 0.5B | 16GB | 8GB | 入门/原型 |
| `emind_1_5b()` | 1.5B | 40GB | 16GB | 边缘部署 |
| `emind_4b()` | 4B | 48GB | 24GB | **行业垂类主力** |
| `emind_7b()` | 7B | 80GB | 30GB | 通用能力 |
| `emind_4b_moe()` | 4B总/1B激活 | 24GB | 16GB | MoE 验证 |
| `dsv3_423b()` | 423B/37B激活 | 集群 | — | 大模型(需 100+ H100) |

### 方式 A:预设快速训练(推荐)

```bash
# 训练 4B Dense
python scripts/train_v3.py --preset 4b --mode sft \
    --data data/sft.jsonl --output checkpoints/emind-4b --bf16

# 加载 Qwen2.5-7B 做 SFT
python scripts/train_v3.py --from-pretrained Qwen/Qwen2.5-7B \
    --mode sft --data data/sft.jsonl --output checkpoints/emind-7b-sft --bf16

# LoRA 微调(单卡 24GB)
python scripts/train_v3.py --preset 7b --mode sft --lora --lora-rank 16 \
    --data data/sft.jsonl --bf16
```

### 方式 B:自定义配置

```python
from model import EmindConfig, create_model

cfg = EmindConfig.emind_4b()   # 从预设开始
cfg.d_model = 3072             # 调宽度
cfg.n_layers = 28              # 调深度
cfg.max_seq_len = 16384        # 调上下文
cfg.use_moe = True             # 切 MoE
cfg.n_experts = 16             # 专家数

model = create_model(cfg)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
```

### 方式 C:从零训练

```bash
# 1. 训练 128K 分词器(~1h,CPU)
python scripts/train_tokenizer_v3.py --corpus data/corpus/ --vocab-size 128000

# 2. 预训练
python scripts/train_v3.py --preset 7b --mode pretrain \
    --data data/pretrain.jsonl --output checkpoints/emind-7b-base --bf16

# 3. SFT
python scripts/train_v3.py --preset 7b --mode sft \
    --data data/sft.jsonl --output checkpoints/emind-7b-sft --bf16
```

### 方式 D:升级已有模型

```bash
# emind-1.5b_sft(V2.0)→ V3.0
python scripts/migrate_to_v3.py --src models/emind-1.5b_sft/ --dst checkpoints/emind-1.5b-v3/
bash scripts/upgrade_1.5b_v3.sh  # 继续预训练(4×H100,12-24h)
```

详见 [docs/V3_TRAINING_GUIDE.md](docs/V3_TRAINING_GUIDE.md) | [docs/V3_UPGRADE_1_5B_GUIDE.md](docs/V3_UPGRADE_1_5B_GUIDE.md)

---

## 📐 模型架构

### 默认配置(423B,对标 DeepSeek-V3)

| 组件 | 方案 |
|------|------|
| 位置编码 | RoPE + YaRN + Dynamic NTK(rope_theta=1M,128K) |
| 注意力 | GQA + SWA 层间交替(head_dim=128,per-head QK-Norm) |
| 前馈 | MoE fine-grained(256e×Top-8 + 1 shared,aux-loss-free) |
| MTP | Multi-Token Prediction(DSV3 风格) |
| forward | EmindOutput NamedTuple(向后兼容) |

### 全规模配置表

| 规模 | d_model | layers | MoE | seq | 预设 |
|------|---------|--------|-----|-----|------|
| 0.5B | 896 | 24 | Dense | 8K | `emind_0_5b()` |
| 1.5B | 1536 | 28 | Dense | 32K | `emind_1_5b()` |
| 4B | 2560 | 32 | Dense | 32K | `emind_4b()` |
| 7B | 4096 | 32 | Dense | 32K | `emind_7b()` |
| 4B MoE | 2048 | 16 | 32e×4+1 | 32K | `emind_4b_moe()` |
| **423B** | **7168** | **61** | **256e×8+1** | **128K** | `dsv3_423b()` |

---

## 🔬 zhengliu 蒸馏工具箱

```bash
cd zhengliu && pip install -e ".[all]"
python -m zhengliu  # 15 项菜单
```

多 Teacher(GPT-4/Claude/Gemini)· KTO/ORPO · 医疗/法律/金融 · 文档转数据 · RAG · V3.0 适配

---

## 🚀 推理部署

```bash
python cli.py serve --vllm --model checkpoints/emind-7b-sft --vllm-prefix-caching
python cli.py serve --port 3333  # Web 服务
```

---

## 📋 CLI 参考

| 命令 | 用途 |
|------|------|
| `train --preset 4b` | 训练(预设) |
| `infer` | 推理 |
| `serve` | 服务 |
| `eval` | 评测 |
| `rl --rl-mode grpo` | 强化学习 |

---

## 🧪 测试

```bash
python -m pytest tests/test_v3_0_arch.py -v   # V3 架构(26项)
python -m pytest tests/test_emind.py -v        # 核心
ruff check . --select F,E9                      # Lint
```

---

## 🗺️ 路线图

| 阶段 | 状态 |
|------|------|
| V3.0 架构(head_dim/SWA/MoE/MTP/128K) | ✅ 完成 |
| 4B/7B 训练验证 | 🔄 进行中 |
| 多模态(视觉/语音/视频) | 📋 计划 |
| Agent + MCP | 📋 计划 |
| 4D 分布式(TP/PP/EP) | 📋 计划 |

---

## 🤝 贡献

欢迎贡献!请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可

**个人研究免费·商业须授权**。详见 [LICENSE](LICENSE)。商业授权:business@yiziyun.com

---

<div align="center">
  <strong>© 2026 Emind-T1 AI</strong><br>
  <a href="https://yiziyun.com">厦门亦梓科技有限公司</a> ·
  <a href="https://yyzjai.cn">苏州云养智健人工智能科技有限公司</a><br>
  <small>亦智大模型研究院 — 思考 · 创造 · 超越</small>
</div>
