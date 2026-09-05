# 项目记忆：Itcast_qa_system

## ML 实验环境（重要）
- 本机原无 torch/transformers。managed python 3.13.12 的 venv：
  `C:/Users/lizhongxin/.workbuddy/binaries/python/envs/default`
  安装命令：`python -m venv` 后 `pip install torch --index-url https://download.pytorch.org/whl/cpu` + `transformers scikit-learn openai numpy`
- 运行任何涉及 torch 的脚本前必须加环境变量，否则 Windows 上 OpenMP 多线程会 Segmentation Fault：
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_INTRA_OP_PARALLELISM=1`
  并在代码顶部 `import torch; torch.set_num_threads(1)`。
- `accelerate` 在该 venv 下 import 即段错误 → 不要用 `transformers.Trainer`，改用原生 torch 训练循环。

## 项目约定
- 意图识别二分类标签：0=通用知识, 1=专业咨询（`QueryClassifier.predict_category` 用 `pred==1→专业咨询`）。
- 意图微调基座：`rag_qa/core/bert_query_classifier`（BertForSequenceClassification 2分类）。
  注意 `rag_qa/core/bert_results/checkpoint-87` 是**文档分段 NER 模型**（标签 B-EOP/O），不是意图模型，勿混淆。
- 微调后模型输出目录：`rag_qa/core/bert_results/intent-ckpt`。
- 大模型调用：config 里 `LLM_MODEL=deepseek-v4-flash`，API key 取环境变量 `ALIYUN_API_KEY`，base_url 为 dashscope compatible-mode。
- `import rag_qa.*` 会触发 `rag_qa/__init__.py` → 需要 `milvus_model`（未装）。需加载子模块时用 `importlib.util.spec_from_file_location` 文件级加载，绕过包 `__init__`。
- deepseek-v4-flash 对超短无意义输入（如 "hi"）会返回空串，但完整分类提示词下正常返回 JSON。
