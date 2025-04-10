# ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement

This paper is accepted by ICLR-2025 as Oral Presentation.

Paper: [Arxiv Link](https://arxiv.org/abs/2410.02108)

## Abstract
Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose Reasoning Generalist via Self-Improvement (ReGenesis), a method to self-synthesize reasoning paths as post-training data by progressing from abstract to concrete. More specifically, ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that ReGenesis achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4.6% after post training, ReGenesis delivers around 6.1% performance improvement. We also conduct in-depth analysis of our framework and show ReGenesis is effective across various LLMs and design choices.

## Environment
```ruby
pip install -r ./requirements_vllm.txt
```

## Pipeline
### 1. Generate Reasoning Structure (Section `3.1`)

```bash
cd ./src/reasoning
CUDA_VISIBLE_DEVICES=0 python reasoning_paths_gen.py
```

### 2. Process the reasoning (Filtering) to do Exact Match (Section `3.2.1`)

```bash
cd ./src/reasoning
CUDA_VISIBLE_DEVICES=1 python process_reason.py
```

### 3. Filter out samples which gets no reasoning structure in Step 1

```bash
cd ./src/reasoning
python filter_questions_cannot_answer.py
```

### 4. Add truth to the prompt and run self_discover again on samples filtered out by Step 3

```bash
cd ./src/reasoning
CUDA_VISIBLE_DEVICES=0 python truth_convert_reason.py
```
After this step, you will go back to step 2 for filtering again to get `step4-2.json`

### 5. Prepare for Fine-tune 

```bash
cd ./src/finetune_code
python convert_format.py
```

### 6. Fine-tune 

```bash
cd ./src/finetune_code
# need a new env
pip install -r requirements.txt
sh ft_mistral.sh
```

### 7. Eval Dataset

We use the codes from `https://github.com/OpenBMB/Eurus`

## Citation

```ruby
@article{peng2024regenesis,
  title={ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement},
  author={Peng, Xiangyu and Xia, Congying and Yang, Xinyi and Xiong, Caiming and Wu, Chien-Sheng and Xing, Chen},
  journal={arXiv preprint arXiv:2410.02108},
  year={2024}
}
```