## Environment
```ruby
pip install -r ./requirements_vllm.txt
```

## Pipeline
### 1. Generate Reasoning Structure (Section `3.1`)

```bash
cd ./src/reasoning
CUDA_VISIBLE_DEVICES=0 python self_discover_vllm.py
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