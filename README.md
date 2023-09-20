# komt
![multi task instruction tuning.jpg](images%2Fmulti%20task%20instruction%20tuning.jpg)

Recently, due to the success of ChatGPT, numerous large language models have emerged in an attempt to catch up with ChatGPT's capabilities. 
However, when it comes to Korean language performance, it has been observed that many models still struggle to provide accurate answers or generate Korean text effectively. 
This study addresses these challenges by introducing a multi-task instruction technique that leverages supervised datasets from various tasks to create training data for Large Language Models (LLMs).

## News or Update
### 2023.09.20
- finetune_with_lora 학습시 4bit, 8bit 선택하여 학습할수 있도록 기능추가
### 2023.09.19
- komt-llama2 모델을 쉽게 사용할수 있도록 예제와 학습 방법, 데이터셋을 추가합니다.
### 2023.09.17 
- 개선된 multi-task dataset으로 학습한 komt-llama2-7b-v1 모델을 배포합니다.(가끔씩 end token 적용이 안되는 문제, 답변을 너무 길게 하는 문제등 수정) 
- [davidkim205/komt-llama2-7b-v1](https://huggingface.co/davidkim205/komt-llama2-7b-v1)
- [davidkim205/komt-llama2-7b-v1-lora](https://huggingface.co/davidkim205/komt-llama2-7b-v1-lora)
- [davidkim205/komt-llama2-7b-v1-ggml](https://huggingface.co/davidkim205/komt-llama2-7b-v1-ggml) 
### 2023.08.16 
- We are releasing the [davidkim205/komt-Llama-2-7b-chat-hf-ggml](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf-ggml) model
### 2023.08.17
- We are releasing the [davidkim205/komt-Llama-2-13b-hf-lora](https://huggingface.co/davidkim205/komt-Llama-2-13b-hf-lora) and [davidkim205/komt-Llama-2-13b-hf-ggml]https://huggingface.co/davidkim205/komt-Llama-2-13b-hf-ggml) models

## Released Model Checkpoints
- [davidkim205/komt-llama2-7b-v1](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf)
- [davidkim205/komt-llama2-7b-v1-lora](https://huggingface.co/davidkim205/komt-llama2-7b-v1-lora)
- [davidkim205/komt-Llama-2-7b-chat-hf-lora](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf-lora)

### old version
- komt-Llama-2-7b-chat-hf : [davidkim205/komt-Llama-2-7b-chat-hf](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf)
- komt-Llama-2-7b-chat-hf-lora : [davidkim205/komt-Llama-2-7b-chat-hf-lora](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf-lora)
- komt-Llama-2-7b-chat-hf-ggml : [davidkim205/komt-Llama-2-7b-chat-hf-ggml](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf-ggml)
- komt-Llama-2-13b-chat-hf : [davidkim205/komt-Llama-2-13b-hf](https://huggingface.co/davidkim205/komt-Llama-2-13b-hf)
- komt-Llama-2-13b-chat-hf-lora : [davidkim205/komt-Llama-2-13b-hf-lora](https://huggingface.co/davidkim205/komt-Llama-2-13b-hf-lora)
- komt-Llama-2-13b-chat-hf-ggml : [davidkim205/komt-Llama-2-13b-hf-ggml](https://huggingface.co/davidkim205/komt-Llama-2-13b-hf-ggml)

## Hardware and Software
- nvidia driver : 535.54.03
- CUDA Version: 12.2

## Setup

```
git clone https://github.com/davidkim205/komt.git
cd komt

conda create -n komt python=3.10
conda activate komt

pip install -r requirements.txt

```
## Usage
우리는 komt-llama2 모델을 사용할수 있는 다양한 방법을 제공합니다.

### text-generation-webui
![text-generation-webui.gif](images%2Ftext-generation-webui.gif)

``` 
# text-generation-webui 코드 받기
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui/

# conda 환경생성 
conda create -n text-generation-webui python=3.10
conda activate text-generation-webui

# pip install
pip install -r requirements.txt

# model download
pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download;print(hf_hub_download(repo_id='davidkim205/komt-llama2-7b-v1-ggml', filename='ggml-model-q4_0.gguf', local_dir='./models/'))"
 
# server 실행
python server.py
```
### llama2-webui
![llama2-webui.gif](images%2Fllama2-webui.gif)

https://github.com/liltom-eth/llama2-webui

llama2-webui를 git clone후 requirements를 install 합니다. 그런다음  용량이 크기때문에 git lfs을 이용하여 komt-llama2-7b를 다운로드 받습니다.

``` 
git clone https://github.com/liltom-eth/llama2-webui.git
cd llama2-webui
pip install -r requirements.txt
```
model을 다운로드후 app을 실행합니다.
```
sudo apt install git-lfs
git lfs clone https://huggingface.co/davidkim205/komt-llama2-7b-v1

python app.py --backend_type transformers --model_path ./komt-llama2-7b-v1/

```
### llama.cpp 
![llama.cpp-example.gif](images%2Fllama.cpp-example.gif)
```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt

pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download;print(hf_hub_download(repo_id='davidkim205/komt-llama2-7b-v1-ggml', filename='ggml-model-q4_0.gguf', local_dir='./models/'))"

make -j && ./main -m ./models/ggml-model-q4_0.gguf -p "인삼은 어떤 효과가 있는가요? ##output:"
```
### llama.cpp with google colab
google colab에서 llama.cpp를 사용하여 komt를 사용하는 방법 

https://colab.research.google.com/drive/1uLHXv-6NT7yj4FHECrZezfo5pVL-ht63?usp=sharing


### usage_komt_with_lora
python과 jupyter를 이용한 예제입니다.
- [usage_komt_with_lora.py](usage_komt_with_lora.py)
- [usage_komt_with_lora.ipynb](usage_komt_with_lora.ipynb)
``` 
$ python infer.py 
Downloading (…)/adapter_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 528/528 [00:00<00:00, 5.02MB/s]
Downloading (…)lve/main/config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 631/631 [00:00<00:00, 4.96MB/s]
Downloading pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 27.0G/27.0G [04:29<00:00, 100MB/s]
Downloading (…)neration_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:00<00:00, 1.36MB/s]
Downloading adapter_model.bin: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 80.1M/80.1M [00:00<00:00, 82.7MB/s]
Downloading (…)okenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 749/749 [00:00<00:00, 6.66MB/s]
Downloading tokenizer.model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 500k/500k [00:00<00:00, 111MB/s]
Downloading (…)in/added_tokens.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 21.0/21.0 [00:00<00:00, 131kB/s]
Downloading (…)cial_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 96.0/96.0 [00:00<00:00, 608kB/s]
/home/david/anaconda3/envs/komt/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:399: UserWarning: `num_beams` is set to 1. However, `early_stopping` is set to `True` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `early_stopping`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/home/david/anaconda3/envs/komt/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:399: UserWarning: `num_beams` is set to 1. However, `early_stopping` is set to `True` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `early_stopping`.
  warnings.warn(
<s> ### instruction: 고양이는 왜 물을 싫어하나요?

### Response: 고양이는 사람과 달리 물을 싫어합니다. 이는 물에 녹아 있는 헤어쳐발과 물의 냄새 때문입니다. 고양이는 헤어쳐발이 물에 녹아 있으면 물을 마시고 싶지 않아하며, 물의 냄새에도 민감합니다. 이러한 이유로 고양이는 물을 싫어하게 되었습니다. 

고양이는 사람과 달리 체온이 높아 체온을 유지하기 위해 많은 칼로리를 필요로 합니다. 따라서 고양이는 물을 마시지 않고 물을 싫어합니다. 고양이는 체온을 유지하기 위해 물을 섭취하지 않으며, 물을 마시고 싶지 않습니다. 

또한, 고양이는 물을 마시면 손이 차가워지는 등 물에 녹아 있는 헤어쳐발 때문에 물을 싫어합니다. 헤어쳐발은 물을 녹여 손을 
고양이는 사람과 달리 물을 싫어합니다. 이는 물에 녹아 있는 헤어쳐발과 물의 냄새 때문입니다. 고양이는 헤어쳐발이 물에 녹아 있으면 물을 마시고 싶지 않아하며, 물의 냄새에도 민감합니다. 이러한 이유로 고양이는 물을 싫어하게 되었습니다. 

고양이는 사람과 달리 체온이 높아 체온을 유지하기 위해 많은 칼로리를 필요로 합니다. 따라서 고양이는 물을 마시지 않고 물을 싫어합니다. 고양이는 체온을 유지하기 위해 물을 섭취하지 않으며, 물을 마시고 싶지 않습니다. 

```
## Fine-tune
komt-llama2 모델을 학습시키는 방법을 제공합니다. 

논문과 배포한 모델에 사용한 데이터셋중 라이센스가 없는 KorQuAD 1.0 데이터셋을 datasets에 추가했습니다.

논문에 대한 자세한 내용은 아래 Korean Multi-task Instruction Tuning 를 참고하세요.

### Fine-tune with lora
![finetune_with_lora.gif](images%2Ffinetune_with_lora.gif)
먼저 github에서 코드를 받은후 패키지를 설치합니다.(위 setup참조)

finetune_with_lora.py는 custom dataset을 이용하여 모델 학습을 위한 코드입니다.
기본적으로 아래와 같이 argument가 없을경우 default로 davidkim205/komt-llama2-7b-v1모델을 base로 [komt_squad.json](datasets%2Fkomt_squad.json)로 학습이 진행됩니다.
``` 

python finetune_with_lora.py

```
모델이나 dataset 이나 batchsize등은 아래와 같이 수정이 가능합니다.
```
python finetune_with_lora.py --model_name_or_path davidkim205/komt-llama2-7b-v1 --data_path datasets/komt_squad.json --num_train_epochs 1 --per_device_train_batch_size 1 --learning_rate 1e-5
```
보다 자세한 argument에 대한 자세한 설명은 `python finetune_with_lora.py  -h` 확인하세요.

#### finetune 8-bit models with Low Rank Adaption (LoRA)
finetune_with_lora.py는  기본적으로 4-bit로 양자화하여 학습을 합니다. 
8bit로 양자화할경우 아래와 같이 사용하면 됩니다.
```
python finetune_with_lora.py --bits 8
```

----
# Korean Multi-task Instruction Tuning

## Abstract
With the recent success of ChatGPT, numerous large language models have emerged in an attempt to catch up with ChatGPT's capabilities. However, it has become evident that these models still struggle to provide accurate responses in Korean or face challenges when generating Korean text. In this study, we introduce the multi-task instruction technique, which is based on supervised datasets from various tasks, to create training data for large language models, aiming to address these issues.

## Introduction

The recent Korean large language models, such as GPT-4-LLM, Dolly, and Vicuna, have predominantly relied on translated datasets. However, using translated datasets presents several challenges:

- Language and Cultural Differences
Languages and cultures have unique expressions, vocabularies, and grammatical structures. Using translated datasets can hinder the model's ability to understand and learn effectively due to these differences.
- Translation Errors and Semantic Distortions
Machine translations are not perfect and can introduce errors or distort the meaning of the original text. This can lead to the model learning incorrect information or failing to grasp the true meaning of the source data.
- Data Quality
The quality of translated data depends on the accuracy of the source data. If the source data is inaccurate or noisy, the translated data can suffer from the same issues.
- Word Embedding Consistency
Mapping words from different languages into a consistent embedding space can be challenging. This can result in the model failing to learn the correct relationships between words or failing to recognize semantic differences among translated words.
- Data Quantity and Diversity
Using translated foreign datasets may not provide sufficient quantity and diversity of data, depending on the language and topic domain. Obtaining the required data quantity and diversity can be challenging.
- Difficulty in Understanding Context
Translated data often fails to convey the original context accurately, making it difficult for the model to understand the real meaning and context of specific words or sentences.

- Specialized Terminology and Idiomatic Expressions
Specialized terminology and idiomatic expressions in specific fields may not be appropriately handled during translation, causing the model to perform poorly in certain subjects or domains.
- Data Bias
Translating data from various countries and cultures can introduce biases or cultural differences into the model, potentially increasing bias in the model's responses.
- Performance Degradation
When original data is translated, some information may be lost in the translation process, leading to a potential decrease in the model's performance compared to using the original data directly.

## 2. Multi-task Instruction
To address these challenges and improve dataset quality, we propose an Instruction Turning Framework (ITF) that leverages multi-task datasets and instruction tuning, inspired by Google's FLAN (Finetuned LANguage Models are zero-shot Learners) technique.

### 2.1. Multi-task Datasets
We have curated multi-task datasets based on various existing Korean datasets, specifically tailored to each task. We have avoided relying on translated datasets used in previous Korean large language models. Our dataset sources include:
- AIHub Dataset: 305,900 samples
- KISTI AI Dataset: 824,337 samples
- KorQuad Dataset: 66,181 samples
- Miscellaneous Datasets: 346,803 samples
- Total Dataset Size: 1,543,221 samples

### 2.2. Instruction Tuning
Our ITF incorporates the instruction tuning technique proposed by Google's FLAN, resulting in improved zero-shot performance.
We have publicly released the freely licensed KorQuad 1.0 dataset on GitHub. However, due to licensing policies, we cannot release the other datasets.

## 3. Evaluation
For objective model evaluation, we initially used EleutherAI's lm-evaluation-harness but obtained unsatisfactory results. Consequently, we conducted evaluations using ChatGPT, a widely used model, as described in [link](https://arxiv.org/pdf/2308.06502.pdf).


## 4. Conclusion
In this study, we have proposed a method to optimize the Llama2 model for the Korean language. Experimental results demonstrate that the use of multi-task instruction outperforms other Korean-supporting Llama2 models, showcasing its superior performance. Furthermore, multi-task instruction exhibits excellent performance.
In future research, we plan to leverage multi-task instruction to develop various service models and applications.

---

# References
### Llama 2
https://github.com/facebookresearch/llama
### Llama 1
https://github.com/facebookresearch/llama/tree/llama_v1

### llama.cpp
https://github.com/ggerganov/llama.cpp
