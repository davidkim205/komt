# komt : Korean Multi-task Instruction Tuning
![multi task instruction tuning.jpg](images%2Fmulti%20task%20instruction%20tuning.jpg)

Recently, due to the success of ChatGPT, numerous large language models have emerged in an attempt to catch up with ChatGPT's capabilities. 
However, when it comes to Korean language performance, it has been observed that many models still struggle to provide accurate answers or generate Korean text effectively. 
This study addresses these challenges by introducing a multi-task instruction technique that leverages supervised datasets from various tasks to create training data for Large Language Models (LLMs).

## News or Update
### 2023.10.24
- komt-mistral-7b-v1 모델 추가
> - [davidkim205/komt-mistral-7b-v1](https://huggingface.co/davidkim205/komt-mistral-7b-v1)
> - [davidkim205/komt-mistral-7b-v1-lora](https://huggingface.co/davidkim205/komt-mistral-7b-v1-lora)
> - [davidkim205/komt-mistral-7b-v1-gguf] (https://huggingface.co/davidkim205/komt-mistral-7b-v1-gguf)

### 2023.10.20
- komt-llama-30b-v1 모델 추가
> - [davidkim205/komt-llama-30b-v1](https://huggingface.co/davidkim205/komt-llama-30b-v1)
> - [davidkim205/komt-llama-30b-v1-lora](https://huggingface.co/davidkim205/komt-llama-30b-v1-lora)


### 2023.09.27
- chatgpt 기반 평가 결과에 아래 모델 추가
> - naver Cue
> - clova X
> - nlpai-lab/kullm-polyglot-12.8b-v2
> - kfkas/Llama-2-ko-7b-Chat
> - beomi/KoAlpaca-Polyglot-12.8B

### 2023.09.25
- komt-llama2-13b-v1 모델 추가
> - [davidkim205/komt-llama2-13b-v1](https://huggingface.co/davidkim205/komt-llama2-13b-v1)
> - [davidkim205/komt-llama2-13b-v1-lora](https://huggingface.co/davidkim205/komt-llama2-13b-v1-lora)
> - [davidkim205/komt-llama2-13b-v1-ggml](https://huggingface.co/davidkim205/komt-llama2-13b-v1-ggml)
### 2023.09.24
- Fine-tune with deepspeed 학습 방법 추가
### 2023.09.23
- usage komt with vllm 코드와 설치 방법 추가
### 2023.09.22
- 모델 평가 결과표 추가
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
### komt-llama2-7b
- [davidkim205/komt-llama2-7b-v1](https://huggingface.co/davidkim205/komt-llama2-7b-v1)
- [davidkim205/komt-llama2-7b-v1-lora](https://huggingface.co/davidkim205/komt-llama2-7b-v1-lora)
- [davidkim205/komt-llama2-7b-v1-ggml](https://huggingface.co/davidkim205/komt-llama2-7b-v1-ggml)
### komt-llama2-13b
- [davidkim205/komt-llama2-13b-v1](https://huggingface.co/davidkim205/komt-llama2-13b-v1)
- [davidkim205/komt-llama2-13b-v1-lora](https://huggingface.co/davidkim205/komt-llama2-13b-v1-lora)
- [davidkim205/komt-llama2-13b-v1-ggml](https://huggingface.co/davidkim205/komt-llama2-13b-v1-ggml)
### komt-llama-30b
- [davidkim205/komt-llama-30b-v1](https://huggingface.co/davidkim205/komt-llama-30b-v1)
- [davidkim205/komt-llama-30b-v1-lora](https://huggingface.co/davidkim205/komt-llama-30b-v1-lora)
### komt-mistral-7b
- [davidkim205/komt-mistral-7b-v1](https://huggingface.co/davidkim205/komt-mistral-7b-v1)
- [davidkim205/komt-mistral-7b-v1-lora](https://huggingface.co/davidkim205/komt-mistral-7b-v1-lora)
- [davidkim205/komt-mistral-7b-v1-gguf](https://huggingface.co/davidkim205/komt-mistral-7b-v1-gguf)
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

## transformers
``` 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer, GenerationConfig

model_name='davidkim205/komt-llama2-7b-v1'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

def gen(x):
    generation_config = GenerationConfig(
        temperature=0.8,
        top_p=0.8,
        top_k=100,
        max_new_tokens=512,
        early_stopping=True,
        do_sample=True,
    )
    q = f"### instruction: {x}\n\n### Response: "
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])

    start_tag = f"\n\n### Response: "
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str

print(gen('제주도를 1박2일로 혼자 여행하려고 하는데 여행 코스를 만들어줘'))
```
결과
``` 
### Response: 제주도를 1박2일로 혼자 여행하려면 다음과 같은 여행 코스를 만들어 계획할 수 있습니다:

1일차:
- 아침: 제주도의 아름다운 해변을 구경하기 위해 해변에 도착하세요. 일출을 감상하며 자연의 아름다움을 만끽하세요.
- 오후: 제주도의 대표적인 관광지인 한라산을 탐험하세요. 등산로를 따라 올라가면서 경치를 즐기고 설명을 듣으며 쉬운 산책을 즐기세요.
- 저녁: 제주도의 맛있는 음식점에서 저녁을 보내세요. 신선한 해산물과 향신료로 만든 음식을 맛보는 것은 제주도 여행의 완벽한 경험이 될 것입니다.

2일차:
- 아침: 한라산 일대를 탐험하기 위해 한라산 케이프로 이동하세요. 이 케이프는 등산을 즐기는 사람들에게 최적의 선택입니다. 

```
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
### usage komt with vllm
![vllm.gif](images%2Fvllm.gif)
vllm 라이브러리를 사용하기 위해서는 아래와 같이 conda 환경을 생성한후에 requirements_vllm.txt으로 패키지들을 설치해야합니다.
``` 
conda create -n vllm python=3.10
conda activate vllm
pip install -r requirements_vllm.txt
```
예제 코드는 아래와 같이 실행한후에 질문을 입력하면 됩니다.
``` 
$ python usage_komt_with_vllm.py 
INFO 09-25 18:48:20 llm_engine.py:72] Initializing an LLM engine with config: model='davidkim205/komt-llama2-7b-v1', tokenizer='davidkim205/komt-llama2-7b-v1', tokenizer_mode=auto, trust_remote_code=False, dtype=torch.float16, download_dir=None, load_format=auto, tensor_parallel_size=1, seed=0)
INFO 09-25 18:48:20 tokenizer.py:30] For some LLaMA-based models, initializing the fast tokenizer may take a long time. To eliminate the initialization time, consider using 'hf-internal-testing/llama-tokenizer' instead of the original tokenizer.
INFO 09-25 18:48:36 llm_engine.py:199] # GPU blocks: 1048, # CPU blocks: 512
>제주도 데이트 코스 알려줘
Processed prompts: 100%|██████████████████████████████████████████| 1/1 [00:15<00:00, 15.30s/it]
Prompt: '### instruction: 제주도 데이트 코스 알려줘\n\n### Response: ', Generated text: '제주도 데이트 코스 알려드리겠습니다.\n1. 아침에 일찍 일어나서 제주상공원에서 아침 해돋이를 보쩰 인사를 드립니다.\n2. 상공원을 돌아다니며 자연의 아름다움을 만끽합니다. 특히, 용두보 폭포를 건너 다니며 멋진 경치를 감상합니다.\n3. 오후 1시쯤 제주시의 유명한 향기를 맡을 수 있는 성산일출봉 근처 퍼즐을 풀어보세요. 여기에서는 노래방, 샤프심 강연, 워커힐 컨서트, 한라산성 발견 여숙 등 흥미로운 체험을 할 수 있습니다.\n4. 제주특유의 다양한 해산물 (해초, 김치, 해석 등)을 구경하고 싶다면, 자주짓네미나 제주시의 전통시장을 방문해보세요. 해산물 사찰 근처에 위치한 특수시장에서는 제주감귤을 맛볼 수 있습니다.\n5. 마지막으로 저녁에는 성산일출봉에서 한라산의 일출을 볼 수 있습니다. 일출을 감상하며 그 아름다움에 대한 감사를 표현합니다.\n\n이제 제주특별의 매력을 즐기실 준비가 되셨나요? 헛된 일상에서 벗어나 여유로움을 느낄 수 있는 제주도 데이트 코스를 즐기보세요.'

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
### Fine-tune with deepspeed
finetune_with_ds.py은 DeepSpeed기반으로 ZeRO-3 Offload을 사용하여 학습을 합니다. 
CPU Offloading을 통하여 GPU 메모리 사용량을 줄지만 CPU 메모리를 사용하기때문에 hw 사양에 맞게 조정을 해야합니다.
deepspeed 파일은 configs/deepseed_config.json에 추가하였습니다.

deepspeed를 이용할경우 아래와 같이 conda 환경을 추가한다음 해당 패키지를 설치해야 합니다.
``` 
conda create -n ds python=3.10
conda activate ds
pip install -r requirements_ds.txt
```

finetune_with_deepspeed 사용방법은 아래와 같습니다.
``` 
deepspeed finetune_with_ds.py
```
argument 수정시 아래를 참고하세요.
``` 
deepspeed finetune_with_ds.py --model_name_or_path davidkim205/komt-llama2-7b-v1 --data_path datasets/komt_squad.json --num_train_epochs 1 --per_device_train_batch_size 1 --learning_rate 1e-5 --deepspeed configs/deepspeed_config.json
```

## 평가결과
chatgpt를 이용하여 질문과 대답에대한 평가를 아래와 같이 진행하였습니다. 모델 평가를 위한 질문과 답변 chatgpt의 평가 결과는 eval_results를 참고하세요.


| model                                   | score   | average(0~5) | percentage |
| --------------------------------------- |---------| ------------ | ---------- |
| gpt-3.5-turbo(close)                    | 147     | 3.97         | 79.45%     |
| naver Cue(close)                        | 140     | 3.78         | 75.67%     |
| clova X(close)                          | 136     | 3.67         | 73.51%     |
| WizardLM-13B-V1.2(open)                 | 96      | 2.59         | 51.89%     |
| Llama-2-7b-chat-hf(open)                | 67      | 1.81         | 36.21%     |
| Llama-2-13b-chat-hf(open)               | 73      | 1.91         | 38.37%     |
| nlpai-lab/kullm-polyglot-12.8b-v2(open) | 70      | 1.89         | 37.83%     |
| kfkas/Llama-2-ko-7b-Chat(open)          | 96      | 2.59         | 51.89%     |
| beomi/KoAlpaca-Polyglot-12.8B(open)     | 100     | 2.70         | 54.05%     |
| **komt-llama2-7b-v1 (open)(ours)**      | **117** | **3.16**     | **63.24%** |
| **komt-llama2-13b-v1  (open)(ours)**    | **129** | **3.48**     | **69.72%** |
| **komt-llama-30b-v1  (open)(ours)**    | **129** | **3.16**     | **63.24%** |
| **komt-mistral-7b-v1  (open)(ours)**    | **131** | **3.54**     | **70.81%** |

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
For objective model evaluation, we initially used EleutherAI's lm-evaluation-harness but obtained unsatisfactory results. Consequently, we conducted evaluations using ChatGPT, a widely used model, as described in [Self-Alignment with Instruction Backtranslation](https://arxiv.org/pdf/2308.06502.pdf) and [Three Ways of Using Large Language Models to Evaluate Chat](https://arxiv.org/pdf/2308.06259.pdf) .


| model                                   | score   | average(0~5) | percentage |
| --------------------------------------- |---------| ------------ | ---------- |
| gpt-3.5-turbo(close)                    | 147     | 3.97         | 79.45%     |
| naver Cue(close)                        | 140     | 3.78         | 75.67%     |
| clova X(close)                          | 136     | 3.67         | 73.51%     |
| WizardLM-13B-V1.2(open)                 | 96      | 2.59         | 51.89%     |
| Llama-2-7b-chat-hf(open)                | 67      | 1.81         | 36.21%     |
| Llama-2-13b-chat-hf(open)               | 73      | 1.91         | 38.37%     |
| nlpai-lab/kullm-polyglot-12.8b-v2(open) | 70      | 1.89         | 37.83%     |
| kfkas/Llama-2-ko-7b-Chat(open)          | 96      | 2.59         | 51.89%     |
| beomi/KoAlpaca-Polyglot-12.8B(open)     | 100     | 2.70         | 54.05%     |
| **komt-llama2-7b-v1 (open)(ours)**      | **117** | **3.16**     | **63.24%** |
| **komt-llama2-13b-v1  (open)(ours)**    | **129** | **3.48**     | **69.72%** |
| **komt-llama-30b-v1  (open)(ours)**    | **129** | **3.16**     | **63.24%** |
| **komt-mistral-7b-v1  (open)(ours)**    | **131** | **3.54**     | **70.81%** |


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
