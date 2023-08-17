# komt
Recently, due to the success of ChatGPT, numerous large language models have emerged in an attempt to catch up with ChatGPT's capabilities. 
However, when it comes to Korean language performance, it has been observed that many models still struggle to provide accurate answers or generate Korean text effectively. 
This study addresses these challenges by introducing a multi-task instruction technique that leverages supervised datasets from various tasks to create training data for Large Language Models (LLMs).
## News or Update
### 2023.08.16 
- We are releasing the [davidkim205/komt-Llama-2-7b-chat-hf-ggml](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf-ggml) model
### 2023.08.17
- We are releasing the [davidkim205/komt-Llama-2-13b-hf-lora](https://huggingface.co/davidkim205/komt-Llama-2-13b-hf-lora) and [davidkim205/komt-Llama-2-13b-hf-ggml]https://huggingface.co/davidkim205/komt-Llama-2-13b-hf-ggml) models

## Released Model Checkpoints
- komt-Llama-2-7b-chat-hf : [davidkim205/komt-Llama-2-7b-chat-hf](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf)
- komt-Llama-2-7b-chat-hf-lora : [davidkim205/komt-Llama-2-7b-chat-hf-lora](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf-lora)
- komt-Llama-2-7b-chat-hf-ggml : [davidkim205/komt-Llama-2-7b-chat-hf-ggml](https://huggingface.co/davidkim205/komt-Llama-2-7b-chat-hf-ggml)
- komt-Llama-2-13b-chat-hf : [davidkim205/komt-Llama-2-13b-hf](https://huggingface.co/davidkim205/komt-Llama-2-13b-hf)
- komt-Llama-2-13b-chat-hf-lora : [davidkim205/komt-Llama-2-13b-hf-lora](https://huggingface.co/davidkim205/komt-Llama-2-13b-hf-lora)
- komt-Llama-2-13b-chat-hf-ggml : [davidkim205/komt-Llama-2-13b-hf-ggml]https://huggingface.co/davidkim205/komt-Llama-2-13b-hf-ggml)
- komt-polyglot-ko-12.8b : The model is currently being trained.
- komt-polyglot-ko-12.8b-lora : The model is currently being trained.
- komt-polyglot-ko-5.8b : The model is currently being trained.

## Hardware and Software
- nvidia driver : 535.54.03
- CUDA Version: 12.2

## Setup

```
git clone git@github.com:davidkim205/komt.git
cd komt

conda create -n komt python=3.10
conda activate komt

pip install -r requirements.txt

```
### for lora
After completing the basic installation, you need to install the 'requirements_lora.txt' located in the 'lora' directory.
``` 
pip install -r lora/requirements_lora.txt
```
### for GGML
If you choose to download it directly, please refer to the instructions provided at [llama.cpp](https://github.com/ggerganov/llama.cpp#usage).

``` 
pip install -r llama.cpp/requirements.txt
```
## Inference
``` 
python infer.py davidkim205/komt-Llama-2-7b-chat-hf
```
### LoRA Inference
``` 
python lora/infer_loray.py davidkim205/komt-Llama-2-7b-chat-hf
```

### GGML Inference
```
cd llama.cpp 
make -j && ./main -m ./models/komt-Llama-2-7b-chat-hf-ggml/ggml-model-q4_0.bin -p "자동차 종합(정기)검사 의무기간은 얼 마인가요?"
```
When using the original [llama.cpp](https://github.com/ggerganov/llama.cpp) 
``` 
make -j && ./main -m ./models/komt-Llama-2-7b-chat-hf-ggml/ggml-model-q4_0.bin -p "### instruction: 누전차단기가 내려가는 이유는 무엇입 니까?\n\n### Response:"
```


## evaluate
``` 
python eval.py davidkim205/komt-Llama-2-7b-chat-hf-lora
```
### LoRA Inference
``` 
python lora/eval.py davidkim205/komt-Llama-2-7b-chat-hf-lora
```
## Fine-tuning
working!!!
### Fine-tuning using FSDP
working!!!
### Fine-tuning using LoRA
working!!!

## dataset
we collected various datasets based on the Korean language. A total of 1,642,421 datasets were created. These datasets include AI Hub, 모두의 말뭉치 (Korean language corpus), KISTI AI, and ShareGPT.

## Multi-Task Instruction


## Model Benchmark
- Used EleutherAI's lm-evaluation-harness https://github.com/EleutherAI/lm-evaluation-harness/tree/polyglot
 
### Korean Semantic Textual Similarity
| model name                    | Version | Metric |      Value |      | Stderr |
|-------------------------------| ------: | ------ |-----------:| ---- | -----: |
| meta-llama/Llama-2-7b-chat-hf |       0 | acc    |     0.4817 | ±    | 0.0220 |
| beomi/llama-2-ko-7b           |       0 | acc    |     0.5029 | ±    | 0.0220 |
| kfkas/Llama-2-ko-7b-Chat      |       0 | acc    |     0.5318 | ±    | 0.0219 |
| **komt-llama-2-7b (ours)**    |       0 | acc    | **0.5530** | ±    | 0.0218 |

# Memory Usage
- komt-Llama-2-7b-chat-hf : 23611MiB / 24576MiB 
- komt-Llama-2-7b-chat-hf-lora : 4665MiB / 24576MiB
- komt-Llama-2-13b-hf : 50658MiB / 81920MiB 
- komt-Llama-2-13b-hf-lora : 9101MiB / 24564MiB 
- komt-polyglot-ko-12.8b : 50188MiB / 81920MiB 
- komt-polyglot-ko-12.8b-lora : working!!!
- komt-polyglot-ko-5.8b : 23414MiB / 81920MiB 

### Examples
| 영화 해리포터 시리즈 배급사가 어디야                         |
| ------------------------------------------------------------ |
| **meta-llama/Llama-2-7b-chat-hf**                            |
| Warner Bros.                                                 |
| **kfkas/Llama-2-ko-7b-Chat**                                 |
| 해리포터 시리즈 배급사는 워너 브라더스(WB)입니다. 해리포터 시리즈는 2001년 영화로 처음 개봉되었습니다. 시리즈는 2018년 마지막 편인 해리포터와 마법사의 돌까지 총 8부작으로 제작되었습니다. |
| **beomi/llama-2-ko-7b**                                      |
| 1. Universal Pictures. Universal Pictures는 Universal Pictures가 영화 배급사로서 배급하고 있는 영화가 무엇인지 알려주는 웹사이트이다. Universal Pictures의 홈페이지에는 영화와 관련된 다양한 자료가 담겨 있다. 이 웹사이트는 크게 4개의 섹션으로 구성되어 있다. 그 중 첫 번째 섹션인 ‘Movies’에는 Universal Pictures가 배급하고 있는 영화가 소개되어 있다. Universal Pictures가 배급하고 있는 영화의 리스트를 보고 싶은 사용자는 화면 왼쪽에 있는 ‘Films’를 클릭한다. 그러면 Universal Pictures가 배급하고 있는 영화의 리스트가 화면에 나타난다. |
| **davidkim205/komt-Llama-2-7b-chat-hf**                      |
| 워너 브라더스\n해리포터 시리즈는 콘텐츠로는 크게 세 부분으로 나누고, 배급사인 워너 브라더스는 네 부분으로 나누어 판매하고 있다. 그 중 《해리포터와 혼혈 왕자》(Harry Potter and the Half-Blood Prince) 같은 일부는 한 번에 한 챕터씩 발매하기도 한다. 이처럼 분할발매는 2007년 《해리포터와 혼혈 왕자》(Harry Potter and the Half-Blood Prince) 이후로 시작되었다. 《해리포터 시리즈》(Harry Potter Series) 또는 《해리 포터 시리즈》(Harry Potter Series) 라고도 부른다. 시리즈에 대한 논란에도 불구하고, 해리 포터 시리즈는 전 세계적으로 유명한 책 시리즈이다. 《해리포터와 죽음의 성물》(Harry Potter and the Deathly Hallows – Part 1)은 2010년 11월 26일 개봉하였고, 2011년 1월 25일 《해리포터와 죽음의 성물》(Harry Potter and the Deathly Hallows – Part 2)이 개봉한다. |
| **davidkim205/komt-Llama-2-7b-chat-hf-lora**                 |
| 워너 브라더스\n데이비드 잉글리시는 1990년대 초 시리즈를 시작했다. 이 영화들은 총 8부작으로 만들어졌으며, 첫 번째 작품은 《해리 포터와 마법사의 돌》이다. 작가는 J. K. 롤링이며, 영화 시리즈는 배급사인 워너 브라더스가 만들었다. 촬영은 1997년 7월 11일부터 12월 22일까지 런던에서 이루어졌다. 촬영 장소로는 런던 크릭스 스튜디오와 런던 시티 오피스가 사용되었다. 데이비드 잉글리시는 해리 포터 역할을 맡았다. |
| **davidkim205/komt-Llama-2-13b-hf**                          |
| 워너 브라더스\n또한, 영화 해리포터와 해리포터 시리즈에서 빗자루를 타고 등장하는 스네이프 대신 이번 작품에서는 볼드모트가 빗자루를 타고 등장한다. 영화 해리포터와 해리포터 시리즈의 등장인물 목록 참고. 《해리포터와 죽음의 성물 1부》(Harry Potter and the Deathly Hallows – Part 1)는 2010년에 개봉한 영화로서 J. K. 롤링의 동명 소설(해리포터와 죽음의 성물)을 원작으로 하여 만든, 영국과 미국 합작, 판타지 스릴러 영화다. 소설 7부작 중 마지막 일곱 번째에 해당하는 작품이며, 영화 해리포터와 죽음의 성물 시리즈(해리포터 (영화 시리즈))의 첫 번째 작품이다. 영화 해리포터 시리즈(해리포터 (영화 시리즈))의 최종편으로, 영화 해리포터와 죽음의 성물 2부에서 데스 이터가 된 볼드모트가 등장하여 결전을 벌인다. 《해리포터와 죽음의 성물 1부》(Harry Potter and the Deathly Hallows – Part 1)는 2010년에 개봉한 영화로서 J. K. 롤링의 동명 소설(해리포터와 죽음의 성물)을 원작으로 하여 만든, 영국과 미국 합작, 판타지 스릴러 영화다. 소설 7부작 중 마지막 일곱 번째에 해당하는 작품이며, 영화 해리포터와 죽음의 성물 시리즈(해리포터 (영화 시리즈))의 첫 번째 작품이다 |

## References
- https://github.com/facebookresearch/llama)
- https://huggingface.co/EleutherAI/polyglot-ko-12.8b

-----------------
## Original LLaMA
### Llama 2
https://github.com/facebookresearch/llama
### Llama 1
https://github.com/facebookresearch/llama/tree/llama_v1

### llama.cpp
https://github.com/ggerganov/llama.cpp
