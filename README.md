<p align="center" width="100%">
<img src="assets/icon.png" alt="KoreanLM icon" style="width: 500px; display: block; margin: auto; border-radius: 10%;">
</p>


# KoreanLM: 한국어 언어모델 프로젝트

KoreanLM은 한국어 언어모델을 개발하기 위한 오픈소스 프로젝트입니다. 현재 대부분의 언어모델들은 영어에 초점을 맞추고 있어, 한국어에 대한 학습이 상대적으로 부족하고 토큰화 과정에서 비효율적인 경우가 있습니다. 이러한 문제를 해결하고 한국어에 최적화된 언어모델을 제공하기 위해 KoreanLM 프로젝트를 시작하게 되었습니다.


## 프로젝트 목표

1. 한국어에 특화된 언어모델 개발: 한국어의 문법, 어휘, 문화적 특성을 반영하여 한국어를 더 정확하게 이해하고 생성할 수 있는 언어모델을 개발합니다.

2. 효율적인 토큰화 방식 도입: 한국어 텍스트의 토큰화 과정에서 효율적이고 정확한 분석이 가능한 새로운 토큰화 방식을 도입하여 언어모델의 성능을 향상시킵니다.

3. 거대 언어모델의 사용성 개선: 현재 거대한 사이즈의 언어모델들은 기업이 자사의 데이터를 파인튜닝하기 어려운 문제가 있습니다. 이를 해결하기 위해 한국어 언어모델의 크기를 조절하여 사용성을 개선하고, 자연어 처리 작업에 더 쉽게 적용할 수 있도록 합니다.


## 사용 방법

KoreanLM은 GitHub 저장소를 통해 배포됩니다. 프로젝트를 사용하려면 다음과 같은 방법으로 설치하실 수 있습니다.

```bash
git clone https://github.com/quantumaikr/KoreanLM.git
cd KoreanLM
pip install -r requirements.txt
```

## 예제

다음은 transformers 라이브러리를 통해 모델과 토크나이저를 로딩하는 예제입니다.

```python

import transformers
model = transformers.AutoModelForCausalLM.from_pretrained("quantumaikr/KoreanLM")
tokenizer = transformers.AutoTokenizer.from_pretrained("quantumaikr/KoreanLM")

```


## 훈련 (파인튜닝)

```bash
torchrun --nproc_per_node=4 --master_port=1004 train.py \
    --model_name_or_path quantumaikr/KoreanLM \
    --data_path korean_data.json \    
    --num_train_epochs 3 \
    --cache_dir './data' \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
```

```bash
pip install deepspeed
torchrun --nproc_per_node=4 --master_port=1004 train.py \
    --deepspeed "./deepspeed.json" \
    --model_name_or_path quantumaikr/KoreanLM \
    --data_path korean_data.json \    
    --num_train_epochs 3 \
    --cache_dir './data' \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
```

## 훈련 (LoRA)

```bash
python finetune-lora.py \
    --base_model 'quantumaikr/KoreanLM' \
    --data_path './korean_data.json' \
    --output_dir './KoreanLM-LoRA' \
    --cache_dir './data' 
```

## 추론 

```bash
python generate.py \
    --load_8bit  \
    --share_gradio \
    --base_model 'quantumaikr/KoreanLM'  \
    --lora_weights 'quantumaikr/KoreanLM-LoRA' \
    --cache_dir './data' 

```

## 사전학습 모델 공개 및 웹 데모

[학습모델](https://huggingface.co/quantumaikr/KoreanLM/tree/main)

<i>* 데모 링크는 추후 공계예정</i>




## 기여방법

1. 이슈 제기: KoreanLM 프로젝트와 관련된 문제점이나 개선사항을 이슈로 제기해주세요.

2. 코드 작성: 개선사항이나 새로운 기능을 추가하기 위해 코드를 작성하실 수 있습니다. 작성된 코드는 Pull Request를 통해 제출해주시기 바랍니다.

3. 문서 작성 및 번역: 프로젝트의 문서 작성이나 번역 작업에 참여하여 프로젝트의 질을 높여주세요.

4. 테스트 및 피드백: 프로젝트를 사용하면서 발견한 버그나 개선사항을 피드백해주시면 큰 도움이 됩니다.

## 라이선스

KoreanLM 프로젝트는 Apache 2.0 License 라이선스를 따릅니다. 프로젝트를 사용하실 때 라이선스에 따라 주의사항을 지켜주시기 바랍니다.


## 기술 문의

KoreanLM 프로젝트와 관련된 문의사항이 있으시면 이메일 또는 GitHub 이슈를 통해 문의해주시기 바랍니다. 이 프로젝트가 한국어 언어모델에 대한 연구와 개발에 도움이 되길 바라며, 많은 관심과 참여 부탁드립니다.


이메일: hi@quantumai.kr


---

This repository has implementations inspired by [open_llama](https://github.com/openlm-research/open_llama), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [alpaca-lora](https://github.com/tloen/alpaca-lora) projects.