import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


from utils import Iteratorize, Stream
from utils import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    cache_dir: str = None,
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "quantumaikr/KoreanLM-LoRA",
    prompt_template: str = "",  
    server_name: str = "0.0.0.0",
    share_gradio: bool = True,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='quantumaikr/KoreanLM'"

    prompter = Prompter(prompt_template)
    
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=False,
    )
    
    if device == "cuda":
        koreanlm = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        koreanlm = PeftModel.from_pretrained(
            koreanlm,
            lora_weights,
            torch_dtype=torch.float16,
        )

    elif device == "mps":
        koreanlm = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map={"": device},
            cache_dir=cache_dir,
        )
        koreanlm = PeftModel.from_pretrained(
            koreanlm,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16
        )
    else:
        koreanlm = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

        koreanlm = PeftModel.from_pretrained(
            koreanlm,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    koreanlm.push_to_hub('KoreanLM-LoRA')

    if not load_8bit:
        koreanlm.half()

    koreanlm.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        koreanlm = torch.compile(koreanlm)

    def evaluate(
        instruction,
        input=None,
        temperature=0.4,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=512,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    koreanlm.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = koreanlm.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about KoreanLM.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.4, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=1, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=512, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output", value=1),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="😎 KoreanLM: 한국어 언어모델 😎",
        description="""
<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/quantumaikr/KoreanLM/main/assets/icon.png" alt="KoreanLM icon" style="width: 200px; display: block; margin: auto; border-radius: 20%;">
KoreanLM은 한국어 언어모델 브랜드명 입니다.<br>
[(주)퀀텀아이](https://quantumai.kr)
</p><br>

""",
        thumbnail="https://raw.githubusercontent.com/quantumaikr/KoreanLM/main/assets/icon.png",
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    # Old testing code follows.


if __name__ == "__main__":
    fire.Fire(main)