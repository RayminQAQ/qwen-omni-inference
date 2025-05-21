# Load model directly
# from transformers import AutoProcessor, AutoModel

import gc
from pathlib import Path
from datasets import Dataset, load_dataset
import time
from tqdm.auto import tqdm
import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoProcessor, AutoModelForTextToWaveform
from qwen_omni_utils import process_mm_info
from transformers import set_seed
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# For resize image
from PIL import Image

def clear_resources(name: str) -> None:
    # if hasattr(self, name):
    #     delattr(self, name)
    torch.cuda.empty_cache()
    gc.collect()

def calculate_metrics(
    all_choices: list,
    all_answers: list,
    all_response: list,
    all_index2ans: list = None,
    allow_random: bool = True,
) -> dict:
    """calculate_metrics"""
    if all_index2ans is None:
        all_index2ans = [None] * len(all_response)

    predictions = [
        parse_multi_choice_response(response, all_choices, index2ans, allow_random)
        for response, index2ans in zip(all_response, all_index2ans)
    ]

    accuracy = accuracy_score(all_answers, predictions)
    f1 = f1_score(all_answers, predictions, average="weighted", zero_division=1)
    precision = precision_score(all_answers, predictions, average="weighted", zero_division=1)
    recall = recall_score(all_answers, predictions, average="weighted", zero_division=1)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def parse_multi_choice_response(
    response: str,
    all_choices: list = ["A", "B", "C", "D"],
    index2ans: dict = None,
    allow_random: bool = True,
) -> str:
    """parse_multi_choice_response"""
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)

    if index2ans is not None and len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans and ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        if allow_random:
            pred_index = random.choice(all_choices)
        else:
            pred_index = ""

    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)

        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index

def get_template_response(image_path: str, audio_path: str, question: str) -> list:

    system_msg = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text."
            }
        ]
    }
    
    user_content = [] # List[Dict[str, Any]]
    if audio_path is not None:
        user_content.append({
            "type": "audio",
            "audio": audio_path
        })
    if image_path is not None:
        user_content.append({
            "type": "image",
            "image": image_path
        })
    user_content.append({
        "type": "text",
        "text": question
    })
    user_msg = {
        "role": "user",
        "content": user_content
    }
    return [system_msg, user_msg]

def load_data(dataset_name_or_path: str, 
              prompt_template_path: str = None):
    """ Init: multiple choices answer """
    all_choices = ["A", "B", "C", "D"]
    
    """ Init: prompt template"""
    if prompt_template_path is not None:
        try:
            with Path(prompt_template_path).open("r", encoding="utf-8") as file:
                prompt_template = file.read()
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load the prompt template: {e}") from e
    else:
        prompt_template = "{question}"

    
    """ preprocess function: use the prompt template to format the question """
    def _preprocess(example):
        example["question"] = (
            f"{example['instruction']}\n"
            + f"{example['question']}\n"
            + "".join(example[f"option{i + 1}"] for i in range(len(all_choices)))
        )
        example["answer"] = example["answer"].replace("A", "0").replace("B", "1").replace("C", "2").replace("D", "3") # anwer to index: 這邊是用 0,1,2,3來表示答案
        example["question"] = prompt_template.format(question=example["question"])
        return example
    
    dataset = load_dataset(
                    "json",
                    data_files=dataset_name_or_path,
                    split="train",
                )
    
    return_dataset = Dataset.from_list([
                _preprocess(example)
                for example in tqdm(dataset, desc="Processing dataset", unit="example")
            ])
    
    return return_dataset

def load_model(model_name_or_path: str):
    # Error checking
    if model_name_or_path not in ["Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen2.5-Omni-3B", "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4", "Qwen/Qwen2.5-Omni-7B-AWQ"]:
        raise ValueError(f"Invalid model name: {model_name_or_path}. Please use 'Qwen/Qwen2.5-Omni-7B' or 'Qwen/Qwen2.5-Omni-3B'.")

    # Load with optimizations
    print(f"Loading {model_name_or_path}")
    if model_name_or_path == "Qwen/Qwen2.5-Omni-7B-AWQ":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B-AWQ")
        model = AutoModelForTextToWaveform.from_pretrained("Qwen/Qwen2.5-Omni-7B-AWQ")
        return model, processor
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16, # Use BF16
        # load_in_4bit=True,

        device_map="auto",         # Auto-distribute across GPUs
        #attn_implementation="flash_attention_2" # Use Flash Attention 2
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)
    return model, processor

def main():
    """ Parameter """
    USE_AUDIO_IN_VIDEO_FLAG = False
    dataset_name_or_path = "TOCFL-MultiBench/TOCFL-MultiBench.json"
    prompt_template_path = "prompt/base.txt"
    model_name_or_path = "Qwen/Qwen2.5-Omni-3B" # str["Qwen/Qwen2.5-Omni-7B" | "Qwen/Qwen2.5-Omni-3B" | "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4" | "Qwen/Qwen2.5-Omni-7B-AWQ"]
    max_new_tokens = 1
    #tensor_type = "bf16" # "bf16", "auto"
    
    """ Load dataset """
    dataset = load_data(dataset_name_or_path, prompt_template_path)
    print(f"Dataset loaded: {dataset}")
    
    """ Load model """
    model, processor = load_model(model_name_or_path)
    
    """ Evaluation """
    results = []
    all_response = []
    all_answers = []
    OOM = []
    
    audio_reponse = []
    audio_answers = []
    image_response = []
    image_answers = []
    
    strat_time = time.time()
    for idx, data in enumerate(tqdm(dataset, desc="Evaluating")):
        """ Load Template: 在 dataset 加入一個欄位叫做 template_response """
        chat_template = get_template_response(image_path=data["image"], audio_path=data["audio"], question=data["question"])
        text_prompt_vc = processor.apply_chat_template(chat_template, add_generation_prompt=True, tokenize=False)
        audios_vc, image_vc, _ = process_mm_info(chat_template, use_audio_in_video=USE_AUDIO_IN_VIDEO_FLAG)
        
        # Resize image
        if image_vc is not None:
            image_vc = [
                Image.new("RGB", (224, 224))
                for _ in image_vc
            ]
        
        input_token = processor(
            text=text_prompt_vc, audio=audios_vc, images=image_vc,
            return_tensors="pt", padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO_FLAG
        )
        input_token = input_token.to(model.device).to(model.dtype)
        
        del chat_template
        del text_prompt_vc
        del audios_vc
        del image_vc
        
        """ Inference """
        
        # OutOfMemory = False
        # try:
        input_len = input_token["input_ids"].shape[-1]
        with torch.no_grad():
            text_ids = model.generate(
                **input_token,
                return_audio=False,
                use_audio_in_video=USE_AUDIO_IN_VIDEO_FLAG,
                max_new_tokens=max_new_tokens, # Allow longer solutions
                # thinker_max_new_tokens = 1, # DEBUG
                # talker_max_new_tokens = 1, # DEBUG
                #num_beams=1,
            )
            
            generated_ids = text_ids[:, input_len:]
            text_response = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
                
                # print(text_response)
                
        # except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        #     # 只捕捉 CUDA OOM 的錯誤
        #     if "out of memory" in str(e).lower():
        #         print(f"[WARN] OOM at idx={idx}, id={data['id']}, setting response to empty.")
        #         clear_resources("")  # 清理快取
        #         text_response = ""
        #         OOM.append(data["id"])
        #         OutOfMemory = True
        #     else:
        #         # 如果不是 OOM，還是往上丟錯誤
        #         raise
        
        """ Save result"""
        all_response.append(text_response)
        all_answers.append(data["answer"])
        results.extend([
            {
                "id": data["id"],
                "question": data["question"],
                "generation": text_response,
                "answer": data["answer"],
                # "OOM": OutOfMemory,
            }
        ])
        
        """ Save audio and image response """
        if data["audio"] is not None:
            audio_reponse.append(text_response)
            audio_answers.append(data["answer"])
        if data["image"] is not None:
            image_response.append(text_response)
            image_answers.append(data["answer"])
            
    total_time = time.time() - strat_time
    
    """ Calculate metrics """
    idx2letter = {str(i): c for i, c in enumerate(["A","B","C","D"])}
    letter_answers = [idx2letter[a] for a in all_answers ]
    
    all_metrics = calculate_metrics(
        all_choices=["A", "B", "C", "D"],
        all_answers=letter_answers,
        all_response=all_response,
    )
    
    idx2letter = {str(i): c for i, c in enumerate(["A","B","C","D"])}
    letter_answers = [idx2letter[a] for a in image_answers ]
    image_metrics = calculate_metrics(
        all_choices=["A", "B", "C", "D"],
        all_answers=letter_answers,
        all_response=image_response,
    )
    
    idx2letter = {str(i): c for i, c in enumerate(["A","B","C","D"])}
    letter_answers = [idx2letter[a] for a in audio_answers ]
    audio_metrics = calculate_metrics(
        all_choices=["A", "B", "C", "D"],
        all_answers=letter_answers,
        all_response=audio_reponse,
    )
    
    """ Print results """
    print(f"Overll Metrics: {all_metrics}")
    print(f"Audio metrics: {audio_metrics}")
    print(f"Image metrics: {image_metrics}")
    print(f"Total time: {total_time} seconds")
    
    """ Save results """
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 1. 儲存詳細的 generation 結果為 JSON
    import json
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 2. 儲存整體／影像／語音的指標為 JSON
    metrics_summary = {
        "overall": all_metrics,
        "audio": audio_metrics,
        "image": image_metrics,
        "total_time": total_time,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=4)

    # 3. （可選）也把每條 sample 的結果存成 CSV，方便快速瀏覽
    import csv
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "generation", "answer"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Saved {len(results)} samples to {output_dir}/results.json and results.csv")
    print(f"✅ Saved metrics summary to {output_dir}/metrics.json")
    # print(f"✅ OOM samples: {OOM}")
    # print(f"✅ OOM size: {len(OOM)}")
    
if __name__ == "__main__":
    set_seed(11207330)
    clear_resources("")
    main()
