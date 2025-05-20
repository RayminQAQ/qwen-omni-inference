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
from qwen_omni_utils import process_mm_info
from transformers import set_seed
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    if model_name_or_path not in ["Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen2.5-Omni-3B"]:
        raise ValueError(f"Invalid model name: {model_name_or_path}. Please use 'Qwen/Qwen2.5-Omni-7B' or 'Qwen/Qwen2.5-Omni-3B'.")

    # Load with optimizations
    print(f"Loading {model_name_or_path}")
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16, # Use BF16
        device_map="auto",         # Auto-distribute across GPUs
        #attn_implementation="flash_attention_2" # Use Flash Attention 2
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)
    return model, processor

def evaluate(model, processor, dataset):
    ###############
    start_time = time.time()
    """ write your evaluation code here"""
    end_time = time.time()
    ###############
    pass

def main():
    """ Parameter """
    USE_AUDIO_IN_VIDEO_FLAG = False
    dataset_name_or_path = "TOCFL-MultiBench/TOCFL-MultiBench.json"
    prompt_template_path = "prompt/base.txt"
    model_name_or_path = "Qwen/Qwen2.5-Omni-3B" # str["Qwen/Qwen2.5-Omni-7B" | "Qwen/Qwen2.5-Omni-3B"]
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
    
    for data in tqdm(dataset, desc="Evaluating"): 
        """ Load Template: 在 dataset 加入一個欄位叫做 template_response """
        chat_template = get_template_response(image_path=data["image"], audio_path=data["audio"], question=data["question"])
        text_prompt_vc = processor.apply_chat_template(chat_template, add_generation_prompt=True, tokenize=False)
        audios_vc, image_vc, _ = process_mm_info(chat_template, use_audio_in_video=USE_AUDIO_IN_VIDEO_FLAG)
        input_token = processor(
            text=text_prompt_vc, audio=audios_vc, images=image_vc,
            return_tensors="pt", padding=True,
            #use_audio_in_video=USE_AUDIO_IN_VIDEO_FLAG
        )
        input_token = input_token.to(model.device).to(model.dtype)
        
        """ Inference """
        with torch.no_grad():
            text_ids = model.generate(
                **input_token,
                use_audio_in_video=USE_AUDIO_IN_VIDEO_FLAG,
                return_audio=False,
                max_new_tokens=max_new_tokens # Allow longer solutions
            )
        text_response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(text_response)
        
        """ Save result"""
        all_response.extend(text_response)
        all_answers.extend(data["answer"])
        results.extend([
            {
                "id": data["id"],
                "question": data["question"],
                "generation": text_response,
                "answer": data["answer"],
            }
        ])
        
    """ Calculate metrics """
    metrics = calculate_metrics(
        all_choices=["A", "B", "C", "D"],
        all_answers=all_answers,
        all_response=all_response,
    )
    
    print(f"Metrics: {metrics}")
    print(f"Results: {results}")
    
    """ Save results """
    
if __name__ == "__main__":
    set_seed(11207330)
    clear_resources("")
    main()