import argparse
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import os
from datasets import load_dataset
import json
import traceback
from datetime import datetime
import time
from pathlib import Path
import gc
import random
import numpy as np
import librosa
import whisper
import re
from decord import VideoReader, cpu
from ola.conversation import conv_templates, SeparatorStyle
from ola.model.builder import load_pretrained_model
from ola.utils import disable_torch_init
from ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_question_image_token
from ola.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN

# 輔助函數：設定隨機種子
def set_seed(seed_value=42):
    """設定所有相關的隨機種子以確保可重現性。"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"隨機種子已設定為: {seed_value}")

# 輔助函數：清理 CUDA 快取並觸發垃圾回收
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 參考自 TOCFL-MultiBench/src/system.py 中的 _get_model_config
def get_model_config(tensor_type: str) -> dict:
    """根據張量類型獲取模型配置"""
    if tensor_type == "fp16":
        return {"torch_dtype": torch.float16}
    if tensor_type == "bf16":
        return {"torch_dtype": torch.bfloat16}
    if tensor_type == "int8":
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "torch_dtype": torch.float16,
        }
    if tensor_type in ["fp4", "nf4"]:
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4" if tensor_type == "nf4" else "fp4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            "torch_dtype": torch.bfloat16,
        }
    return {"torch_dtype": "auto"}

def load_ola_model(model_name_or_path: str, tensor_type: str = "auto"):
    """載入 Ola 模型和相關處理器"""
    print(f"正在載入 Ola 模型從 {model_name_or_path}...")
    
    # 設置環境變數
    os.environ['LOWRES_RESIZE'] = '384x32'
    os.environ['HIGHRES_BASE'] = '0x32'
    os.environ['VIDEO_RESIZE'] = "0x64"
    os.environ['VIDEO_MAXRES'] = "480"
    os.environ['VIDEO_MINRES'] = "288"
    os.environ['MAXRES'] = '1536'
    os.environ['MINRES'] = '0'
    os.environ['REGIONAL_POOL'] = '2x'
    os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
    os.environ['LOAD_VISION_EARLY'] = '1'
    os.environ['SKIP_LOAD_VIT'] = '1'
    
    # 設定模型載入配置
    model_kwargs = get_model_config(tensor_type)
    print(f"使用配置: {model_kwargs}")
    
    # 使用 Ola 的 load_pretrained_model 函數載入模型
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(model_name_or_path, None)
    
    model = model.to('cuda')
    model = model.eval()
    
    # 根據設定調整模型精度
    if tensor_type == "bf16" or (isinstance(model_kwargs.get("torch_dtype"), torch.dtype) and model_kwargs["torch_dtype"] == torch.bfloat16):
        model = model.bfloat16()
    elif tensor_type == "fp16" or (isinstance(model_kwargs.get("torch_dtype"), torch.dtype) and model_kwargs["torch_dtype"] == torch.float16):
        model = model.half()
    
    return tokenizer, model, image_processor

def load_audio(audio_file_name):
    """載入並處理音訊檔案"""
    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    speechs = []
    speech_wavs = []

    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        mels.append(chunk)

    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]

    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs

def extract_audio_from_video(video_file_path):
    """從影片中提取音訊"""
    import moviepy.editor as mp
    my_clip = mp.VideoFileClip(video_file_path)
    return my_clip.audio

def run_inference(tokenizer, model, image_processor, text_prompt: str, image_paths: list = None, audio_paths: list = None, video_paths: list = None):
    """使用 Ola 模型進行推論"""
    print("準備推論輸入...")
    
    # 準備相關變數
    USE_SPEECH = False
    multimodal = {"files": [], "text": text_prompt}
    audio_path = None
    
    # 處理輸入路徑
    if image_paths and len(image_paths) > 0:
        multimodal["files"].append(image_paths[0])  # 目前只處理第一個圖片
    elif video_paths and len(video_paths) > 0:
        multimodal["files"].append(video_paths[0])  # 目前只處理第一個影片
    
    # 處理音訊路徑
    if audio_paths and len(audio_paths) > 0:
        audio_path = audio_paths[0]  # 目前只處理第一個音訊
    
    if not multimodal["files"]:
        print("警告: 沒有提供任何多媒體檔案，只處理文字提示。")
        multimodal["files"].append("dummy_placeholder")  # 確保有檔案列表
    
    try:
        # 確定模態類型
        visual_file = multimodal["files"][0]
        if isinstance(visual_file, str) and visual_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            modality = "video"
        elif isinstance(visual_file, str) and visual_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            modality = "image"
        else:
            modality = "text"  # 默認為純文字
        
        # 決定是否使用語音
        if audio_path:
            USE_SPEECH = True
        elif modality == "video":
            USE_SPEECH = True
        else:
            USE_SPEECH = False
        
        # 準備語音處理
        speechs = []
        speech_lengths = []
        speech_wavs = []
        speech_chunks = []
        
        # 處理視頻
        if modality == "video":
            vr = VideoReader(visual_file, ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 64, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            video = [Image.fromarray(frame) for frame in spare_frames]
            image_sizes = None
        else:
            # 處理圖片
            if modality == "image":
                image = [Image.open(visual_file)]
                image_sizes = [image[0].size]
            else:
                image = None
                image_sizes = None
        
        # 處理音訊
        if USE_SPEECH and audio_path:
            speech, speech_length, speech_chunk, speech_wav = load_audio(audio_path)
            speechs.append(speech.bfloat16().to('cuda'))
            speech_lengths.append(speech_length.to('cuda'))
            speech_chunks.append(speech_chunk.to('cuda'))
            speech_wavs.append(speech_wav.to('cuda'))
            print("載入外部音訊")
        elif USE_SPEECH and modality == "video":
            # 從視頻中提取音訊
            audio = extract_audio_from_video(visual_file)
            audio.write_audiofile("./video_audio.wav")
            video_audio_path = './video_audio.wav'
            speech, speech_length, speech_chunk, speech_wav = load_audio(video_audio_path)
            speechs.append(speech.bfloat16().to('cuda'))
            speech_lengths.append(speech_length.to('cuda'))
            speech_chunks.append(speech_chunk.to('cuda'))
            speech_wavs.append(speech_wav.to('cuda'))
            print("從視頻中提取音訊")
        else:
            # 處理無音訊情況
            speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
            speech_lengths = [torch.LongTensor([3000]).to('cuda')]
            speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
            speech_chunks = [torch.LongTensor([1]).to('cuda')]
        
        # 準備對話模板
        conv_mode = "qwen_1_5"  # Ola 使用 qwen 對話模板
        qs = text_prompt
        
        # 根據條件添加不同的標記
        if USE_SPEECH and audio_path:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + "User's question in speech: " + DEFAULT_SPEECH_TOKEN + '\n'
        elif USE_SPEECH:
            qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        # 建立對話
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 準備輸入
        if USE_SPEECH and audio_path:
            input_ids = tokenizer_speech_question_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
        elif USE_SPEECH:
            input_ids = tokenizer_speech_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
        else:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
        
        # 處理視頻或圖像
        if modality == "video":
            video_processed = []
            for idx, frame in enumerate(video):
                image_processor.do_resize = False
                image_processor.do_center_crop = False
                frame = process_anyres_video(frame, image_processor)
                
                if frame_idx is not None and idx in frame_idx:
                    video_processed.append(frame.unsqueeze(0))
                elif frame_idx is None:
                    video_processed.append(frame.unsqueeze(0))
            
            if frame_idx is None:
                frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
            
            video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
            video_processed = (video_processed, video_processed)
            
            video_data = (video_processed, (384, 384), "video")
                    
        elif modality == "image":
            # 處理單張圖像的情況
            if len(image) == 1:
                # 調整大小確保能被16整除
                w, h = image[0].size
                new_w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
                new_h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
                resized_img = image[0].resize((new_w, new_h))
                
                # 處理圖像
                batch = image_processor(resized_img, return_tensors="pt")
                image_tensor = batch.pixel_values.bfloat16().to("cuda")
                image_highres_tensor = image_tensor.clone()
                
                # 不使用 stack，因為只有一張圖像
                image_sizes = [resized_img.size]
            else:
                # 對於多張圖像的情況
                image_tensor_list = []
                image_highres_tensor_list = []
                image_sizes = []
                
                for img in image:
                    # 調整大小
                    w, h = img.size
                    new_w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
                    new_h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
                    resized_img = img.resize((new_w, new_h))
                    
                    # 處理圖像並收集張量
                    batch = image_processor(resized_img, return_tensors="pt")
                    image_tensor_list.append(batch.pixel_values[0])
                    image_highres_tensor_list.append(batch.pixel_values[0])
                    image_sizes.append(resized_img.size)
                
                # 堆疊多個張量
                image_tensor = torch.stack(image_tensor_list, dim=0).bfloat16().to("cuda")
                image_highres_tensor = torch.stack(image_highres_tensor_list, dim=0).bfloat16().to("cuda")
        # 設定 padding token
        pad_token_ids = 151643
        attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')
        
        # 設定停止條件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        # 生成參數
        gen_kwargs = {
            "max_new_tokens": 1024,  # 對於完整回答，設定較大的 token 數量
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1
        }
        
        # 模型推論
        with torch.inference_mode():
            if modality == "video":
                output_ids = model.generate(
                    inputs=input_ids,
                    images=video_data[0][0],
                    images_highres=video_data[0][1],
                    modalities=video_data[2],
                    speech=speechs,
                    speech_lengths=speech_lengths,
                    speech_chunks=speech_chunks,
                    speech_wav=speech_wavs,
                    attention_mask=attention_masks,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
            elif  modality == "image":
                    output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    images_highres=image_highres_tensor,
                    image_sizes=image_sizes,
                    modalities=['image'],  # 確保這裡是 'image'
                    speech=speechs,
                    speech_lengths=speech_lengths,
                    speech_chunks=speech_chunks,
                    speech_wav=speech_wavs,
                    attention_mask=attention_masks,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    )
            else:
                # 純文字推理
                output_ids = model.generate(
                    inputs=input_ids,
                    speech=speechs,
                    speech_lengths=speech_lengths,
                    speech_chunks=speech_chunks,
                    speech_wav=speech_wavs,
                    attention_mask=attention_masks,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
        
        # 處理輸出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        
        # 移除停止詞
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        # 尋找單個字母答案 (A, B, C, D) 
        answer_match = re.search(r'\b[A-D]\b', outputs)
        if answer_match:
            return answer_match.group(0), outputs
        else:
            # 如果沒有找到明確的選項，返回整個輸出和完整回答
            return outputs, outputs
        
    except Exception as e:
        print(f"推論過程中發生錯誤: {e}")
        traceback.print_exc()
        return "推論失敗，請查看錯誤日誌。", None

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="執行 Ola-7B 模型對資料集進行推論")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="THUdyh/Ola-7b",
        help="要載入的模型名稱或路徑 (預設為 THUdyh/Ola-7b)"
    )
    parser.add_argument(
        "--tensor_type",
        type=str,
        default="bf16",
        choices=["auto", "fp16", "bf16", "int8", "fp4", "nf4"],
        help="模型載入時使用的張量類型 (影響量化和精度)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="資料集的路徑 (例如 Hugging Face Hub 上的名稱，或本地 JSON/CSV 檔案路徑)"
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="資料集的特定配置名稱 (例如 MMBench 的 'cc')"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="要使用的資料集分割 (例如 'train', 'test', 'validation')"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face Hub 的 token (如果模型或私有資料集需要授權)"
    )
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        default=None,
        help="Prompt 模板檔案的路徑 (例如 prompt/base.txt)"
    )
    parser.add_argument(
        "--video_field",
        type=str,
        default="video",
        help="資料集中表示影片路徑的欄位名稱 (可選)"
    )
    parser.add_argument(
        "--id_field",
        type=str,
        default="id",
        help="資料集中表示樣本唯一ID的欄位名稱 (用於結果追蹤)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="儲存推論結果的目錄"
    )

    args = parser.parse_args()

    # --- Output directory setup ---
    output_base_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_output_dir = output_base_dir / timestamp
    current_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"結果將儲存於: {current_output_dir}")

    # --- Load Prompt Template ---
    prompt_template = "{question}"
    if args.prompt_template_path:
        try:
            with Path(args.prompt_template_path).open("r", encoding="utf-8") as f:
                prompt_template = f.read()
            print(f"從 {args.prompt_template_path} 載入 Prompt 模板成功。")
        except FileNotFoundError:
            print(f"警告: Prompt 模板檔案 {args.prompt_template_path} 未找到，將使用預設模板。")
        except Exception as e:
            print(f"警告: 載入 Prompt 模板 {args.prompt_template_path} 失敗: {e}，將使用預設模板。")
    else:
        print("未提供 Prompt 模板路徑，將使用預設模板。")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    elif os.getenv("HUGGINGFACE_TOKEN"):
        from huggingface_hub import login
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        print("使用環境變數 HUGGINGFACE_TOKEN 進行登入。")

    # 載入 Ola 模型
    tokenizer, model, image_processor = load_ola_model(args.model_name_or_path, args.tensor_type)

    print(f"正在載入資料集: {args.dataset_path} (config: {args.dataset_config_name}, split: {args.dataset_split})")
    try:
        data_files_arg = {args.dataset_split: args.dataset_path.split(";")}
        
        if args.dataset_path.lower().endswith(".json"):
            dataset = load_dataset("json", data_files=data_files_arg, split=args.dataset_split)
        elif args.dataset_path.lower().endswith(".csv"):
            dataset = load_dataset("csv", data_files=data_files_arg, split=args.dataset_split)
        else: 
            dataset = load_dataset(args.dataset_path, name=args.dataset_config_name, split=args.dataset_split, trust_remote_code=True)
    except Exception as e:
        print(f"載入資料集失敗: {e}")
        traceback.print_exc()
        print("請確認資料集路徑、設定和分割是否正確，以及是否需要登入 Hugging Face Hub。")
        return

    total_samples = len(dataset)
    print(f"開始對資料集的 {total_samples} 個樣本進行推論...")

    start_time = time.time()

    all_results_data = []
    all_choices_map = ["A", "B", "C", "D"]

    for i, example in enumerate(dataset):
        item_id = example.get(args.id_field, f"item_{i}")
        
        # --- 預處理和模板應用 ---
        instruction_text = example.get("instruction", "")
        question_text = example.get("question", "")
        options_text = []
        index_to_ans_map = {}

        # 處理選項欄位
        raw_options = {}
        for idx, choice_letter in enumerate(all_choices_map):
            option_key = f"option{idx+1}"
            option_val = example.get(option_key, "")
            raw_options[choice_letter] = option_val
            if option_val:
                options_text.append(option_val)
            
            if option_val and isinstance(option_val, str) and "）" in option_val:
                index_to_ans_map[choice_letter] = option_val.split("）", 1)[-1].strip()
            elif option_val:
                index_to_ans_map[choice_letter] = option_val
            else:
                index_to_ans_map[choice_letter] = ""
        
        # 組合提示文字
        if not question_text and instruction_text:
            print(f"資訊：樣本 {item_id} (索引 {i}) 的 'question' 欄位為空，但 'instruction' 存在。將使用 instruction 和 options 組合 prompt。")
            combined_text_for_template = f"{instruction_text}\n{''.join(options_text)}"
        elif not question_text and not instruction_text and not options_text:
            audio_path_check = example.get("audio")
            video_path_check = example.get(args.video_field)
            if not audio_path_check and not video_path_check:
                print(f"警告：樣本 {item_id} (索引 {i}) 同時缺少 'question', 'instruction', 'options', 'audio', 'video'。將跳過。")
                continue
            else:
                print(f"資訊：樣本 {item_id} (索引 {i}) 的 'question', 'instruction', 'options' 為空，但存在音訊/影片。將使用通用提示。")
                combined_text_for_template = "請根據提供的多媒體內容回答問題。"
        else:
            combined_text_for_template = f"{instruction_text}\n{question_text}\n{''.join(options_text)}"
        
        final_text_prompt = prompt_template.format(question=combined_text_for_template.strip())
        
        original_answer = example.get("answer")

        # 收集多媒體路徑
        image_path = example.get("image") 
        audio_path = example.get("audio") 
        video_path = example.get(args.video_field)

        # 處理圖片路徑
        current_image_paths = []
        if image_path:
            paths_to_check = [image_path] if isinstance(image_path, str) else image_path if isinstance(image_path, list) else []
            for p in paths_to_check:
                if isinstance(p, str) and os.path.exists(p):
                    current_image_paths.append(p)
                elif isinstance(p, str):
                    print(f"警告: 樣本 {item_id} 的圖片路徑 {p} 不存在。")
            if not current_image_paths: current_image_paths = None
        else:
            current_image_paths = None
        
        # 處理音訊路徑
        current_audio_paths = []
        if audio_path:
            paths_to_check = [audio_path] if isinstance(audio_path, str) else audio_path if isinstance(audio_path, list) else []
            for p in paths_to_check:
                if isinstance(p, str) and os.path.exists(p):
                    current_audio_paths.append(p)
                elif isinstance(p, str):
                    print(f"警告: 樣本 {item_id} 的音訊路徑 {p} 不存在。")
            if not current_audio_paths: current_audio_paths = None
        else:
            current_audio_paths = None

        # 處理影片路徑
        current_video_paths = []
        if video_path:
            paths_to_check = [video_path] if isinstance(video_path, str) else video_path if isinstance(video_path, list) else []
            for p in paths_to_check:
                if isinstance(p, str) and os.path.exists(p):
                    current_video_paths.append(p)
                elif isinstance(p, str):
                    print(f"警告: 樣本 {item_id} 的影片路徑 {p} 不存在。")
            if not current_video_paths: current_video_paths = None
        else:
            current_video_paths = None

        print(f"\n處理樣本 ID: {item_id} ( {i+1} / {total_samples} )")
        
        # 進行模型推論
        response, full_response = run_inference(
            tokenizer, model, image_processor, 
            final_text_prompt, 
            current_image_paths, 
            current_audio_paths, 
            current_video_paths
        )
        
        response_preview = str(response)[:200].replace('\n', ' ')
        print(f"  模型回應 (ID: {item_id}): {response_preview}...")
        
        all_results_data.append({
            "id": item_id,
            "question": final_text_prompt,
            "image_path(s)": current_image_paths if current_image_paths else [],
            "audio_path(s)": current_audio_paths if current_audio_paths else [],
            "video_path(s)": current_video_paths if current_video_paths else [],
            "generation": response.strip(),
            "full_response": full_response,
            "answer": original_answer,
            "index2ans": index_to_ans_map
        })
        
        # 每50個樣本或最後一個樣本時儲存臨時結果
        if (i + 1) % 50 == 0 or (i + 1) == total_samples:
            temp_output_filename = current_output_dir / f"ola_results_{os.path.basename(args.dataset_path).replace('.json','').replace('.csv','').replace('.jsonl','').replace('/','_')}_{args.dataset_split}_temp.json"
            try:
                with open(temp_output_filename, "w", encoding="utf-8") as f:
                    json.dump(all_results_data, f, ensure_ascii=False, indent=4)
                print(f"臨時結果 ({len(all_results_data)}/{total_samples}) 已儲存到 {temp_output_filename}")
            except Exception as e:
                print(f"儲存臨時結果失敗: {e}")

        clear_cuda_cache()

    print("\n所有樣本推論完成。")
    clear_cuda_cache()
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"總執行時間: {runtime:.2f} 秒")

    # --- 計算準確率 ---
    correct_predictions = 0
    if all_results_data:
        for item in all_results_data:
            if isinstance(item.get("generation"), str) and isinstance(item.get("answer"), str):
                if item["generation"].strip().upper() == item["answer"].strip().upper():
                    correct_predictions += 1
        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        print(f"準確率 (Accuracy): {accuracy:.2f}%")
    else:
        accuracy = 0
        print("沒有結果可計算準確率。")

    # --- 儲存設定檔案 ---
    config_data = {
        "model_name_or_path": args.model_name_or_path,
        "tensor_type": args.tensor_type,
        "dataset_path": args.dataset_path,
        "dataset_config_name": args.dataset_config_name,
        "dataset_split": args.dataset_split,
        "prompt_template_path": args.prompt_template_path if args.prompt_template_path else "使用預設模板",
        "prompt_template_content": prompt_template,
        "max_new_tokens_for_generation": 1024,
        "id_field": args.id_field,
        "video_field": args.video_field,
        "output_dir": str(current_output_dir),
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "accuracy": f"{accuracy:.2f}%",
        "runtime_seconds": f"{runtime:.2f}"
    }
    config_filename = current_output_dir / "config.json"
    try:
        with open(config_filename, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        print(f"設定檔案已儲存到 {config_filename}")
    except Exception as e:
        print(f"儲存設定檔案失敗: {e}")

    # --- 儲存最終結果 ---
    final_results_filename = current_output_dir / f"results.json"
    try:
        with open(final_results_filename, "w", encoding="utf-8") as f:
            json.dump(all_results_data, f, ensure_ascii=False, indent=4)
        print(f"最終結果已儲存到 {final_results_filename}")
        
        temp_output_filename_to_check = current_output_dir / f"ola_results_{os.path.basename(args.dataset_path).replace('.json','').replace('.csv','').replace('.jsonl','').replace('/','_')}_{args.dataset_split}_temp.json"
        if os.path.exists(temp_output_filename_to_check):
            if len(all_results_data) == total_samples:
                 os.remove(temp_output_filename_to_check)
                 print(f"已刪除臨時檔案: {temp_output_filename_to_check}")

    except Exception as e:
        print(f"儲存最終結果失敗: {e}")
        print("原始結果 (可能不完整):")
        for res_item in all_results_data:
            print(res_item)

if __name__ == "__main__":
    # 確保必要的套件已安裝
    required_packages = ["datasets", "huggingface_hub", "librosa", "whisper", "moviepy", "decord"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"錯誤: 缺少必要的套件: {', '.join(missing_packages)}. 請執行: pip install {' '.join(missing_packages)}")
        exit(1)
        
    main() 