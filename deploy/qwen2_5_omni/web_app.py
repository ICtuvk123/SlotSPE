#!/usr/bin/env python
import argparse
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Qwen2.5-Omni Gradio demo")
    parser.add_argument("--model-path", default="deploy/qwen2_5_omni/models/Qwen2.5-Omni-3B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--flash-attn2", action="store_true")
    return parser.parse_args()


args = parse_args()
model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
if args.flash_attn2:
    model_kwargs["attn_implementation"] = "flash_attention_2"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)


def infer(prompt: str, image_path: str | None, audio_path: str | None, video_path: str | None, return_audio: bool):
    content = [{"type": "text", "text": prompt or "Describe the input."}]
    if image_path:
        content.append({"type": "image", "image": image_path})
    if audio_path:
        content.append({"type": "audio", "audio": audio_path})
    if video_path:
        content.append({"type": "video", "video": video_path})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    use_audio_in_video = bool(video_path)

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device)
    if torch.is_floating_point(next(model.parameters())):
        inputs = inputs.to(model.dtype)

    if not return_audio:
        text_ids = model.generate(**inputs, use_audio_in_video=use_audio_in_video, return_audio=False)
        decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0], None

    text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio_in_video)
    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_audio = Path("deploy/qwen2_5_omni/web_output.wav")
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
    return decoded[0], str(output_audio)


with gr.Blocks(title="Qwen2.5-Omni Local") as demo:
    gr.Markdown("# Qwen2.5-Omni Local")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="Describe the input in one concise paragraph.")
            image = gr.Image(label="Image", type="filepath")
            audio = gr.Audio(label="Audio", type="filepath")
            video = gr.Video(label="Video")
            return_audio = gr.Checkbox(label="Generate speech", value=False)
            run = gr.Button("Run", variant="primary")
        with gr.Column():
            text_output = gr.Textbox(label="Text output", lines=12)
            audio_output = gr.Audio(label="Speech output", type="filepath")

    run.click(
        infer,
        inputs=[prompt, image, audio, video, return_audio],
        outputs=[text_output, audio_output],
    )

demo.launch(server_name=args.host, server_port=args.port)

