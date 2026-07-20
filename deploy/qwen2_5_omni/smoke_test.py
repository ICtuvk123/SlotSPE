#!/usr/bin/env python
import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def build_conversation(args: argparse.Namespace) -> list[dict]:
    content = [{"type": "text", "text": args.prompt}]
    if args.image:
        content.append({"type": "image", "image": args.image})
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    if args.video:
        content.append({"type": "video", "video": args.video})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni local smoke test")
    parser.add_argument("--model-path", default="deploy/qwen2_5_omni/models/Qwen2.5-Omni-3B")
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    parser.add_argument("--video")
    parser.add_argument("--output-audio", default="deploy/qwen2_5_omni/output.wav")
    parser.add_argument("--no-audio", action="store_true", help="Return text only.")
    parser.add_argument("--flash-attn2", action="store_true", help="Use FlashAttention 2.")
    args = parser.parse_args()

    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.flash_attn2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    conversation = build_conversation(args)
    use_audio_in_video = bool(args.video)
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

    if args.no_audio:
        text_ids = model.generate(**inputs, use_audio_in_video=use_audio_in_video, return_audio=False)
        decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(decoded[0])
        return

    text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio_in_video)
    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(decoded[0])

    output_audio = Path(args.output_audio)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
    print(f"Wrote audio to {output_audio}")


if __name__ == "__main__":
    main()

