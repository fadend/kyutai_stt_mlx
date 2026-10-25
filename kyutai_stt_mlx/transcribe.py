# Based on the source code copyright Kyutai Labs, licensed under Apache and MIT.
# See https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/scripts/stt_from_file_mlx.py

import argparse
import json
import sys

import mlx.core as mx
import mlx.nn as nn
import sentencepiece
from kyutai_stt_mlx import sound
import huggingface_hub
import moshi_mlx

_DEFAULT_MODEL = "kyutai/stt-1b-en_fr-mlx"
_DEFAULT_MAX_STEPS = 4096


class Transcriber:
    def __init__(
        self, max_steps: int = _DEFAULT_MAX_STEPS, hf_repo: str = _DEFAULT_MODEL
    ):
        lm_config = huggingface_hub.hf_hub_download(hf_repo, "config.json")
        with open(lm_config, "rt", encoding="utf8") as fobj:
            lm_config = json.load(fobj)
        print(f"lm_config:\n{json.dumps(lm_config, indent=2)}", file=sys.stderr)
        mimi_weights = huggingface_hub.hf_hub_download(hf_repo, lm_config["mimi_name"])
        moshi_name = lm_config.get("moshi_name", "model.safetensors")
        moshi_weights = huggingface_hub.hf_hub_download(hf_repo, moshi_name)
        text_tokenizer = huggingface_hub.hf_hub_download(
            hf_repo, lm_config["tokenizer_name"]
        )

        lm_config = moshi_mlx.models.LmConfig.from_config_dict(lm_config)
        model = moshi_mlx.models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        if moshi_weights.endswith(".q4.safetensors"):
            nn.quantize(model, bits=4, group_size=32)
        elif moshi_weights.endswith(".q8.safetensors"):
            nn.quantize(model, bits=8, group_size=64)

        print(f"Loading model weights from {moshi_weights}:", file=sys.stderr)
        if hf_repo.endswith("-candle"):
            model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
        else:
            model.load_weights(moshi_weights, strict=True)

        print(f"Loading the text tokenizer from {text_tokenizer}", file=sys.stderr)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(text_tokenizer)  # type: ignore

        print(f"Loading the audio tokenizer {mimi_weights}", file=sys.stderr)
        self.audio_tokenizer = moshi_mlx.models.mimi.Mimi(
            moshi_mlx.models.mimi_202407(32)
        )
        self.audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)
        print("Warming up the model", file=sys.stderr)
        model.warmup()
        self.gen = moshi_mlx.models.LmGen(
            model=model,
            max_steps=max_steps,
            text_sampler=moshi_mlx.utils.Sampler(top_k=25, temp=0),
            audio_sampler=moshi_mlx.utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )

    def transcribe(self, in_file, out_file):
        audio = sound.read_sound(in_file, sample_rate=24000)
        print(f"starting inference {audio.shape}", file=sys.stderr)
        audio = mx.concat([mx.array(audio), mx.zeros((1, 48000))], axis=-1)
        for start_idx in range(0, audio.shape[-1] // 1920 * 1920, 1920):
            block = audio[:, None, start_idx : start_idx + 1920]
            other_audio_tokens = self.audio_tokenizer.encode_step(block).transpose(
                0, 2, 1
            )
            text_token = self.gen.step(other_audio_tokens[0])
            text_token = text_token[0].item()
            audio_tokens = self.gen.last_audio_tokens()
            text = None
            if text_token not in (0, 3):
                text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                text = text.replace("▁", " ")
                print(text, end="", flush=True, file=out_file)
        print(file=out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="The file to transcribe.")
    parser.add_argument("--max-steps", default=_DEFAULT_MAX_STEPS, type=int)
    parser.add_argument(
        "--hf-repo",
        default=_DEFAULT_MODEL,
        help="Hugging Face repo with the models to use.",
    )
    parser.add_argument("--output", help="Path at which to store output.")
    args = parser.parse_args()

    transcriber = Transcriber(hf_repo=args.hf_repo, max_steps=args.max_steps)
    with open(args.output, "wt", encoding="utf8") as out_file:
        transcriber.transcribe(args.input_path, out_file)


if __name__ == "__main__":
    main()
