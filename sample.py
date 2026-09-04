import re

from typing import Callable

import torch

from export import load_bin
from model import GemmaModel, Tensor


def build_mask(
    tokens: Tensor,  # (1, T)
    seq_len: int,
    sliding_window: int,
    image_start_tok: int,
    image_end_tok: int,
    image_soft_tok: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    
    # This part is kind of confusing. I'm drawing the result in each step
    ninf = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    # ninf
    # [-inf -inf -inf -inf -inf -inf ]
    # [-inf -inf -inf -inf -inf -inf ]
    # [-inf -inf -inf -inf -inf -inf ]
    # [-inf -inf -inf -inf -inf -inf ]
    # [-inf -inf -inf -inf -inf -inf ]
    # [-inf -inf -inf -inf -inf -inf ]
    causal = torch.triu(ninf, 1)
    # causal:
    # [  0  -inf -inf -inf -inf -inf ]
    # [  0    0  -inf -inf -inf -inf ]
    # [  0    0    0  -inf -inf -inf ]
    # [  0    0    0    0  -inf -inf ]
    # [  0    0    0    0    0  -inf ]
    # [  0    0    0    0    0    0  ]
    windowed = torch.tril(ninf, -sliding_window)
    # windowed:
    # [  0    0    0    0    0    0  ]
    # [  0    0    0    0    0    0  ]
    # [  0    0    0    0    0    0  ]
    # [-inf   0    0    0    0    0  ]
    # [-inf -inf   0    0    0    0  ]
    # [-inf -inf -inf   0    0    0  ]

    T = tokens.size(1)
    image_id = torch.full((seq_len,), -1, dtype=torch.long, device=tokens.device)

    inside = False
    for t in range(T):
        tok = tokens[0, t].item()
        if tok == image_start_tok:
            inside = True
        elif tok == image_end_tok:
            inside = False
        elif tok == image_soft_tok and inside:
            image_id[t] = True

    same = image_id.unsqueeze(0) == image_id.unsqueeze(1)  # (T, T)
    is_image = image_id.unsqueeze(0) != -1
    image_mask = same & is_image
    # image_mask:
    # [  0    0    0    0    0    0  ]
    # [  0    0    0    0    0    0  ]
    # [  0    0    1    1    0    0  ]
    # [  0    0    1    1    0    0  ]
    # [  0    0    0    0    0    0  ]
    # [  0    0    0    0    0    0  ]
    global_mask = causal.masked_fill(image_mask, 0.0)
    # global_mask:
    # [  0  -inf -inf -inf -inf -inf ]
    # [  0    0  -inf -inf -inf -inf ]
    # [  0    0    0    0  -inf -inf ]
    # [  0    0    0    0  -inf -inf ]
    # [  0    0    0    0    0  -inf ]
    # [  0    0    0    0    0    0  ]
    local_mask = (causal + windowed).masked_fill(image_mask, 0.0)
    # local_mask:
    # [  0  -inf -inf -inf -inf -inf ]
    # [  0    0  -inf -inf -inf -inf ]
    # [  0    0    0    0  -inf -inf ]
    # [-inf   0    0    0  -inf -inf ]
    # [-inf -inf   0    0    0  -inf ]
    # [-inf -inf -inf   0    0    0  ]

    return global_mask, local_mask


@torch.inference_mode()
def sample(
    model: GemmaModel,
    tokens: Tensor,  # (B, T)
    pixel_values: Tensor,  # (n_images, 3, H, W)
    seq_len: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    token_callback: Callable[[Tensor, Tensor], dict[str, Tensor] | bool] | None = None,
):
    model.eval()
    config = model.config
    device = tokens.device
    
    inputs = tokens
    # Build the mask
    global_mask, local_mask = build_mask(
        tokens, seq_len, config.sliding_window,
        model.image_start_token, model.image_end_token, model.image_soft_token,
        config.dtype, device,
    )
    kv_cache_layers = []

    i = 0
    emb = model.get_embeddings(inputs, pixel_values)

    while i < seq_len - tokens.size(1):
        positions = torch.arange(i, i + inputs.size(-1), device=device)
        logits = model.call(emb, positions, seq_len, global_mask, local_mask, kv_cache_layers)
        i += inputs.size(-1)

        logits = logits[:, -1, :]
        if temperature == 0.0:
            next_tok = logits.argmax(-1)
        else:
            logits = logits / temperature

            if top_k is not None:
                topk_values, topk_indices = logits.topk(top_k, -1)
                mask = torch.full_like(logits, float("-inf"), device=device)
                mask.scatter_(-1, topk_indices, topk_values)
                logits = mask

            if top_p is not None:
                sorted_logits, sorted_indices = logits.sort(-1, True)
                sorted_probs = sorted_logits.softmax(-1)
                cumsum_probs = sorted_probs.cumsum(-1)
                remove_indices = cumsum_probs > top_p
                remove_indices[:, 1:] = remove_indices[:, :-1].clone()
                remove_indices[:, 0] = False
                sorted_logits[remove_indices] = float("-inf")
                logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

            if repetition_penalty is not None and repetition_penalty > 1.0:
                B = logits.size(0)
                # vocab_size + 1 for the extra <image_soft_token>
                mask = torch.zeros(B, config.vocab_size + 1, dtype=torch.bool, device=device)
                mask.scatter_(-1, tokens, True)
                masked_logits = logits[mask]
                pos_mask = masked_logits > 0
                neg_mask = masked_logits < 0
                masked_logits[pos_mask] /= repetition_penalty
                masked_logits[neg_mask] *= repetition_penalty
                logits[mask] = masked_logits

            probs = logits.softmax(-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

        inputs = next_tok.unsqueeze(1)
        tokens = torch.cat((tokens, inputs), dim=1)
        emb = model.embedding(inputs)

        if token_callback is not None:
            probs = logits.softmax(-1)
            callback_return = token_callback(next_tok, probs)
            if isinstance(callback_return, dict):
                inputs = callback_return["tokens"]
                pixel_values = callback_return["pixel_values"].to(config.dtype)
                # Build the mask again
                global_mask, local_mask = build_mask(
                    tokens, seq_len, config.sliding_window,
                    model.image_start_token, model.image_end_token, model.image_soft_token,
                    config.dtype, device,
                )
                emb = model.get_embeddings(inputs, pixel_values)
            elif callback_return:
                break
            
    return tokens


def user_template(text: str, is_multimodal):
    if is_multimodal:
        path_regex = r"([a-zA-Z]:\\)?([^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]+\.[a-zA-Z0-9]+"
        image_regex = fr"@image\{{({path_regex})\}}"

        template = [{"role": "user", "content": []}]
        last_end = 0

        for match in re.finditer(image_regex, text):
            # Add text before the match
            if match.start() > last_end:
                template[0]["content"].append({
                    "type": "text", 
                    "text": text[last_end:match.start()]
                })
            
            # Add the image
            template[0]["content"].append({"type": "image", "image": match.group(1)})
            
            last_end = match.end()

        # Add remaining text
        if last_end < len(text):
            template[0]["content"].append({"type": "text", "text": text[last_end:]})
    else:
        template = [{"role": "user", "content": [{"type": "text", "text": text}]}]

    return template


def chat(
    model_path: str,
    seq_len: int = 10000,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    device: str = "cpu",
):
    model, tokenizer = load_bin(model_path)
    model: GemmaModel = torch.compile(model)  # type: ignore
    model = model.to(device)

    def new_turn():
        user_input = input("\nUser: ")
        template = user_template(user_input, model.multimodal)
        inputs = tokenizer.apply_chat_template(template)
        print("Model: ", end="", flush=True)
        return inputs

    def token_callback(token: Tensor, probs: Tensor):
        tok = token.tolist()[0]
        if tok == tokenizer.vocab["<end_of_turn>"]:
            return new_turn()
        
        print(tokenizer.decode([tok]), end="", flush=True)
        return False

    try:
        inputs = new_turn()
        sample(
            model,
            inputs["tokens"],
            inputs["pixel_values"].to(model.config.dtype),
            seq_len=seq_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            token_callback=token_callback,
        )

    except KeyboardInterrupt:
        pass


def generate(
    model_path: str,
    prompt: str,
    seq_len: int = 10000,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    device: str = "cpu",
):
    model, tokenizer = load_bin(model_path)
    model = model.to(device)

    def token_callback(token: Tensor, probs: Tensor):
        tok = token.tolist()[0]
        if tok == tokenizer.vocab["<eos>"]:
            return True

        print(tokenizer.decode([tok]), end="", flush=True)
        return False

    print(prompt, end="", flush=True)
    inputs = torch.tensor([tokenizer.encode(prompt)], device=device)

    try:
        sample(
            model,
            inputs,
            torch.tensor([]),
            seq_len=seq_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            token_callback=token_callback,
        )

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    chat("./exported/gemma-3-4b-it-fp16_q", device="cpu", temperature=0.0, seq_len=512)
