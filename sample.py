from typing import Callable

import torch

from export import load_bin
from model import GemmaModel, Tensor


@torch.inference_mode()
def sample(
    model: GemmaModel,
    tokens: Tensor,  # (B, T)
    seq_len: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    token_callback: Callable[[Tensor, Tensor], Tensor | bool] | None = None,
):
    model.eval()
    config = model.config
    device = tokens.device
    
    inputs = tokens
    ninf = torch.full((seq_len, seq_len), float("-inf"), dtype=config.dtype, device=device)
    global_mask = torch.triu(ninf, 1)
    local_mask = global_mask + torch.tril(ninf, -config.sliding_window)
    kv_cache_layers = []

    i = 0
    while i < seq_len - tokens.size(1):
        positions = torch.arange(i, i + inputs.size(-1), device=device)
        logits = model.call(inputs, positions, seq_len, global_mask, local_mask, kv_cache_layers)
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
                mask = torch.zeros(B, config.vocab_size, dtype=torch.bool, device=device)
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

        if token_callback is not None:
            probs = logits.softmax(-1)
            callback_return = token_callback(next_tok, probs)
            if isinstance(callback_return, torch.Tensor):
                inputs = callback_return
            elif callback_return:
                break
            
    return tokens


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
    model = model.to(device)

    def new_turn():
        user_input = input("\nUser: ")
        inputs = torch.tensor(
            tokenizer.apply_chat_template([{"role": "user", "content": user_input}]),
            device=device,
        ).unsqueeze(0)

        print("Model: ", end="", flush=True)
        return inputs

    def token_callback(token: Tensor, probs: Tensor):
        tok = token.tolist()[0]
        if tok == tokenizer.vocab["<end_of_turn>"]:
            return new_turn()
        
        print(tokenizer.decode([tok]), end="", flush=True)
        return False

    try:
        sample(
            model,
            new_turn(),
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
    generate("./exported/gemma-3-4b-it_q.bin", prompt="Once upon a time", device="cpu")
