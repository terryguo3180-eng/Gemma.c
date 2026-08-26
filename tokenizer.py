from io import BytesIO
from PIL import Image
from urllib import request

import torch

from preprocess import preprocess_image


class GemmaTokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        tokens_per_image: int,
        image_size: int,
    ):
        self.vocab = vocab
        self.idx_to_str = {i: s for s, i in vocab.items()}
        self.tokens_per_image = tokens_per_image
        self.image_size = image_size

        self.ranks = {}
        self.merges = {}

        for i, (s1, s2) in enumerate(merges):
            pair = (vocab[s1], vocab[s2])
            self.ranks[pair] = i
            self.merges[pair] = vocab[s1 + s2]

    def encode(self, text: str) -> list[int]:
        tokens = [self.vocab["<bos>"]]
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                for byte in char.encode("utf-8", errors="replace"):
                    tokens.append(self.vocab[f"<0x{hex(byte)[2:].upper()}>"])

        while True:
            best_rank = float("inf")
            best_pair = None

            for pair in zip(tokens[1:], tokens[2:]):
                if ((rank := self.ranks.get(pair)) is not None) and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            i = 1
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i + 1])
                if pair == best_pair:
                    tokens[i:i + 2] = [self.merges[pair]]
                i += 1

        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.idx_to_str[tok] for tok in tokens)

    def apply_chat_template(self, conversation: (
        list[  # list of all the turns
            dict[
                str,  # "role" | "content"
                str |  # values for "role": "user" | "model" | "assistant"
                list[  # values for "content": list of all the inputs
                    dict[
                        str,  # "type" | "text" | "url" | "image"
                        str,  # values
    ]]]])) -> dict[str, torch.Tensor]:
        
        tokens = [self.vocab["<bos>"]]
        pixel_values = []

        for turn in conversation:
            tokens.append(self.vocab["<start_of_turn>"])

            role = "model" if turn["role"] in {"model", "assistant"} else turn["role"]
            assert isinstance(role, str)
            tokens.extend(self.encode(role)[1:])  # exclude <bos> at the beginning
            tokens.append(self.vocab["\n"])

            content = turn["content"]
            assert not isinstance(content, str)
            for input in content:
                if input["type"] == "text":
                    tokens.extend(self.encode(input["text"])[1:])
                elif input["type"] == "image":
                    tokens.append(self.vocab["\n"])
                    tokens.append(self.vocab["\n"])
                    tokens.append(self.vocab["<start_of_image>"])
                    for _ in range(self.tokens_per_image):
                        # Image placeholders
                        tokens.append(self.vocab["<image_soft_token>"])
                    tokens.append(self.vocab["<end_of_image>"])
                    tokens.append(self.vocab["\n"])
                    tokens.append(self.vocab["\n"])

                    # Get image path / image object
                    if "url" in input:
                        path = input["url"]
                    elif "image" in input:
                        path = input["image"]
                    else:
                        assert False

                    if isinstance(path, str) and path.startswith("https://"):
                        with request.urlopen(path) as resp:
                            img = Image.open(BytesIO(resp.read()))
                    elif isinstance(path, str):
                        img = Image.open(path)
                    elif isinstance(path, Image):
                        img = path
                    else:
                        assert False

                    img = preprocess_image(img, self.image_size)
                    pixel_values.append(img)

            tokens.append(self.vocab["<end_of_turn>"])
            tokens.append(self.vocab["\n"])

        # Model's turn next
        tokens.append(self.vocab["<start_of_turn>"])
        tokens.extend(self.encode("model")[1:])
        tokens.append(self.vocab["\n"])

        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0)  # (n_images, 3, H, W)
        else:
            pixel_values = torch.empty(0)

        return {
            "tokens": torch.tensor([tokens], dtype=torch.long),
            "pixel_values": pixel_values,
        }
