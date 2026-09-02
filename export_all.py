from pathlib import Path

import torch

from export import export_hf


with open("exported.txt", "r") as f:
    exported_models = f.read().splitlines()

models_path = Path("./models")
export_path = Path("./exported")

with open("exported.txt", "a") as f:
    for model in models_path.iterdir():
        for dtype, suf1 in [
            (torch.float32, "-fp32"),
            (torch.float16, "-fp16"),
            (torch.bfloat16, "-bf16"),
        ]:
            for quant, suf2 in [(False, ""), (True, "_q")]:
                outpath = str(export_path / (model.stem + suf1 + suf2))
                if outpath in exported_models:
                    continue
                print(f"Exporting '{outpath}'...")
                export_hf(str(model), outpath, dtype, quant)
                f.write(outpath + "\n")
