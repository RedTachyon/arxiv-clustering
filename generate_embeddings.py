import numpy as np
import pickle

from math import ceil

from tqdm import trange

import torch
import transformers
from transformers import AutoTokenizer, AutoModel

from typarse import BaseParser


class Parser(BaseParser):
    model: str = "prajjwal1/bert-small"
    device: str = "cuda:0"
    batch_size: int = 100

    _help = {
        "model": "Model to use for embedding",
        "device": "Device to use for embedding",
        "batch_size": "Batch size for embedding"
    }

    _abbrev = {
        "model": "m",
        "device": "d",
        "batch_size": "b"
    }


def embed_abstract(text: list[str],
                   tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
                   model: torch.nn.Module) -> torch.Tensor:

    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        tokens = {k: v.to("cuda:1") for k, v in tokens.items()}
        out = model(**tokens)

    del tokens
    res = out[0].mean(1).cpu()
    del out

    return res


if __name__ == "__main__":
    args = Parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained('prajjwal1/bert-small')

    model.to(args.device)
    model.eval()

    # Disable gradients to save memory
    for param in model.parameters():
        param.requires_grad = False

    with open("abstracts.pkl", "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} abstracts for embedding")

    # Embed the abstracts in batches

    print("Embedding abstracts. This might take several hours, depending on the model and the hardware.")

    batches = ceil(len(data) / args.batch_size)

    all_embeds = []
    for i in trange(batches):
        batch = data[i * args.batch_size: (i + 1) * args.batch_size]
        embeds = embed_abstract(batch, model, tokenizer)
        all_embeds.append(embeds)

    embeds_full = np.concatenate([e.numpy() for e in all_embeds])

    np.save("big_embeds", embeds_full)
