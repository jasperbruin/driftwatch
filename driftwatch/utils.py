import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_embeddings(model, tokenizer, texts, device):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encodings = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
            if model.config.model_type in ["gpt2", "gpt_neo", "opt", "mistral", "falcon", "bloom"]:
                return hidden_states[:, -1, :].cpu().numpy()
            elif model.config.model_type in ["t5", "mbart"]:
                return hidden_states.mean(dim=1).cpu().numpy()
            elif "bert" in model.config.model_type or "electra" in model.config.model_type:
                return hidden_states[:, 0, :].cpu().numpy()
        return None

def batch_generator(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

def introduce_gradual_drift(text_list, fraction_shuffle=0.5):
    new_texts = []
    for txt in text_list:
        words = txt.split()
        if len(words) < 2:
            new_texts.append(txt)
            continue
        k = int(len(words) * fraction_shuffle)
        if k < 1:
            new_texts.append(txt)
            continue
        indices = list(range(len(words)))
        random.shuffle(indices)
        shuffle_indices = indices[:k]
        to_shuffle = [words[i] for i in shuffle_indices]
        random.shuffle(to_shuffle)
        for i, idx in enumerate(shuffle_indices):
            words[idx] = to_shuffle[i]
        new_texts.append(" ".join(words))
    return new_texts

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon (Metal Performance Shaders)
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")   # Fallback to CPU