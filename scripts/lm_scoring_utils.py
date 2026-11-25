"""Small causal-LM scoring helpers used by the context replacement experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


DEFAULT_MODEL = "hsc748NLP/GujiGPT_fan"
DEFAULT_INPUT_PATH = "data/1_mengzi_prose/replacement_source.txt"
PUNCTUATION = set(
    "，。！？；：、“”‘’（）《》〈〉【】〔〕—…·,.!?;:\"'()[]{}<>"
)


@dataclass
class CharScore:
    position: int
    char: str
    perplexity: float | None
    log_probability: float | None


def causal_lm_context_limit(model, tokenizer) -> int:
    """Best-effort context length for a causal LM/tokenizer pair."""
    limits = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)

    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        model_limit = getattr(config, attr, None)
        if isinstance(model_limit, int) and model_limit > 0:
            limits.append(model_limit)

    return min(limits) if limits else 1024


def tokenize_with_char_positions(tokenizer, text: str):
    """Tokenize text and map each source character to token positions."""
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        char_to_token_positions: list[list[int]] = [[] for _ in range(len(text))]
        for token_pos, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            for char_pos in range(max(0, start), min(len(text), end)):
                char_to_token_positions[char_pos].append(token_pos)
        return inputs, char_to_token_positions
    except (NotImplementedError, TypeError, ValueError):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        char_to_token_positions = [[] for _ in range(len(text))]
        char_cursor = 0
        for token_pos, token in enumerate(tokens):
            if token in {"[PAD]", "[CLS]", "[SEP]", "[MASK]"}:
                piece_len = 0
            elif token == "[UNK]":
                piece_len = 1
            else:
                piece = token[2:] if token.startswith("##") else token
                piece_len = len(piece)
            for _ in range(piece_len):
                if char_cursor < len(text):
                    char_to_token_positions[char_cursor].append(token_pos)
                    char_cursor += 1
        return inputs, char_to_token_positions


def score_token_log_probabilities(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_context_tokens: int,
    score_block_tokens: int,
) -> list[float | None]:
    """Score each token from previous context, using sliding chunks for long text."""
    log_probabilities, _ = score_token_log_probabilities_and_entropies(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_context_tokens=max_context_tokens,
        score_block_tokens=score_block_tokens,
    )
    return log_probabilities


def score_token_log_probabilities_and_entropies(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_context_tokens: int,
    score_block_tokens: int,
) -> tuple[list[float | None], list[float | None]]:
    """Return each token's log-probability and the entropy (nats) of the
    next-token distribution it was drawn from, using sliding chunks for long text."""
    token_count = input_ids.shape[1]
    token_log_probabilities: list[float | None] = [None] * token_count
    token_entropies: list[float | None] = [None] * token_count
    if token_count <= 1:
        return token_log_probabilities, token_entropies

    block_size = max(1, min(score_block_tokens, max_context_tokens // 2 or 1))
    for target_start in range(1, token_count, block_size):
        target_end = min(token_count, target_start + block_size)
        chunk_start = max(0, target_end - max_context_tokens)
        if chunk_start >= target_start:
            chunk_start = max(0, target_start - 1)

        chunk_input_ids = input_ids[:, chunk_start:target_end]
        model_inputs = {"input_ids": chunk_input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask[:, chunk_start:target_end]

        with torch.no_grad():
            logits = model(**model_inputs).logits[0]

        for token_pos in range(target_start, target_end):
            local_prev_pos = token_pos - chunk_start - 1
            if local_prev_pos < 0:
                continue
            token_log_probs = F.log_softmax(logits[local_prev_pos].float(), dim=-1)
            token_id = int(input_ids[0, token_pos].item())
            token_log_probabilities[token_pos] = float(token_log_probs[token_id].item())
            token_entropies[token_pos] = float(
                -(token_log_probs.exp() * token_log_probs).sum().item()
            )

    return token_log_probabilities, token_entropies


def score_char_perplexities(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_context_tokens: int,
    score_block_tokens: int,
) -> list[CharScore]:
    """Compute per-character perplexity conditioned on previous tokens."""
    inputs, char_to_token_positions = tokenize_with_char_positions(tokenizer, text)
    input_ids = inputs["input_ids"].to(device)
    if input_ids.shape[1] == 0:
        raise ValueError("Could not tokenize input text.")

    attention_mask = inputs.get("attention_mask")
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    token_log_probabilities = score_token_log_probabilities(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_context_tokens=max_context_tokens,
        score_block_tokens=score_block_tokens,
    )

    scores: list[CharScore] = []
    for char_pos, char in enumerate(text):
        token_positions = char_to_token_positions[char_pos]
        if not token_positions:
            scores.append(CharScore(char_pos, char, None, None))
            continue

        log_probability = 0.0
        valid = True
        for token_pos in token_positions:
            token_log_probability = token_log_probabilities[token_pos]
            if token_log_probability is None:
                valid = False
                break
            log_probability += token_log_probability

        if not valid:
            scores.append(CharScore(char_pos, char, None, None))
            continue

        negative_log_probability = -log_probability
        perplexity = math.exp(negative_log_probability) if negative_log_probability < 700 else math.inf
        scores.append(CharScore(char_pos, char, perplexity, log_probability))

    return scores
