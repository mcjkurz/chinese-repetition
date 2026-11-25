"""
Measure verbatim memorization of regulated-verse couplets.

For each couplet the script gives the model the first line and greedy-decodes
the second. Exact-match rate is the share of second lines reproduced character
for character. Mean loss is the teacher-forced negative log probability of the
true second line, which still moves when exact match is rare.

Two prompts are scored:
- matched: the i/o prefix used in fine-tuning for that meter's model
- none: the first line and comma only, with no control token

The comparison that speaks to the reviewer is train versus the poem-held-out
test set, under the same prompt. A large exact-match or loss gap means
fine-tuning stored training couplets. The same comparison on the untuned
base model measures memorization that was already in pretraining. The script
refuses to run when the train and test files share a poem id.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import opencc
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from common import (
    METER_NAMES,
    ensure_padding_token,
    select_device,
    split_couplet,
)


METERS = (5, 7)
CONDITIONS = ("matched", "none")
DEFAULT_MODEL_PATH = "models/gujigpt-io-finetuned-penta"
DEFAULT_TRAIN_TEMPLATE = "data/2-3_regulated_verse/couplets_{meter}_train.json"
DEFAULT_TEST_TEMPLATE = "data/2-3_regulated_verse/couplets_{meter}_test.json"
DEFAULT_OUTPUT = "results/gpt_memorization/memorization_recall.json"


def load_couplets(path: str) -> list[dict]:
    """Load a couplet JSON list."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path} is not a list of couplets")
    return items


def subsample(items: list[dict], max_samples: int | None, seed: int) -> list[dict]:
    """Take up to max_samples records, balanced by dynasty and label."""
    if max_samples is None or max_samples >= len(items):
        return items

    rng = random.Random(seed)
    buckets = defaultdict(list)
    for item in items:
        buckets[(item.get("dynasty", "?"), item.get("label", "?"))].append(item)
    keys = list(buckets)
    rng.shuffle(keys)
    base = max_samples // len(keys)
    remainder = max_samples % len(keys)
    targets = {key: base for key in keys}
    for key in keys[:remainder]:
        targets[key] += 1

    selected = []
    selected_ids = set()
    for key, target in targets.items():
        pool = buckets[key]
        rng.shuffle(pool)
        chosen = pool[:target]
        selected.extend(chosen)
        selected_ids.update(id(item) for item in chosen)

    if len(selected) < max_samples:
        leftover = [item for item in items if id(item) not in selected_ids]
        rng.shuffle(leftover)
        selected.extend(leftover[: max_samples - len(selected)])
    rng.shuffle(selected)
    return selected[:max_samples]


def poem_split_status(train: list[dict], test: list[dict]) -> dict:
    """Verify that train and test do not share a poem, couplet, or line."""
    if not train or not test or "poem_id" not in train[0] or "poem_id" not in test[0]:
        raise SystemExit(
            "Train/test records have no poem_id. Rebuild them with "
            "scripts/prepare_data_io.py before measuring memorization."
        )

    train_poems = {item["poem_id"] for item in train}
    test_poems = {item["poem_id"] for item in test}
    train_couplets = {item["couplet"] for item in train}
    test_couplets = {item["couplet"] for item in test}

    def lines_of(items: list[dict]) -> set[str]:
        found = set()
        for item in items:
            line1, line2 = split_couplet(item["couplet"])
            if line1 is not None and line2 is not None:
                found.update((line1, line2))
        return found

    shared_poems = train_poems & test_poems
    shared_couplets = train_couplets & test_couplets
    shared_lines = lines_of(train) & lines_of(test)
    if shared_poems or shared_couplets or shared_lines:
        raise SystemExit(
            "Refusing to score a split that leaks training material into test: "
            f"{len(shared_poems)} shared poems, {len(shared_couplets)} shared "
            f"couplets, {len(shared_lines)} shared lines."
        )
    return {
        "split_unit": "poem",
        "train_poems": len(train_poems),
        "test_poems": len(test_poems),
        "shared_poems": 0,
        "shared_couplets": 0,
        "shared_lines": 0,
    }


def prompt_for(item: dict, meter: int, condition: str, line1: str) -> str:
    """Return the first-line prompt for one condition."""
    if condition == "matched":
        if item["meter"] != meter:
            raise ValueError(f"Couplet meter {item['meter']} does not match {meter}")
        return item["label"] + line1 + "，"
    if condition == "none":
        return line1 + "，"
    raise ValueError(f"Unsupported condition: {condition}")


def continuation_text(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    """Decode generated ids and drop whitespace."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return "".join(text.split())


@torch.no_grad()
def greedy_continuations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    n_chars: int,
    device: torch.device,
) -> list[str]:
    """Greedy-decode n_chars new characters for each prompt.

    Decoding is one character at a time so a pad token equal to the end-of-text
    token cannot stop the line early. Prompts are left-padded, which keeps the
    last position aligned with the last real character.
    """
    if not prompts:
        return []

    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenizer.padding_side = previous_side
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generated = []

    for _ in range(n_chars):
        next_id = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].argmax(dim=-1)
        generated.append(next_id)
        input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_id)[:, None]],
            dim=1,
        )

    token_rows = torch.stack(generated, dim=1)
    return [
        continuation_text(tokenizer, token_rows[row].tolist())
        for row in range(token_rows.shape[0])
    ]


@torch.no_grad()
def mean_second_line_loss(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    targets: list[str],
    device: torch.device,
) -> list[float]:
    """Return mean teacher-forced NLL of each true second line."""
    encoded_examples = []
    max_length = 0
    for prompt, target in zip(prompts, targets):
        prefix_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = prefix_ids + target_ids
        encoded_examples.append((input_ids, len(prefix_ids), len(target_ids)))
        max_length = max(max_length, len(input_ids))

    pad_token_id = tokenizer.pad_token_id
    input_ids_tensor = torch.full(
        (len(encoded_examples), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids_tensor)
    for row, (ids, _prefix_len, _target_len) in enumerate(encoded_examples):
        if not ids:
            continue
        length = len(ids)
        input_ids_tensor[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1

    logits = model(input_ids=input_ids_tensor, attention_mask=attention_mask).logits
    losses = []
    for row, (_ids, prefix_len, target_len) in enumerate(encoded_examples):
        if prefix_len == 0 or target_len == 0:
            losses.append(float("nan"))
            continue
        token_losses = []
        for offset in range(target_len):
            target_pos = prefix_len + offset
            target_id = int(input_ids_tensor[row, target_pos].item())
            log_probs = F.log_softmax(logits[row, target_pos - 1, :], dim=-1)
            token_losses.append(float(-log_probs[target_id].item()))
        losses.append(sum(token_losses) / len(token_losses))
    return losses


def char_accuracy(prediction: str, gold: str) -> float:
    """Return the fraction of gold characters matched in order."""
    if not gold:
        return float("nan")
    hits = sum(
        1
        for index, character in enumerate(gold)
        if index < len(prediction) and prediction[index] == character
    )
    return hits / len(gold)


def accumulate(bucket: dict, item: dict, exact: bool, accuracy: float, loss: float) -> None:
    """Add one scored couplet to a result bucket."""
    bucket["n"] += 1
    bucket["exact"] += int(exact)
    bucket["char_correct"] += accuracy
    bucket["loss_sum"] += loss
    label = item.get("label", "?")
    dynasty = item.get("dynasty", "?")
    bucket["by_label"][label]["n"] += 1
    bucket["by_label"][label]["exact"] += int(exact)
    bucket["by_label"][label]["loss_sum"] += loss
    bucket["by_dynasty"][dynasty]["n"] += 1
    bucket["by_dynasty"][dynasty]["exact"] += int(exact)
    bucket["by_dynasty"][dynasty]["loss_sum"] += loss


def new_bucket() -> dict:
    """Return an empty accumulation bucket."""
    return {
        "n": 0,
        "exact": 0,
        "char_correct": 0.0,
        "loss_sum": 0.0,
        "by_label": defaultdict(lambda: {"n": 0, "exact": 0, "loss_sum": 0.0}),
        "by_dynasty": defaultdict(lambda: {"n": 0, "exact": 0, "loss_sum": 0.0}),
    }


def rate(numerator: float, denominator: float) -> float | None:
    """Return a ratio, or None when the denominator is zero."""
    if not denominator:
        return None
    return numerator / denominator


def finish_bucket(bucket: dict) -> dict:
    """Convert raw counts into the reported rates."""
    def block(raw: dict, include_char: bool) -> dict:
        finished = {
            "n": raw["n"],
            "exact_match_rate": rate(raw["exact"], raw["n"]),
            "mean_loss": rate(raw["loss_sum"], raw["n"]),
        }
        if include_char:
            finished["mean_char_accuracy"] = rate(raw["char_correct"], raw["n"])
        return finished

    by_label = {
        label: block(raw, include_char=False)
        for label, raw in sorted(bucket["by_label"].items())
    }
    by_dynasty = {
        dynasty: block(raw, include_char=False)
        for dynasty, raw in sorted(bucket["by_dynasty"].items())
    }
    finished = block(bucket, include_char=True)
    finished["by_label"] = by_label
    finished["by_dynasty"] = by_dynasty
    return finished


def score_split(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    converter: opencc.OpenCC | None,
    items: list[dict],
    meter: int,
    conditions: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict]:
    """Score every condition on one split."""
    buckets = {condition: new_bucket() for condition in conditions}
    prepared = []
    for item in items:
        couplet = item["couplet"] if converter is None else converter.convert(item["couplet"])
        line1, line2 = split_couplet(couplet)
        if line1 is None or line2 is None or len(line2) != meter:
            continue
        prepared.append((item, line1, line2))

    for condition in conditions:
        for start in tqdm(
            range(0, len(prepared), batch_size),
            desc=f"{METER_NAMES[meter]} {condition}",
        ):
            batch = prepared[start : start + batch_size]
            prompts = [
                prompt_for(item, meter, condition, line1)
                for item, line1, _line2 in batch
            ]
            golds = [line2 for _item, _line1, line2 in batch]
            predictions = greedy_continuations(model, tokenizer, prompts, meter, device)
            losses = mean_second_line_loss(model, tokenizer, prompts, golds, device)
            for (item, _line1, gold), prediction, loss in zip(batch, predictions, losses):
                prediction = prediction[:meter]
                accumulate(
                    buckets[condition],
                    item,
                    prediction == gold,
                    char_accuracy(prediction, gold),
                    loss,
                )
    return {condition: finish_bucket(bucket) for condition, bucket in buckets.items()}


def print_table(results: dict) -> None:
    """Print the train/test exact-match and loss comparison."""
    print("\nVerbatim second-line recall")
    print(
        f"{'meter':<6} {'split':<6} {'prompt':<8} {'n':>7} "
        f"{'exact':>8} {'char':>8} {'loss':>8} {'inner':>8} {'outer':>8}"
    )
    for meter_name, splits in results["meters"].items():
        for split_name, conditions in splits.items():
            if split_name == "split_check":
                continue
            for condition, row in conditions.items():
                inner = row["by_label"].get("i", {}).get("exact_match_rate")
                outer = row["by_label"].get("o", {}).get("exact_match_rate")

                def fmt(value: float | None) -> str:
                    return "n/a" if value is None else f"{value:.4f}"

                print(
                    f"{meter_name:<6} {split_name:<6} {condition:<8} {row['n']:>7} "
                    f"{fmt(row['exact_match_rate']):>8} "
                    f"{fmt(row['mean_char_accuracy']):>8} "
                    f"{fmt(row['mean_loss']):>8} "
                    f"{fmt(inner):>8} {fmt(outer):>8}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Measure verbatim second-line recall on poem-held-out couplets"
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Defaults to --model-path",
    )
    parser.add_argument("--train-template", default=DEFAULT_TRAIN_TEMPLATE)
    parser.add_argument("--test-template", default=DEFAULT_TEST_TEMPLATE)
    parser.add_argument("--meters", type=int, nargs="+", choices=METERS, default=list(METERS))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=["train", "test"],
        help="Which splits to score (default: both, so memorization can be compared)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="matched uses the fine-tuning prefix; none uses the first line only",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Couplets scored per split per meter, balanced by dynasty and label (default: 10000)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip simplified to traditional conversion",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stored in the output so base and fine-tuned runs can be told apart",
    )
    args = parser.parse_args()

    device = select_device()
    print(f"Using device: {device}")
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_padding_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    converter = None if args.no_convert else opencc.OpenCC("s2t")
    results = {
        "metric": (
            "Exact second-line recall under greedy decoding from the first line, "
            "plus teacher-forced mean loss of the true second line."
        ),
        "model_path": args.model_path,
        "run_name": args.run_name,
        "max_samples": args.max_samples,
        "meters": {},
    }

    for meter in args.meters:
        meter_name = METER_NAMES[meter]
        train = load_couplets(args.train_template.format(meter=meter_name))
        test = load_couplets(args.test_template.format(meter=meter_name))
        split_check = poem_split_status(train, test)
        print(
            f"{meter_name}: poem-disjoint confirmed "
            f"({split_check['train_poems']} train poems, {split_check['test_poems']} test poems)"
        )
        pools = {"train": train, "test": test}
        meter_results = {"split_check": split_check}
        for split_name in args.splits:
            scored = subsample(pools[split_name], args.max_samples, args.seed)
            print(f"Scoring {len(scored)} {meter_name} {split_name} couplets")
            meter_results[split_name] = score_split(
                model,
                tokenizer,
                converter,
                scored,
                meter,
                tuple(args.conditions),
                args.batch_size,
                device,
            )
        results["meters"][meter_name] = meter_results

    output_path = args.output
    if args.run_name and output_path == DEFAULT_OUTPUT:
        output_path = f"results/gpt_memorization/memorization_recall_{args.run_name}.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print_table(results)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
