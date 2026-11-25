# Cipher fine-tuning of GujiGPT

Does GujiGPT learn the entropy signature of parallelism from couplets whose sense it cannot use?

- `couplets.json`: for each meter (`penta`, `hepta`) and corpus (`cipher`, `real`), the 100,000 training couplets and the 1,000 held-out inner and outer couplets, in traditional characters. Each `cipher` couplet is the `real` one with every character replaced, throughout the corpus and its test sets, by another character of similar frequency. The cipher was made with `scripts/make_cipher_couplets.py --level char`.
- `finetune_cipher.py`: fine-tunes GujiGPT with the i/o control tokens and scores the model before training (epoch 0) and after every epoch. At every position of the second line it records next-character entropy and surprisal, for inner and outer test couplets under both prompts. The scoring is identical to that of the article: on the article's fine-tuned model it reproduces the saved values to within 0.0001.

## Run (CUDA)

```bash
pip install torch transformers numpy
nohup python finetune_cipher.py --bf16 > run.log 2>&1 &          # cipher, both meters, 4 epochs, lr 1e-4
nohup python finetune_cipher.py --bf16 --corpora cipher real > run.log 2>&1 &   # with the real-text reference
```

- Smoke test: `python finetune_cipher.py --limit-train 64 --limit-test 8 --epochs 1`.
- `--batch-size` (default 8, as in the article) can be raised for speed, but this changes the training schedule.
- `--save-model` also keeps the final models, at about 450 MB each. `--save-every-epoch` keeps the model after every epoch (under `--model-dir`, default the output directory).

Upload the `results/` directory. It holds `{meter}_{corpus}_epoch{k}.json` (per-couplet values) and `summary.md`/`summary.json` (mean entropy and surprisal by position for every epoch, prompt and test set).

## What to look for

If the signature is learned without sense, the cipher model on enciphered inner couplets under the I prompt should develop:

- dips in entropy at positions 2 and 4, as in real couplets;
- lower entropy than on enciphered outer couplets under the O prompt.
