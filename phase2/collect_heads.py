"""Collect per-head attention outputs (input to o_proj) for ITI-style probing."""
import torch
from phase2.loaders import get_transformer_layers


def get_head_config(model) -> tuple[int, int]:
    # PeftModel delegates .config to the base model, but fall back explicitly
    try:
        cfg = model.config
    except AttributeError:
        cfg = model.base_model.model.config
    num_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // num_heads
    return num_heads, head_dim


def collect_head_activations(
    model,
    tokenizer,
    D_steer: list,
    device: str,
    boundary_idx_fn,
    prompt_fn=None,
    N_rollouts: int = 10,
    min_samples: int = 30,
) -> tuple[dict, dict]:
    """
    Collect per-head attention outputs (o_proj inputs) at boundary positions.

    For each question, generates N_rollouts with temperature sampling to get
    both correct (H+) and incorrect (H-) trajectories, then runs a teacher-forced
    forward pass on the full sequence to extract head activations at boundary_idx.

    Returns:
        H_pos: dict[(layer_idx, head_idx) -> Tensor (n_pos, head_dim)]
        H_neg: dict[(layer_idx, head_idx) -> Tensor (n_neg, head_dim)]
    """
    from phase1.inference import extract_answer, normalize_answer

    layers = get_transformer_layers(model)
    num_heads, head_dim = get_head_config(model)
    n_layers = len(layers)

    raw_pos = {(l, h): [] for l in range(n_layers) for h in range(num_heads)}
    raw_neg = {(l, h): [] for l in range(n_layers) for h in range(num_heads)}

    head_cache: dict = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # input[0]: [batch, seq_len, num_heads * head_dim]
            head_cache[layer_idx] = input[0].detach().cpu()
        return hook

    handles = [
        layers[l].self_attn.o_proj.register_forward_hook(make_hook(l))
        for l in range(n_layers)
    ]

    model.eval()
    n_questions = len(D_steer)

    try:
        for q_idx, item in enumerate(D_steer):
            gold = item['answer'].split('####')[1].strip()

            if prompt_fn is not None:
                prompt = prompt_fn(item)
            else:
                prompt = f"Question: {item['question']}\n\nAnswer:"

            enc = tokenizer(prompt, return_tensors='pt').to(device)

            for _ in range(N_rollouts):
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=0.8,
                        max_new_tokens=128,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                try:
                    b_idx = boundary_idx_fn(out, tokenizer)
                except Exception:
                    b_idx = max(0, enc['input_ids'].shape[1] - 1)

                if b_idx >= out.shape[1]:
                    continue

                gen_text = tokenizer.decode(
                    out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
                )
                pred = extract_answer(gen_text)
                is_correct = (
                    pred is not None
                    and normalize_answer(pred) == normalize_answer(gold)
                )

                # Teacher-forced forward on full sequence to get head activations
                head_cache.clear()
                with torch.no_grad():
                    model(input_ids=out)

                for l in range(n_layers):
                    if l not in head_cache:
                        continue
                    act = head_cache[l]  # [1, full_seq_len, num_heads * head_dim]
                    if b_idx >= act.shape[1]:
                        continue

                    boundary = act[0, b_idx, :].float().view(num_heads, head_dim)

                    for h_idx in range(num_heads):
                        h_act = boundary[h_idx].clone()
                        if is_correct:
                            raw_pos[(l, h_idx)].append(h_act)
                        else:
                            raw_neg[(l, h_idx)].append(h_act)

            if (q_idx + 1) % 20 == 0:
                n_valid = sum(
                    1 for k in raw_pos
                    if len(raw_pos[k]) >= min_samples and len(raw_neg[k]) >= min_samples
                )
                print(f"  [{q_idx+1}/{n_questions}] (layer,head) pairs with >= {min_samples} samples: {n_valid}")

    finally:
        for h in handles:
            h.remove()

    H_pos, H_neg = {}, {}
    for key in raw_pos:
        if len(raw_pos[key]) >= min_samples and len(raw_neg[key]) >= min_samples:
            H_pos[key] = torch.stack(raw_pos[key])
            H_neg[key] = torch.stack(raw_neg[key])

    n_valid = len(H_pos)
    print(f"\nHead activation collection complete.")
    print(f"  Valid (layer, head) pairs: {n_valid} / {n_layers * num_heads}")
    if H_pos:
        sample_key = next(iter(H_pos))
        print(f"  Sample pair {sample_key}: H+={H_pos[sample_key].shape[0]}  H-={H_neg[sample_key].shape[0]}")

    return H_pos, H_neg
