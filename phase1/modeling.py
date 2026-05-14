from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "latent_sequence"])


class Coconut(torch.nn.Module):
    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id, eos_token_id):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.embedding = self.base_causallm.get_input_embeddings()
        self.last_steering_stats = []
        self.last_generation_latents = []
        self.last_trajectory_faithfulness = 0.0

    def _process_kv(self, kv_cache, keep_len):
        if kv_cache is None:
            return None
        if not hasattr(kv_cache, "key_cache") and not isinstance(kv_cache, (list, tuple)):
            return kv_cache

        new_cache = DynamicCache()
        num_layers = len(kv_cache.key_cache) if hasattr(kv_cache, "key_cache") else len(kv_cache)
        for i in range(num_layers):
            if hasattr(kv_cache, "key_cache"):
                k, v = kv_cache.key_cache[i], kv_cache.value_cache[i]
            else:
                k, v = kv_cache[i]
            new_cache.update(k[..., :keep_len, :], v[..., :keep_len, :], layer_idx=i)
        return new_cache

    def _alpha_tensor(self, alpha, device, dtype):
        if torch.is_tensor(alpha):
            return alpha.to(device=device, dtype=dtype)
        return torch.tensor(float(alpha), device=device, dtype=dtype)

    def _prepare_direction(self, steering_vector, hidden_states, mode):
        vector = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if mode == "subspace":
            if vector.dim() != 2:
                raise ValueError("Subspace steering expects a [hidden, k] matrix.")
            return vector
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        return F.normalize(vector, p=2, dim=-1)

    def _apply_steering(self, hidden_states, steering_vector, alpha, gamma, pass_idx, steering_mode, collect_steering_stats):
        if steering_vector is None:
            return hidden_states

        h_t = hidden_states[:, -1, :]
        d_model = h_t.shape[-1]
        alpha_t = self._alpha_tensor(alpha, h_t.device, h_t.dtype) * (gamma ** pass_idx)
        sigma_t = h_t.norm(dim=-1, keepdim=True) / (d_model ** 0.5)
        direction = self._prepare_direction(steering_vector, hidden_states, steering_mode)

        if steering_mode == "subspace":
            h_unit = F.normalize(h_t, p=2, dim=-1)
            projection = h_unit @ direction @ direction.T
            intervention = alpha_t * sigma_t * projection
        else:
            intervention = alpha_t * sigma_t * direction

        steered_h = h_t + intervention
        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] = steered_h

        if collect_steering_stats:
            self.last_steering_stats.append(
                {
                    "h_before": h_t,
                    "h_after": steered_h,
                    "intervention": intervention,
                    "direction": direction,
                }
            )
        return hidden_states

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        steering_vector=None,
        alpha=0.0,
        gamma=1.0,
        steering_mode="vector",
        collect_steering_stats=False,
        detach_latents=True,
        use_kv_cache=True,
    ):
        latent_sequence = []
        self.last_steering_stats = []
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        inputs_embeds = self.embedding(input_ids)
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
        next_compute_range = (0, input_ids.shape[1] if max_n_latents == 0 else latent_indices[:, 1].min().item())

        if not use_kv_cache:
            for pass_idx in range(max_n_latents):
                end = next_compute_range[1]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    past_key_values=None,
                    output_hidden_states=True,
                    use_cache=False,
                )
                next_end = input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
                next_compute_range = (next_compute_range[1], next_end)
                hidden_states = self._apply_steering(
                    outputs.hidden_states[-1],
                    steering_vector,
                    alpha,
                    gamma,
                    pass_idx,
                    steering_mode,
                    collect_steering_stats,
                )
                latent_sequence.append(hidden_states.detach() if detach_latents else hidden_states)
                inputs_embeds = inputs_embeds.clone()
                filling_indices = [(i, l[pass_idx]) for i, l in enumerate(latent_lists) if len(l) > pass_idx]
                for batch_idx, token_idx in filling_indices:
                    inputs_embeds[batch_idx, token_idx] = hidden_states[batch_idx, -1]

            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_hidden_states=False,
                use_cache=False,
            )
            logits = outputs.logits
        else:
            logits = []
            kv_cache = None
            for pass_idx in range(max_n_latents):
                curr_cache = self._process_kv(kv_cache, next_compute_range[0])
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1]],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=curr_cache,
                    output_hidden_states=True,
                    use_cache=True,
                )
                logits.append(outputs.logits)
                next_end = input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
                next_compute_range = (next_compute_range[1], next_end)
                hidden_states = self._apply_steering(
                    outputs.hidden_states[-1],
                    steering_vector,
                    alpha,
                    gamma,
                    pass_idx,
                    steering_mode,
                    collect_steering_stats,
                )
                latent_sequence.append(hidden_states.detach() if detach_latents else hidden_states)
                kv_cache = outputs.past_key_values
                inputs_embeds = inputs_embeds.clone()
                filling_indices = [(i, l[pass_idx]) for i, l in enumerate(latent_lists) if len(l) > pass_idx]
                for batch_idx, token_idx in filling_indices:
                    inputs_embeds[batch_idx, token_idx] = hidden_states[batch_idx, -1]

            final_cache = self._process_kv(kv_cache, next_compute_range[0])
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1]],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                past_key_values=final_cache,
                use_cache=True,
            )
            logits.append(outputs.logits)
            logits = torch.cat(logits, dim=-2)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32),
                shift_labels.view(-1),
            )
        return Outputs(loss, inputs_embeds, logits, latent_sequence)

    def generate_with_latents(
        self,
        input_ids,
        max_new_tokens=128,
        temperature=0.0,
        steering_vector=None,
        alpha=0.0,
        gamma=1.0,
        steering_mode="vector",
    ):
        self.eval()
        tokens = input_ids.tolist()[0]
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                steering_vector=steering_vector,
                alpha=alpha,
                gamma=gamma,
                steering_mode=steering_mode,
            )
        latent_steps = [h[:, -1, :].detach().cpu() for h in outputs.latent_sequence]
        self.last_generation_latents = latent_steps
        mean_latent = torch.mean(torch.stack(latent_steps), dim=0) if latent_steps else None
        scores = []
        if len(latent_steps) > 1:
            for h_prev, h_next in zip(latent_steps[:-1], latent_steps[1:]):
                scores.append(F.cosine_similarity(h_prev, h_next, dim=-1).item())
        self.last_trajectory_faithfulness = float(np.mean(scores)) if scores else 0.0

        if temperature > 0:
            scaled = outputs.logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        tokens.append(next_token)
        curr_input_ids = torch.tensor([tokens], device=input_ids.device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.base_causallm(input_ids=curr_input_ids)
            if temperature > 0:
                scaled = out.logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            curr_input_ids = torch.tensor([tokens], device=input_ids.device)
        return torch.tensor([tokens]), mean_latent, self.last_trajectory_faithfulness
