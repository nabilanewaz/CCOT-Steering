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

    @property
    def config(self):
        return self.base_causallm.config

    @property
    def generation_config(self):
        return self.base_causallm.generation_config

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
        attention_mask=None,
        labels=None,
        steering_vector=None,
        alpha=0.0,
        gamma=1.0,
        steering_mode="vector",
        collect_steering_stats=False,
        detach_latents=False,
        use_kv_cache=True,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
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
                latent_value = hidden_states.detach() if detach_latents else hidden_states
                latent_sequence.append(latent_value)
                inputs_embeds = inputs_embeds.clone()
                filling_indices = [(i, l[pass_idx]) for i, l in enumerate(latent_lists) if len(l) > pass_idx]
                for batch_idx, token_idx in filling_indices:
                    inputs_embeds[batch_idx, token_idx] = latent_value[batch_idx, -1]

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
                latent_value = hidden_states.detach() if detach_latents else hidden_states
                latent_sequence.append(latent_value)
                kv_cache = outputs.past_key_values
                inputs_embeds = inputs_embeds.clone()
                filling_indices = [(i, l[pass_idx]) for i, l in enumerate(latent_lists) if len(l) > pass_idx]
                for batch_idx, token_idx in filling_indices:
                    inputs_embeds[batch_idx, token_idx] = latent_value[batch_idx, -1]

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

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=128,
        do_sample=False,
        temperature=1.0,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("Coconut.generate requires input_ids")
        sampling_temperature = temperature if do_sample else 0.0
        output_ids, _, _ = self.generate_with_latents(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=sampling_temperature,
            steering_vector=kwargs.pop("steering_vector", None),
            alpha=kwargs.pop("alpha", 0.0),
            gamma=kwargs.pop("gamma", 1.0),
            steering_mode=kwargs.pop("steering_mode", "vector"),
        )
        return output_ids

    def generate_with_latents(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=128,
        temperature=0.0,
        steering_vector=None,
        alpha=0.0,
        gamma=1.0,
        steering_mode="vector",
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("Coconut generation currently requires batch_size=1")
        self.eval()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                steering_vector=steering_vector,
                alpha=alpha,
                gamma=gamma,
                steering_mode=steering_mode,
                use_kv_cache=False,
            )

        latent_steps = [h[:, -1, :].detach().cpu() for h in outputs.latent_sequence]
        self.last_generation_latents = latent_steps
        mean_latent = torch.mean(torch.stack(latent_steps), dim=0) if latent_steps else None
        scores = [
            F.cosine_similarity(h_prev, h_next, dim=-1).item()
            for h_prev, h_next in zip(latent_steps[:-1], latent_steps[1:])
        ]
        self.last_trajectory_faithfulness = float(np.mean(scores)) if scores else 0.0

        generated_ids = input_ids.clone()
        inputs_embeds = outputs.inputs_embeds
        logits = outputs.logits
        for _ in range(max_new_tokens):
            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if self.eos_token_id is not None and next_token.item() == self.eos_token_id:
                break

            next_embed = self.embedding(next_token)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            with torch.no_grad():
                next_outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=torch.ones(
                        inputs_embeds.shape[:2], dtype=torch.long, device=input_ids.device
                    ),
                    use_cache=False,
                )
            logits = next_outputs.logits

        return generated_ids, mean_latent, self.last_trajectory_faithfulness
