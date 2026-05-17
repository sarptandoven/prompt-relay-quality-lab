import logging
import types
import torch
import comfy.ldm.modules.attention


log = logging.getLogger(__name__)


def _masked_attention(q, k, v, heads, mask, transformer_options={}, **kwargs):
    # Bypass wrap_attn (sage/etc may ignore masks) by calling attention_pytorch directly.
    return comfy.ldm.modules.attention.attention_pytorch(
        q, k, v, heads, mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options,
        **kwargs,
    )


def _wan_t2v_forward(self, mask_fn, x, context, transformer_options={}, **kwargs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )
    return self.o(x)


def _wan_i2v_forward(self, mask_fn, x, context, context_img_len, transformer_options={}, **kwargs):
    context_img = context[:, :context_img_len]
    context_text = context[:, context_img_len:]

    q = self.norm_q(self.q(x))

    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = comfy.ldm.modules.attention.optimized_attention(
        q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options,
    )

    k = self.norm_k(self.k(context_text))
    v = self.v(context_text)

    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )

    return self.o(x + img_x)


def _ltx_forward(self, mask_fn, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    from comfy.ldm.lightricks.model import apply_rotary_emb

    is_self_attn = context is None
    context = x if is_self_attn else context

    q = self.q_norm(self.to_q(x))
    k = self.k_norm(self.to_k(context))
    v = self.to_v(context)

    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

    if not is_self_attn:
        temporal_mask = mask_fn(q, k, transformer_options)
        if temporal_mask is not None:
            mask = temporal_mask if mask is None else mask + temporal_mask

    if mask is None:
        out = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, self.heads, attn_precision=self.attn_precision,
            transformer_options=transformer_options,
        )
    else:
        out = _masked_attention(q, k, v, self.heads, mask=mask,
                                attn_precision=self.attn_precision,
                                transformer_options=transformer_options)

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)


class _CrossAttnPatch:
    """Descriptor that binds (impl, mask_fn) as a method onto a cross-attn module."""

    def __init__(self, impl, mask_fn):
        self.impl = impl
        self.mask_fn = mask_fn

    def __get__(self, obj, objtype=None):
        impl, mask_fn = self.impl, self.mask_fn

        def wrapped(self_module, *args, **kwargs):
            return impl(self_module, mask_fn, *args, **kwargs)

        return types.MethodType(wrapped, obj)


def detect_model_type(model):
    """Return (arch, patch_size, temporal_stride) for latent geometry.

    temporal_stride is the VAE's pixel→latent temporal compression factor,
    used to convert user-facing pixel frame counts to latent frames.
    """
    diff_model = model.model.diffusion_model

    if hasattr(diff_model, "patch_size") and not hasattr(diff_model, "patchifier"):
        return "wan", tuple(diff_model.patch_size), 4

    if hasattr(diff_model, "patchifier"):
        return "ltx", (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(
        f"Unsupported model type: {type(diff_model).__name__}. "
        f"Currently supports Wan and LTX models."
    )


def _describe_patch(value):
    patch_type = type(value).__name__
    try:
        text = repr(value)
    except Exception:
        text = "<unrepresentable>"
    if len(text) > 160:
        text = text[:157] + "..."
    return f"{patch_type}: {text}"


def _find_conflicting_object_patches(model_clone, key):
    """Return object patches that target the same attention module/forward path.

    Comfy object patches are path-based. Other nodes may patch the exact
    ``*.forward`` method, the parent attention module, or a child under that
    module. Any of those would make Prompt Relay's LTX/Wan cross-attention mask
    order ambiguous, so fail early with a specific report instead of silently
    losing masks in SageAttention/preview/custom-node stacks.
    """
    object_patches = getattr(model_clone, "object_patches", {}) or {}
    target_module = key.rsplit(".", 1)[0]
    conflicts = []
    for existing_key, value in object_patches.items():
        if (
            existing_key == key
            or existing_key == target_module
            or existing_key.startswith(key + ".")
            or key.startswith(existing_key + ".")
            or existing_key.startswith(target_module + ".")
            or target_module.startswith(existing_key + ".")
        ):
            conflicts.append((existing_key, value))
    return conflicts


def _check_unpatched(model_clone, key):
    conflicts = _find_conflicting_object_patches(model_clone, key)
    if not conflicts:
        return

    details = "; ".join(
        f"{existing_key} ({_describe_patch(value)})"
        for existing_key, value in conflicts[:5]
    )
    if len(conflicts) > 5:
        details += f"; ... {len(conflicts) - 5} more"

    raise RuntimeError(
        f"PromptRelay: cannot patch cross-attention forward at '{key}' because "
        f"object_patches already contains conflicting patch(es): {details}. "
        "Prompt Relay must own these attention forward methods so its temporal "
        "mask reaches the backend. Remove/reorder conflicting attention patch "
        "nodes such as SageAttention, preview, NAG, or other custom attention "
        "patchers for this model path."
    )


def _ensure_patches_installed(arch, patched_keys):
    if patched_keys:
        return
    raise RuntimeError(
        f"PromptRelay: detected a supported {arch} model but did not find any "
        "cross-attention modules to patch. This model variant is not covered by "
        "the current Prompt Relay patcher, so running would silently produce "
        "unrelayed output. Update Prompt Relay for this model's attention module "
        "names before using this node."
    )


def _log_patch_install(arch, patched_keys, mask_fn):
    _ensure_patches_installed(arch, patched_keys)
    diagnostics = getattr(mask_fn, "prompt_relay_diagnostics", None)
    log.info(
        "[PromptRelay] Installed %d %s attention patches: %s%s",
        len(patched_keys),
        arch,
        ", ".join(patched_keys[:6]) + (" ..." if len(patched_keys) > 6 else ""),
        " | diagnostics enabled" if diagnostics is not None else "",
    )


def apply_patches(model_clone, arch, mask_fn):
    diffusion_model = model_clone.get_model_object("diffusion_model")
    patched_keys = []

    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention
        for idx, block in enumerate(diffusion_model.blocks):
            key = f"diffusion_model.blocks.{idx}.cross_attn.forward"
            _check_unpatched(model_clone, key)
            cross_attn = block.cross_attn
            impl = _wan_i2v_forward if isinstance(cross_attn, WanI2VCrossAttention) else _wan_t2v_forward
            model_clone.add_object_patch(key, _CrossAttnPatch(impl, mask_fn).__get__(cross_attn, cross_attn.__class__))
            patched_keys.append(key)
        _log_patch_install(arch, patched_keys, mask_fn)
        return patched_keys

    if arch == "ltx":
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
                _check_unpatched(model_clone, key)
                model_clone.add_object_patch(key, _CrossAttnPatch(_ltx_forward, mask_fn).__get__(module, module.__class__))
                patched_keys.append(key)
        _log_patch_install(arch, patched_keys, mask_fn)
        return patched_keys

    raise ValueError(f"Unknown model arch: {arch}")
