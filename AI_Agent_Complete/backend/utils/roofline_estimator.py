"""Utility to estimate roofline metrics before running profiling.

This replicates the formulas from the LLM-Viewer project in a simplified form so
we can predict arithmetic intensity, achievable performance, and expected run
time for the prefill and decode stages given a model configuration and
inference parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# The OPS are based on dense tensor core throughput; INT8 throughput is only
# selected when all tensors use INT8 precision.
HARDWARE_PARAMS: Dict[str, Dict[str, float]] = {
    "nvidia_V100": {"bandwidth": 900e9, "FP16": 112e12, "INT8": 62e12, "onchip_buffer": 20480e3},
    "nvidia_A6000": {"bandwidth": 768e9, "FP16": 154.8e12, "INT8": 309.7e12, "onchip_buffer": 21504e3},
    "nvidia_A6000_Ada": {"bandwidth": 960e9, "FP16": 364.2e12, "INT8": 728.5e12, "onchip_buffer": 36352e3},
    "nvidia_A100": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A100_40G": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A100_80G": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A800_80G_SXM": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A40": {"bandwidth": 696e9, "FP16": 149.7e12, "INT8": 299.3e12, "onchip_buffer": 21504e3},
    "nvidia_H100": {"bandwidth": 3072e9, "FP16": 1979e12 / 2, "INT8": 3958e12 / 2, "onchip_buffer": 33792e3},
    "nvidia_H100_SXM": {"bandwidth": 3072e9, "FP16": 1979e12 / 2, "INT8": 3958e12 / 2, "onchip_buffer": 33792e3},
    "nvidia_H800_SXM5_80G": {
        "bandwidth": 3350e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
    },
    "nvidia_H100_PCIe": {"bandwidth": 2048e9, "FP16": 1513e12 / 2, "INT8": 3026e12 / 2, "onchip_buffer": 29184e3},
    "nvidia_L40": {"bandwidth": 864e9, "FP16": 181e12, "INT8": 362e12, "onchip_buffer": 36352e3},
    "intel_13900k": {"bandwidth": 89.6e9, "FP16": 8 * 5.4e9 * (512 / 16), "onchip_buffer": 36e6},
}


def _ensure_hardware(hardware: str) -> Dict[str, float]:
    if hardware not in HARDWARE_PARAMS:
        raise ValueError(f"Unsupported hardware '{hardware}', please extend HARDWARE_PARAMS.")
    return HARDWARE_PARAMS[hardware]


@dataclass
class ModelSpec:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _read_model_config(model_path: Path) -> Dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find config.json under {model_path}.")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_int(config: Dict[str, Any], *keys: str, default: Optional[int] = None) -> Optional[int]:
    for key in keys:
        if key in config and isinstance(config[key], int):
            return int(config[key])
    return default


def _infer_model_spec(config: Dict[str, Any]) -> ModelSpec:
    hidden_size = _get_int(config, "hidden_size", "n_embd", "dim")
    num_layers = _get_int(config, "num_hidden_layers", "n_layer", "num_layers")
    num_attention_heads = _get_int(config, "num_attention_heads", "n_head", "heads")
    intermediate_size = _get_int(config, "intermediate_size", "ffn_hidden_size", "mlp_hidden_size")
    vocab_size = _get_int(config, "vocab_size", "n_vocab")
    num_key_value_heads = _get_int(config, "num_key_value_heads", "n_kv_head", "kv_heads", default=num_attention_heads)

    if hidden_size is None or num_layers is None or num_attention_heads is None or intermediate_size is None or vocab_size is None:
        raise ValueError("Model config missing essential fields for roofline estimation.")

    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads for head_dim calculation.")

    return ModelSpec(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
    )


def _bytes_by_precision(bits: int) -> float:
    return bits / 8.0


def _accumulate(stage: Dict[str, float], **kwargs: float) -> None:
    for key, value in kwargs.items():
        stage[key] = stage.get(key, 0.0) + float(value)


def _roofline_eval(hardware: str, precision_bits: Dict[str, int], op_bytes: Dict[str, float], total_ops: float) -> Dict[str, float]:
    params = _ensure_hardware(hardware)
    bandwidth = params["bandwidth"]
    max_ops = params["INT8"] if all(bits <= 8 for bits in precision_bits.values()) else params["FP16"]
    memory_access = sum(op_bytes.values())
    if memory_access <= 0:
        arithmetic_intensity = float("inf")
        performance = max_ops
        bound = "compute"
    else:
        arithmetic_intensity = total_ops / memory_access
        turning = max_ops / bandwidth
        if arithmetic_intensity < turning:
            performance = arithmetic_intensity * bandwidth
            bound = "memory"
        else:
            performance = max_ops
            bound = "compute"
    inference_time = total_ops / performance if performance > 0 else float("inf")
    return {
        "bandwidth": bandwidth,
        "max_ops": max_ops,
        "memory_access": memory_access,
        "arithmetic_intensity": arithmetic_intensity,
        "performance": performance,
        "bound": bound,
        "inference_time": inference_time,
    }


def _stage_dict() -> Dict[str, float]:
    return {
        "OPs": 0.0,
        "load_weight": 0.0,
        "load_act": 0.0,
        "store_act": 0.0,
        "load_kv_cache": 0.0,
        "store_kv_cache": 0.0,
    }


class RooflineEstimator:
    def __init__(
        self,
        model_path: Path,
        hardware: str,
        batch_size: int,
        prompt_len: int,
        output_len: int,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: Optional[int] = None,
        use_flashattention: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if prompt_len <= 0:
            raise ValueError("prompt_len must be positive.")
        if output_len <= 0:
            raise ValueError("output_len must be positive.")

        self.model_path = Path(model_path)
        self.hardware = hardware
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit if kv_bit is not None else a_bit
        self.use_flashattention = use_flashattention

        config = _read_model_config(self.model_path)
        self.model_spec = _infer_model_spec(config)

        self.w_byte = _bytes_by_precision(self.w_bit)
        self.a_byte = _bytes_by_precision(self.a_bit)
        self.kv_byte = _bytes_by_precision(self.kv_bit)

    def compute(self) -> Dict[str, Any]:
        decode = _stage_dict()
        prefill = _stage_dict()
        spec = self.model_spec
        bs = self.batch_size
        prompt_len = self.prompt_len
        head_dim = spec.head_dim
        kv_heads = spec.num_key_value_heads
        attn_heads = spec.num_attention_heads

        def add(stage: Dict[str, float], **kwargs: float) -> None:
            for key, value in kwargs.items():
                stage[key] = stage.get(key, 0.0) + float(value)

        def linear(stage: Dict[str, float], ic: int, oc: int, tokens: int, is_kv_proj: bool = False) -> None:
            ops = ic * oc * bs * 2 * tokens
            add(stage, OPs=ops)
            add(stage, load_weight=ic * oc * self.w_byte)
            add(stage, load_act=ic * bs * self.a_byte * tokens)
            if not is_kv_proj:
                add(stage, store_act=oc * bs * self.a_byte * tokens)
            else:
                add(stage, store_kv_cache=oc * bs * self.kv_byte * tokens)

        # Linear layers per transformer block
        q_out = spec.hidden_size
        kv_out = head_dim * kv_heads
        mlp_out = spec.intermediate_size

        # Decode stage (per generated token with context prompt_len)
        linear(decode, spec.hidden_size, q_out, 1, False)
        linear(decode, spec.hidden_size, kv_out, 1, True)
        linear(decode, spec.hidden_size, kv_out, 1, True)
        linear(decode, spec.hidden_size, spec.hidden_size, 1, False)
        linear(decode, spec.hidden_size, mlp_out, 1, False)
        linear(decode, spec.hidden_size, mlp_out, 1, False)
        linear(decode, mlp_out, spec.hidden_size, 1, False)

        # Prefill stage (prompt tokens)
        linear(prefill, spec.hidden_size, q_out, prompt_len, False)
        linear(prefill, spec.hidden_size, kv_out, prompt_len, True)
        linear(prefill, spec.hidden_size, kv_out, prompt_len, True)
        linear(prefill, spec.hidden_size, spec.hidden_size, prompt_len, False)
        linear(prefill, spec.hidden_size, mlp_out, prompt_len, False)
        linear(prefill, spec.hidden_size, mlp_out, prompt_len, False)
        linear(prefill, mlp_out, spec.hidden_size, prompt_len, False)

        def attention_standard(stage: Dict[str, float], seq_len: int, tokens: int) -> None:
            # qk matmul
            qk_ops = seq_len * head_dim * attn_heads * bs * 2 * tokens
            add(stage, OPs=qk_ops)
            if stage is decode:
                add(stage, load_act=head_dim * bs * attn_heads * self.a_byte * tokens)
            else:
                add(stage, load_act=seq_len * head_dim * bs * kv_heads * self.a_byte)
            add(stage, store_act=seq_len * tokens * bs * attn_heads * self.a_byte)
            add(stage, load_kv_cache=seq_len * head_dim * bs * kv_heads * self.kv_byte)
            # sv matmul
            sv_ops = tokens * head_dim * seq_len * attn_heads * bs * 2
            add(stage, OPs=sv_ops)
            add(stage, load_act=seq_len * tokens * bs * attn_heads * self.a_byte)
            add(stage, store_act=tokens * head_dim * bs * attn_heads * self.a_byte)
            add(stage, load_kv_cache=seq_len * head_dim * bs * kv_heads * self.kv_byte)
            # softmax
            softmax_ops = bs * attn_heads * seq_len * tokens * 5
            add(stage, OPs=softmax_ops)
            add(stage, load_act=bs * attn_heads * seq_len * self.a_byte * tokens)
            add(stage, store_act=bs * attn_heads * seq_len * self.a_byte * tokens)

        attention_standard(decode, prompt_len, 1)
        attention_standard(prefill, prompt_len, prompt_len)

        def add_norm(stage: Dict[str, float], tokens: int) -> None:
            ops = bs * spec.hidden_size * tokens * 7
            add(stage, OPs=ops)
            add(stage, load_act=bs * spec.hidden_size * self.a_byte * tokens)
            add(stage, store_act=bs * spec.hidden_size * self.a_byte * tokens)

        def add_residual(stage: Dict[str, float], tokens: int) -> None:
            ops = bs * spec.hidden_size * tokens
            add(stage, OPs=ops)
            add(stage, load_act=bs * spec.hidden_size * self.a_byte * tokens)
            add(stage, store_act=bs * spec.hidden_size * self.a_byte * tokens)

        def add_activation(stage: Dict[str, float], tokens: int) -> None:
            ops = bs * spec.hidden_size * tokens * 2
            add(stage, OPs=ops)
            add(stage, load_act=bs * spec.hidden_size * self.a_byte * tokens * 2)
            add(stage, store_act=bs * spec.hidden_size * self.a_byte * tokens)

        add_norm(decode, 1)
        add_norm(decode, 1)  # two norms per block
        add_residual(decode, 1)
        add_residual(decode, 1)
        add_activation(decode, 1)

        add_norm(prefill, prompt_len)
        add_norm(prefill, prompt_len)
        add_residual(prefill, prompt_len)
        add_residual(prefill, prompt_len)
        add_activation(prefill, prompt_len)

        # Multiply per-layer totals by number of transformer blocks
        for key in list(decode.keys()):
            decode[key] *= spec.num_layers
        for key in list(prefill.keys()):
            prefill[key] *= spec.num_layers

        # LM head applies once per stage
        lm_ops = bs * spec.hidden_size * spec.vocab_size * 2
        lm_weight = spec.hidden_size * spec.vocab_size * self.w_byte
        lm_load_act = spec.hidden_size * bs * self.a_byte
        lm_store_act = spec.vocab_size * bs * self.a_byte
        _accumulate(prefill, OPs=lm_ops * prompt_len)
        _accumulate(prefill, load_weight=lm_weight)
        _accumulate(prefill, load_act=lm_load_act * prompt_len)
        _accumulate(prefill, store_act=lm_store_act * prompt_len)
        _accumulate(decode, OPs=lm_ops)
        _accumulate(decode, load_weight=lm_weight)
        _accumulate(decode, load_act=lm_load_act)
        _accumulate(decode, store_act=lm_store_act)

        decode_memory_access = {
            "load_weight": decode["load_weight"],
            "load_act": decode["load_act"],
            "store_act": decode["store_act"],
            "load_kv_cache": decode.get("load_kv_cache", 0.0),
            "store_kv_cache": decode.get("store_kv_cache", 0.0),
        }
        prefill_memory_access = {
            "load_weight": prefill["load_weight"],
            "load_act": prefill["load_act"],
            "store_act": prefill["store_act"],
            "load_kv_cache": prefill.get("load_kv_cache", 0.0),
            "store_kv_cache": prefill.get("store_kv_cache", 0.0),
        }

        precision_bits = {"w": self.w_bit, "a": self.a_bit, "kv": self.kv_bit}
        decode_roofline = _roofline_eval(self.hardware, precision_bits, decode_memory_access, decode["OPs"])
        prefill_roofline = _roofline_eval(self.hardware, precision_bits, prefill_memory_access, prefill["OPs"])

        decode_total_time = decode_roofline["inference_time"] * self.output_len
        decode_total_ops = decode["OPs"] * self.output_len
        decode_total_memory = decode_roofline["memory_access"] * self.output_len

        overall_ops = prefill["OPs"] + decode_total_ops
        overall_memory = prefill_roofline["memory_access"] + decode_total_memory
        overall_roofline = _roofline_eval(self.hardware, precision_bits, {
            "prefill": prefill_roofline["memory_access"],
            "decode": decode_total_memory,
        }, overall_ops)
        overall_roofline["inference_time"] = prefill_roofline["inference_time"] + decode_total_time

        return {
            "hardware": self.hardware,
            "precision_bits": precision_bits,
            "model": {
                "hidden_size": spec.hidden_size,
                "num_layers": spec.num_layers,
                "num_attention_heads": spec.num_attention_heads,
                "num_key_value_heads": spec.num_key_value_heads,
                "intermediate_size": spec.intermediate_size,
                "vocab_size": spec.vocab_size,
            },
            "params": {
                "batch_size": bs,
                "prompt_len": prompt_len,
                "output_len": self.output_len,
                "use_flashattention": self.use_flashattention,
            },
            "prefill": {**prefill, **prefill_roofline},
            "decode": {**decode, **decode_roofline, "total_ops": decode_total_ops, "total_memory": decode_total_memory, "total_time": decode_total_time},
            "overall": {**overall_roofline, "total_ops": overall_ops, "total_memory": overall_memory},
        }


def compute_roofline(
    model_path: Path,
    hardware: str,
    batch_size: int,
    prompt_len: int,
    output_len: int,
    w_bit: int = 16,
    a_bit: int = 16,
    kv_bit: Optional[int] = None,
    use_flashattention: bool = False,
) -> Dict[str, Any]:
    estimator = RooflineEstimator(
        model_path=model_path,
        hardware=hardware,
        batch_size=batch_size,
        prompt_len=prompt_len,
        output_len=output_len,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
    )
    return estimator.compute()
