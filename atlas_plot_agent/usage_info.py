from dataclasses import dataclass
from typing import Optional


@dataclass
class UsageInfo:
    model: str
    elapsed: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cost: Optional[float]


def get_usage_info(response, model_info, elapsed):
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
        cost = (prompt_tokens / 1_000_000) * model_info.input_cost_per_million + (
            completion_tokens / 1_000_000
        ) * model_info.output_cost_per_million
        return UsageInfo(
            model=model_info.model_name,
            elapsed=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )
    else:
        return UsageInfo(
            model=model_info.model_name,
            elapsed=elapsed,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cost=None,
        )
