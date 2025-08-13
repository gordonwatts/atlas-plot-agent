from dataclasses import dataclass
from typing import List, Optional


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


def sum_usage_infos(usages: List[UsageInfo]) -> UsageInfo:
    """
    Sums a list of UsageInfo objects, returning a new UsageInfo with summed values.
    Model name will be a comma-separated list, elapsed is summed,
    tokens and cost are summed, treating None as 0. Always returns int/float for token/cost fields.
    """
    if len(usages) == 0:
        raise ValueError(
            "`sum_usage_infos` must be given a non-zero length usage list."
        )

    model = ",".join(u.model for u in usages)
    elapsed = sum(u.elapsed if u.elapsed is not None else 0 for u in usages)

    def sum_or_zero(attr):
        vals = [getattr(u, attr) for u in usages]
        return sum(v if v is not None else 0 for v in vals)

    prompt_tokens = sum_or_zero("prompt_tokens")
    completion_tokens = sum_or_zero("completion_tokens")
    total_tokens = sum_or_zero("total_tokens")
    cost = sum_or_zero("cost")

    return UsageInfo(
        model=model,
        elapsed=elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
    )
