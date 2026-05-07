from __future__ import annotations


def resolve_device(requested: str | None = "auto") -> dict[str, str]:
    import torch

    raw = (requested or "auto").strip().lower()
    if raw == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    elif raw == "cpu":
        resolved = "cpu"
    elif raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        resolved = "cuda"
    elif raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {requested!r} but CUDA is not available.")
        try:
            index = int(raw.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"Unsupported device string {requested!r}. Use auto, cpu, cuda, or cuda:<index>."
            ) from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device index {index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        resolved = raw
    else:
        raise ValueError(
            f"Unsupported device string {requested!r}. Use auto, cpu, cuda, or cuda:<index>."
        )

    return {"requested": requested or "auto", "resolved": resolved}
