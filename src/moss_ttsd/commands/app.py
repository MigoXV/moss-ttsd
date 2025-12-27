from __future__ import annotations

import logging
from typing import Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(help="MOSS-TTSD command line tools")


def _resolve_dtype(dtype_name: Optional[str]):
    if dtype_name is None:
        return None

    normalized = dtype_name.lower()
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return None

    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16

    logger.warning("Unknown dtype '%s', defaulting to framework default", dtype_name)
    return None


@app.command()
def serve(
    model_dir: str = typer.Option(
        "./model-bin/MOSS-TTSD-v0.7",
        help="Path to local MOSS-TTSD model directory",
        envvar="MODEL_DIR",
    ),
    codec_path: Optional[str] = typer.Option(
        "model-bin/XY_Tokenizer_TTSD_V0_hf",
        help="Path to local codec model (XY_Tokenizer...)",
        envvar="MOSS_TTSD_CODEC_PATH",
    ),
    voices_dir: Optional[str] = typer.Option(
        "./model-bin/voices",
        help="Root directory of voices; each subdir is a voice name",
        envvar="VOICES_DIR",
    ),
    host: str = typer.Option("0.0.0.0", help="Server host", envvar="HOST"),
    port: int = typer.Option(8000, help="Server port", envvar="PORT"),
    device: str = typer.Option("cpu", help="Device to run model on (cpu/cuda/auto)", envvar="DEVICE"),
    dtype: Optional[str] = typer.Option("bfloat16", help="Torch dtype: float16/float32/bfloat16", envvar="DTYPE"),
    max_new_tokens: int = typer.Option(
        4096,
        help="Max new tokens for generation",
        envvar="MAX_NEW_TOKENS",
    ),
    fallback_audio: str = typer.Option(
        "error",
        help="When inference is unavailable: error/dummy",
        envvar="FALLBACK_AUDIO",
    ),
):
    """Start the OpenAI-compatible TTS service."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    dtype_obj = _resolve_dtype(dtype)

    logger.info("=" * 60)
    logger.info("Starting MOSS-TTSD TTS Service")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  model_dir      : %s", model_dir)
    logger.info("  codec_path     : %s", codec_path)
    logger.info("  voices_dir     : %s", voices_dir)
    logger.info("  host           : %s", host)
    logger.info("  port           : %s", port)
    logger.info("  device         : %s", device)
    logger.info("  dtype          : %s", dtype_obj)
    logger.info("  max_new_tokens : %s", max_new_tokens)
    logger.info("  fallback_audio : %s", fallback_audio)
    logger.info("=" * 60)

    import uvicorn  # type: ignore

    from moss_ttsd.servicer import get_inferencer, initialize_inferencer
    from moss_ttsd.servicer.servicer import create_app

    inferencer = initialize_inferencer(
        model_dir=model_dir,
        codec_path=codec_path,
        voices_dir=voices_dir,
        device=device,
        dtype=dtype_obj,
        max_new_tokens=max_new_tokens,
        fallback_audio=fallback_audio,
    )
    if fallback_audio == "error" and (inferencer is None or not inferencer.is_ready):
        logger.error("Inferencer initialization failed: %s", getattr(inferencer, "init_error", None))
        raise typer.Exit(code=1)
    fastapi_app = create_app(get_inferencer)

    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        reload=False,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
