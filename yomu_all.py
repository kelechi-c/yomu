import io
import os
import re
import time
import hashlib
import base64
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, TypeVar

import pypdfium2 as pdfium
from openai import OpenAI
from PIL import Image
from pypdf import PdfReader
import click
try:
    from .convert2pdf import convert_markdown_file_to_pdf
except ImportError:
    from convert2pdf import convert_markdown_file_to_pdf

import dotenv

dotenv.load_dotenv()


T = TypeVar("T")
OCR_CACHE_VERSION = "v1"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT_FILES = {
    "summary": DEFAULT_PROMPT_DIR / "summary.md",
    "questions": DEFAULT_PROMPT_DIR / "questions.md",
    "mcq_100": DEFAULT_PROMPT_DIR / "mcq_100.md",
    "theory_5_applied": DEFAULT_PROMPT_DIR / "theory_5_applied.md",
    "theory_5_direct": DEFAULT_PROMPT_DIR / "theory_5_direct.md",
}


def _latency_seconds(start: float, end: float) -> str:
    return f"{(end - start):.3f} s"


def _latency_marker(step_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            click.echo(f"  [latency] {step_name}: {_latency_seconds(start, end)}")
            return result

        return wrapper

    return decorator


def _timed_call(step_name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    click.echo(f"  [latency] {step_name}: {_latency_seconds(start, end)}")
    return result


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ocr_cache_path(cache_dir: Path, pdf_bytes: bytes, model: str, max_pages: int) -> Path:
    model_key = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") or "model"
    file_hash = _sha256_hex(pdf_bytes)
    cache_key = f"{file_hash}_{model_key}_p{max_pages}_{OCR_CACHE_VERSION}"
    return cache_dir / f"{cache_key}.txt"


def _read_cached_ocr(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


@_latency_marker("native_text_extraction")
def _extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: List[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(f"\n[Page {page_num}]\n{text.strip()}")
    return "\n".join(chunks).strip()


def _render_pdf_pages(pdf_bytes: bytes, max_pages: int = 60) -> List[Image.Image]:
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    page_count = min(len(doc), max_pages)
    images: List[Image.Image] = []
    for i in range(page_count):
        page = doc[i]
        images.append(page.render(scale=2).to_pil())
    return images


@_latency_marker("ocr_fallback")
def _ocr_with_model(client: OpenAI, model: str, pdf_bytes: bytes, max_pages: int) -> str:
    pages = _render_pdf_pages(pdf_bytes, max_pages=max_pages)
    chunks: List[str] = []
    for idx, image in enumerate(pages, start=1):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all readable text from this lecture page image exactly "
                                "as written. Preserve headings and bullet structure when possible. "
                                "Return plain text only."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        text = _completion_to_text(response)
        if text:
            chunks.append(f"\n[Page {idx}]\n{text}")
    return "\n".join(chunks).strip()


@_latency_marker("prompt_load")
def _load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@_latency_marker("prompt_build")
def _build_prompt(template: str, material_text: str, difficulty: str) -> str:
    rendered = template
    rendered = rendered.replace("{{difficulty}}", difficulty)
    if "{{material}}" in template:
        return rendered.replace("{{material}}", material_text)
    return (
        f"{rendered.rstrip()}\n\n"
        "Lecture material starts below:\n"
        "=== START MATERIAL ===\n"
        f"{material_text}\n"
        "=== END MATERIAL ==="
    )


@_latency_marker("model_generation")
def _generate_output(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning": {"exclude": True, "enabled": False}},
    )
    return _completion_to_text(response)


def _completion_to_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        raise ValueError("Model response is missing 'choices' field.")
    
    message = response.choices[0].message
    content = getattr(message, "content", None)

    return str(content).strip()


def _strip_markdown_bold(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


@_latency_marker("markdown_render")
def _summary_to_markdown(title: str, content: str) -> str:
    normalized_title = _strip_markdown_bold(title).strip()
    lines: List[str] = [f"# {normalized_title}", ""]
    lines.extend(content.splitlines())
    return "\n".join(lines).rstrip() + "\n"


def _load_env_file_if_present(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _normalized_output_name(pdf_stem: str, prompt_type: str) -> str:
    normalized_stem = re.sub(r"[^a-z0-9]+", "_", pdf_stem.lower()).strip("_")
    normalized_stem = normalized_stem or "untitled"
    prompt_suffix = re.sub(r"[^a-z0-9]+", "_", prompt_type.lower()).strip("_") or "output"
    return f"{normalized_stem}_{prompt_suffix}.md"


def _resolve_prompt_file(prompt_type: str, prompt_file: Path | None) -> Path:
    if prompt_file is not None:
        return prompt_file.expanduser().resolve()
    return DEFAULT_PROMPT_FILES[prompt_type]


def _output_title_for_prompt_type(prompt_type: str, source_name: str) -> str:
    title_map = {
        "summary": "Comprehensive Summary",
        "questions": "Question Set",
        "mcq_100": "100 MCQs",
        "theory_5_applied": "Theory Questions (Applied)",
        "theory_5_direct": "Theory Questions (Direct)",
    }
    prefix = title_map.get(prompt_type, "Study Output")
    return f"{prefix} - {source_name}"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input-dir",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory that contains PDF files.",
)
@click.option(
    "--output-dir",
    default="outputs",
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for generated summary PDFs.",
)
@click.option(
    "--model",
    default="qwen/qwen3.5-35b-a3b",
    show_default=True,
    help="Model name (OpenAI-compatible).",
)
@click.option(
    "--base-url",
    default="https://openrouter.ai/api/v1",
    help="Optional OpenAI-compatible base URL (e.g. Gemini OpenAI endpoint).",
)
@click.option(
    "--api-key",
    default=None,
    help="API key. If omitted, uses OPENAI_API_KEY or GEMINI_API_KEY from env/.env.",
)
@click.option(
    "--max-files",
    default=2,
    show_default=True,
    type=click.IntRange(min=0),
    help="Maximum number of PDFs to process. 0 means process all files.",
)
@click.option(
    "--max-ocr-pages",
    default=60,
    show_default=True,
    type=click.IntRange(min=1),
    help="Max pages rendered for OCR fallback per PDF.",
)
@click.option(
    "--ocr-threshold",
    default=5000,
    show_default=True,
    type=click.IntRange(min=0),
    help="If extracted native text length is below this, OCR fallback runs.",
)
@click.option(
    "--ocr-cache-dir",
    default=".yomu_ocr_cache",
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for persistent OCR cache across sessions.",
)
@click.option(
    "--regenerate",
    is_flag=True,
    default=False,
    help="Regenerate markdown even if output .md already exists.",
)
@click.option(
    "--convert-md-to-pdf",
    is_flag=True,
    default=True,
    help="Also convert each generated markdown to PDF via convert2pdf.py.",
)
@click.option(
    "--pdf-output-dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for converted PDFs. Defaults to --output-dir/pdfs.",
)
@click.option(
    "--pdf-engine",
    default="xelatex",
    show_default=True,
    help="Pandoc PDF engine used when --convert-md-to-pdf is set.",
)
@click.option(
    "--prompt-type",
    default="summary",
    show_default=True,
    type=click.Choice(
        ["summary", "questions", "mcq_100", "theory_5_applied", "theory_5_direct"],
        case_sensitive=False,
    ),
    help="Prompt preset to use when --prompt-file is not provided.",
)
@click.option(
    "--prompt-file",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, exists=False, path_type=Path),
    help="Custom prompt template file (.txt/.md). Use {{material}} as placeholder.",
)
@click.option(
    "--difficulty",
    default="medium",
    show_default=True,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Difficulty used by prompt templates that include {{difficulty}}.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    model: str,
    base_url: str | None,
    api_key: str | None,
    max_files: int,
    max_ocr_pages: int,
    ocr_threshold: int,
    ocr_cache_dir: Path,
    regenerate: bool,
    convert_md_to_pdf: bool,
    pdf_output_dir: Path | None,
    pdf_engine: str,
    prompt_type: str,
    prompt_file: Path | None,
    difficulty: str,
) -> None:
    run_start = time.perf_counter()
    _load_env_file_if_present(Path(".env"))
    print('yomu initialized.... \n __________________________________')

    # api_key = os.getenv("OPENAI_API_KEY").strip()
    api_key = (
        os.getenv("GEMINI_API_KEY", "").strip() if 'gemini' in model else os.getenv("OPENAI_API_KEY", "").strip()
    )

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_cache_dir = ocr_cache_dir.expanduser().resolve()
    ocr_cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_pdf_output_dir: Path | None = None
    if convert_md_to_pdf:
        resolved_pdf_output_dir = (pdf_output_dir or (output_dir / "pdfs")).expanduser().resolve()
        resolved_pdf_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_prompt_file = _resolve_prompt_file(prompt_type.lower(), prompt_file)
    if not resolved_prompt_file.exists():
        raise click.ClickException(f"Prompt file not found: {resolved_prompt_file}")
    prompt_template = _load_prompt_template(resolved_prompt_file)

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if max_files > 0:
        pdf_paths = pdf_paths[:max_files]

    if not pdf_paths:
        raise click.ClickException(f"No PDF files found in: {input_dir}")

    client: OpenAI | None = None
    total = len(pdf_paths)
    success_count = 0
    failure_count = 0
    skipped_count = 0

    click.echo(f"found {total} pdf file(s) in {input_dir}")
    click.echo(f"output path -> {output_dir}")
    click.echo(f"ocr cache path -> {ocr_cache_dir}")
    click.echo(f"llm id: {model}")
    click.echo(f"base url: {base_url or '(default openai)'}")
    click.echo(f"regenerate: {regenerate}")
    click.echo(f"convert md to pdf: {convert_md_to_pdf}")
    if convert_md_to_pdf and resolved_pdf_output_dir is not None:
        click.echo(f"pdf output path -> {resolved_pdf_output_dir}")
        click.echo(f"pdf engine: {pdf_engine}")
    click.echo(f"prompt type: {prompt_type.lower()}")
    click.echo(f"prompt file: {resolved_prompt_file}")
    click.echo(f"difficulty: {difficulty.lower()}")

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        file_start = time.perf_counter()
        click.echo(f"\n[{idx}/{total}] preprocessing: {pdf_path.name}")
        try:
            out_filename = _normalized_output_name(pdf_path.stem, prompt_type.lower())
            out_path = output_dir / out_filename
            md_exists = out_path.exists()
            if md_exists and not regenerate:
                click.echo(f"  markdown exists, skipping regeneration: {out_path}")
                skipped_count += 1
            else:
                if not api_key:
                    raise click.ClickException(
                        "Missing API key. Set OPENAI_API_KEY/GEMINI_API_KEY or pass --api-key."
                    )
                # if client is None:
                
                # print(base_url, api_key)
                
                client = OpenAI(
                    api_key=api_key, base_url=base_url,
                )

                pdf_bytes = _timed_call("read_pdf_bytes", pdf_path.read_bytes)
                material_text = _extract_pdf_text(pdf_bytes)

                if len(material_text) < ocr_threshold:
                    click.echo("native text is low; running OCR fallback with model...")
                    cache_path = _ocr_cache_path(ocr_cache_dir, pdf_bytes, model, max_ocr_pages)
                    cached_ocr_text = _timed_call("ocr_cache_lookup", _read_cached_ocr, cache_path)

                    if cached_ocr_text:
                        click.echo("  using cached ocr text")
                        material_text = cached_ocr_text
                    else:
                        material_text = _ocr_with_model(
                            client, model, pdf_bytes, max_pages=max_ocr_pages
                        )
                        _timed_call(
                            "ocr_cache_write", cache_path.write_text, material_text, "utf-8"
                        )

                if not material_text:
                    raise ValueError("Could not extract readable text from PDF.")

                prompt = _build_prompt(prompt_template, material_text, difficulty.lower())

                click.echo("generating output with model...")
                summary_text = _generate_output(client, model, prompt)
                if not summary_text:
                    raise ValueError("Model returned empty output.")

                output_title = _output_title_for_prompt_type(prompt_type.lower(), pdf_path.name)
                output_markdown = _summary_to_markdown(output_title, summary_text)
                _timed_call("write_markdown", out_path.write_text, output_markdown, "utf-8")
                click.echo(f"saved => {out_path}")

            if convert_md_to_pdf:
                if not out_path.exists():
                    raise ValueError(f"Markdown file is not available for conversion: {out_path}")
                assert resolved_pdf_output_dir is not None
                converted_path = resolved_pdf_output_dir / f"{out_path.stem}.pdf"
                _timed_call(
                    "md_to_pdf",
                    convert_markdown_file_to_pdf,
                    md_file=out_path,
                    out_file=converted_path,
                    pdf_engine=pdf_engine,
                    reconvert_existing=regenerate,
                    echo=click.echo,
                )

            success_count += 1
            file_end = time.perf_counter()
            click.echo(f"[latency] file_total: {_latency_seconds(file_start, file_end)}")

        except Exception as exc:
            failure_count += 1
            click.echo(f"failed: {pdf_path.name} -> {exc}")
            file_end = time.perf_counter()
            click.echo(f"[latency] file_total: {_latency_seconds(file_start, file_end)}")

    click.echo("\npipelinecomplete")
    click.echo(f"successfully analyzed: {success_count}")
    click.echo(f"skipped regeneration: {skipped_count}")
    click.echo(f"failed: {failure_count}")
    run_end = time.perf_counter()
    click.echo(f"[latency] full_run_total: {_latency_seconds(run_start, run_end)}")


if __name__ == "__main__":
    main()
