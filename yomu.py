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
import dotenv
from convert2pdf import convert_markdown_file_to_pdf

dotenv.load_dotenv()


T = TypeVar("T")
OCR_CACHE_VERSION = "v2"
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
            click.echo(f" [latency] {step_name}: {_latency_seconds(start, end)}")
            return result

        return wrapper

    return decorator


def _timed_call(step_name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    click.echo(f" [latency] {step_name}: {_latency_seconds(start, end)}")
    return result


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ocr_cache_path(
    cache_dir: Path,
    pdf_bytes: bytes,
    max_pages: int,
    batch_size: int,
) -> Path:
    file_hash = _sha256_hex(pdf_bytes)
    cache_key = (
        f"{file_hash}_p{max_pages}_b{batch_size}_{OCR_CACHE_VERSION}"
    )
    return cache_dir / f"{cache_key}.txt"


def _legacy_ocr_cache_candidates(
    cache_dir: Path,
    pdf_bytes: bytes,
    max_pages: int,
    batch_size: int,
) -> List[Path]:
    file_hash = _sha256_hex(pdf_bytes)
    patterns = [
        f"{file_hash}_*_p{max_pages}_b{batch_size}_{OCR_CACHE_VERSION}.txt",
        f"{file_hash}_*_p{max_pages}_v1.txt",
    ]
    candidates: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(
            cache_dir.glob(pattern),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            if path not in seen:
                candidates.append(path)
                seen.add(path)
    return candidates


def _read_cached_ocr(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _load_cached_ocr(
    canonical_path: Path,
    cache_dir: Path,
    pdf_bytes: bytes,
    max_pages: int,
    batch_size: int,
) -> tuple[str | None, Path | None]:
    cached_text = _read_cached_ocr(canonical_path)
    if cached_text:
        return cached_text, canonical_path

    for legacy_path in _legacy_ocr_cache_candidates(
        cache_dir,
        pdf_bytes,
        max_pages,
        batch_size,
    ):
        cached_text = _read_cached_ocr(legacy_path)
        if cached_text:
            if legacy_path != canonical_path and not canonical_path.exists():
                canonical_path.write_text(cached_text, encoding="utf-8")
            return cached_text, legacy_path

    return None, None


# @_latency_marker("native_text_extraction")
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


def _image_to_base64_jpeg(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_batched_page_text(raw_text: str, expected_pages: List[int]) -> List[str]:
    normalized = raw_text.strip()
    if not normalized:
        return []

    pattern = re.compile(
        r"(?ms)^\[Page\s+(\d+)\]\s*\n(.*?)(?=^\[Page\s+\d+\]\s*\n|\Z)"
    )
    matches = list(pattern.finditer(normalized))
    if matches:
        page_to_text: dict[int, str] = {}
        for match in matches:
            page_num = int(match.group(1))
            page_text = match.group(2).strip()
            if page_text and page_num not in page_to_text:
                page_to_text[page_num] = page_text

        chunks: List[str] = []
        for page_num in expected_pages:
            page_text = page_to_text.get(page_num)
            if page_text:
                chunks.append(f"\n[Page {page_num}]\n{page_text}")
        if chunks:
            return chunks

    if len(expected_pages) == 1:
        return [f"\n[Page {expected_pages[0]}]\n{normalized}"]

    return [f"\n[Pages {expected_pages[0]}-{expected_pages[-1]}]\n{normalized}"]


@_latency_marker("ocr_fallback")
def _ocr_with_model(
    client: OpenAI,
    model: str,
    pdf_bytes: bytes,
    max_pages: int,
    batch_size: int,
) -> str:
    pages = _render_pdf_pages(pdf_bytes, max_pages=max_pages)
    chunks: List[str] = []
    batch_size = max(1, batch_size)

    for batch_start in range(0, len(pages), batch_size):
        batch_pages = pages[batch_start : batch_start + batch_size]
        expected_page_nums = list(
            range(batch_start + 1, batch_start + len(batch_pages) + 1)
        )
        page_map = "\n".join(
            f"- image {i + 1} => page {page_num}"
            for i, page_num in enumerate(expected_page_nums)
        )

        content: List[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You will receive one or more lecture page images in order. "
                    "Extract all readable text from each page exactly as written. "
                    "Preserve headings and bullet structure when possible.\n\n"
                    "Page mapping for this request:\n"
                    f"{page_map}\n\n"
                    "Return plain text only and use this exact structure for every page:\n"
                    "[Page <absolute_page_number>]\n"
                    "<transcribed text>"
                ),
            }
        ]

        for image in batch_pages:
            image_b64 = _image_to_base64_jpeg(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        batch_text = _completion_to_text(response)
        if batch_text:
            chunks.extend(_extract_batched_page_text(batch_text, expected_page_nums))

    return "\n".join(chunks).strip()


# @_latency_marker("prompt_load")
def _load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# @_latency_marker("prompt_build")
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


# @_latency_marker("markdown_render")
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


def _normalize_name_segment(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or fallback


def _normalized_output_name(
    pdf_stem: str,
    prompt_type: str,
    name_suffix: str | None = None,
) -> str:
    normalized_stem = _normalize_name_segment(pdf_stem, "untitled")
    prompt_suffix = _normalize_name_segment(prompt_type, "output")
    parts = [normalized_stem, prompt_suffix]
    if name_suffix:
        parts.append(_normalize_name_segment(name_suffix, "tag"))
    return f"{'_'.join(parts)}.md"


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
    "--ocr-batch-size",
    default=8,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of rendered pages sent per OCR model request.",
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
@click.option(
    "--name-suffix",
    default=None,
    help=(
        "Optional suffix/tag appended to generated markdown and PDF filenames, "
        "for example 'v2' or 'midterm-review'."
    ),
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
    ocr_batch_size: int,
    regenerate: bool,
    convert_md_to_pdf: bool,
    pdf_output_dir: Path | None,
    pdf_engine: str,
    prompt_type: str,
    prompt_file: Path | None,
    difficulty: str,
    name_suffix: str | None,
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
    click.echo(f"ocr batch size: {ocr_batch_size}")
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
    click.echo(f"name suffix: {name_suffix or '(none)'}")

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        file_start = time.perf_counter()
        click.echo(f"\n[{idx}/{total}] preprocessing: {pdf_path.name}")
        try:
            out_filename = _normalized_output_name(
                pdf_path.stem,
                prompt_type.lower(),
                name_suffix=name_suffix,
            )
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

                pdf_bytes = pdf_path.read_bytes()
                # pdf_bytes = _timed_call("read_pdf_bytes", pdf_path.read_bytes)
                material_text = _extract_pdf_text(pdf_bytes)

                if len(material_text) < ocr_threshold:
                    click.echo("native text is low")
                    cache_path = _ocr_cache_path(
                        ocr_cache_dir,
                        pdf_bytes,
                        max_ocr_pages,
                        ocr_batch_size,
                    )
                    cached_ocr_text, cache_hit_path = _load_cached_ocr(
                        cache_path,
                        ocr_cache_dir,
                        pdf_bytes,
                        max_ocr_pages,
                        ocr_batch_size,
                    )

                    if cached_ocr_text:
                        if cache_hit_path == cache_path:
                            click.echo("using shared ocr cache")
                        else:
                            click.echo(
                                f"using legacy ocr cache and promoting it -> {cache_path.name}"
                            )
                        material_text = cached_ocr_text
                    else:
                        click.echo("running OCR fallback")
                        material_text = _ocr_with_model(
                            client,
                            model,
                            pdf_bytes,
                            max_pages=max_ocr_pages,
                            batch_size=ocr_batch_size,
                        )
                        cache_path.write_text(material_text, "utf-8")
                        # _timed_call(
                        #     "ocr_cache_write",
                        # )

                if not material_text:
                    raise ValueError("Could not extract readable text from PDF.")

                prompt = _build_prompt(prompt_template, material_text, difficulty.lower())

                click.echo("generating output with model...")
                summary_text = _generate_output(client, model, prompt)
                if not summary_text:
                    raise ValueError("Model returned empty output.")

                output_title = _output_title_for_prompt_type(prompt_type.lower(), pdf_path.name)
                output_markdown = _summary_to_markdown(output_title, summary_text)
                out_path.write_text(output_markdown, encoding="utf-8")
                # _timed_call("write_markdown", , output_markdown, "utf-8")
                click.echo(f"saved => {out_path}")

            if convert_md_to_pdf:
                if not out_path.exists():
                    raise ValueError(f"Markdown file is not available for conversion: {out_path}")
                assert resolved_pdf_output_dir is not None
                converted_path = resolved_pdf_output_dir / f"{out_path.stem}.pdf"
                # _timed_call("md_to_pdf",
                convert_markdown_file_to_pdf(
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
