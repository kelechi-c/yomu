import re
import click
import subprocess
from pathlib import Path
from typing import Callable


def normalize_markdown(md_text: str) -> str:
    lines = md_text.splitlines()
    out = []

    bullet_re = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")

    for line in lines:
        m = bullet_re.match(line)
        if m:
            indent = len(m.group(1).replace("\t", "    "))
            level = indent // 2
            level = min(level, 2)  # cap nesting depth
            content = line[m.end() :]
            marker = m.group(2)
            out.append(("  " * level) + f"{marker} {content}")
        else:
            out.append(line)

    return "\n".join(out) + "\n"


def convert_markdown_file_to_pdf(
    md_file: Path,
    out_file: Path,
    pdf_engine: str = "xelatex",
    reconvert_existing: bool = False,
    echo: Callable[[str], None] | None = None,
) -> bool:
    """Convert one markdown file to PDF via pandoc.

    Returns True when converted, False when skipped because output exists.
    """
    log = echo or (lambda _: None)
    md_file = md_file.expanduser().resolve()
    out_file = out_file.expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not reconvert_existing:
        log(f"skipping {md_file.name} (exists: {out_file})")
        return False

    temp_md = out_file.parent / f"{md_file.stem}.normalized.md"
    text = md_file.read_text(encoding="utf-8")
    temp_md.write_text(normalize_markdown(text), encoding="utf-8")

    cmd = [
        "pandoc",
        str(temp_md),
        "-o",
        str(out_file),
        f"--pdf-engine={pdf_engine}",
        "-V",
        "geometry:margin=0.5in",
    ]

    try:
        subprocess.run(cmd, check=True)
        log(f"OK   {md_file.name} -> {out_file}")
        return True
    finally:
        if temp_md.exists():
            temp_md.unlink()


def convert_markdown_dir_to_pdf(
    input_dir: Path,
    output_dir: Path | None = None,
    pdf_engine: str = "xelatex",
    reconvert_existing: bool = False,
    echo: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """Convert all markdown files in a directory to PDF.

    Returns (converted_count, skipped_count).
    """
    log = echo or click.echo
    input_dir = input_dir.expanduser().resolve()
    output_dir = (output_dir or (input_dir / "pdfs")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        log(f"No .md files found in {input_dir}")
        raise SystemExit(1)

    converted_count = 0
    skipped_count = 0
    for md_file in md_files:
        out_file = output_dir / f"{md_file.stem}.pdf"
        try:
            converted = convert_markdown_file_to_pdf(
                md_file=md_file,
                out_file=out_file,
                pdf_engine=pdf_engine,
                reconvert_existing=reconvert_existing,
                echo=log,
            )
            if converted:
                converted_count += 1
            else:
                skipped_count += 1
        except subprocess.CalledProcessError as exc:
            log(f"FAIL {md_file.name}: {exc}")

    return converted_count, skipped_count


@click.command()
@click.argument("input_dir", type=click.Path(path_type=Path), default=".")
@click.argument("output_dir", type=click.Path(path_type=Path), required=False)
@click.option(
    "--pdf-engine",
    default="xelatex",
    show_default=True,
    help="Pandoc PDF engine (e.g. xelatex, wkhtmltopdf, weasyprint).",
)
@click.option(
    "--reconvert-existing/--skip-existing",
    default=False,
    show_default=True,
    help="Reconvert files even when the output PDF already exists.",
)
def main(
    input_dir: Path,
    output_dir: Path | None,
    pdf_engine: str,
    reconvert_existing: bool,
) -> None:
    converted_count, skipped_count = convert_markdown_dir_to_pdf(
        input_dir=input_dir,
        output_dir=output_dir,
        pdf_engine=pdf_engine,
        reconvert_existing=reconvert_existing,
        echo=click.echo,
    )
    click.echo(f"Done. converted={converted_count}, skipped={skipped_count}")


if __name__ == "__main__":
    main()
