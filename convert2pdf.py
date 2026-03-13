import click
import subprocess
from pathlib import Path


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
    output_dir = output_dir or (input_dir / "pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        click.echo(f"No .md files found in {input_dir}")
        raise SystemExit(1)

    for md_file in md_files:
        out_file = output_dir / f"{md_file.stem}.pdf"
        if out_file.exists() and not reconvert_existing:
            click.echo(f"skipping {md_file.name} (exists: {out_file})")
            continue

        cmd = [
            "pandoc",
            str(md_file),
            "-o",
            str(out_file),
            f"--pdf-engine={pdf_engine}",
            "-V",
            "geometry:margin=0.5in",
        ]

        try:
            subprocess.run(cmd, check=True)
            click.echo(f"OK   {md_file.name} -> {out_file}")
        except subprocess.CalledProcessError as e:
            click.echo(f"FAIL {md_file.name}: {e}")

    click.echo("Done.")


if __name__ == "__main__":
    main()
