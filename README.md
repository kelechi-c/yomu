# yomu
script/mini-scaffold project for AI-generated lecture summaries, questions, etc, given PDF documents. I made it cus I was lazy

## installation

Install the project dependencies:

```bash
uv sync
```
or:

```bash
pip install -e .
```

Set an `OPENAI_API_KEY` key before running (for OpenAI-compatible providers such as OpenRouter, actually, I have only used openrouter)

if you will use PDF export(you definitely will, unless you like reading raw markdown), install `pandoc` and `xelatex` engine locally, like...

- for Linux:
```sh
sudo dnf install pandoc pdflatex xelatex texlive texlive-collection-latexextra
```
(or `apt`, if on Ubuntu/Debian distros. the extra dependencies will swell with pdf rendering complexity)

- Mac users should be able to install it via `brew install...`, Windows users can figure theirs out... 

## usage

`yomu.py` processes PDF files from a directory, extracts text, falls back to OCR transcription when needed, generates markdown with the selected prompt, and can also convert the markdown output to PDF.

basic example:

```bash
uv run python yomu.py \
  --model qwen/qwen3.5-35b-a3b \
  --base-url https://openrouter.ai/api/v1 \
  --prompt-type summary \
  --input-dir ~/Downloads/materials_1404 \
  --output-dir ~/Downloads/materials_1404/md_summaries \
  --pdf-output-dir ~/Downloads/materials_1404/pdf_summaries \
  --max-files 50 \
  --convert-md-to-pdf \
  --name-suffix qwen_v1 # just as a version tag
```

major options:

- `--prompt-type` supports `summary`, `questions`, `mcq_100`, `theory_5_applied`, and `theory_5_direct`
- `--difficulty` supports `low`, `medium`, and `high`, for questions
- `--regenerate` forces regeneration when an output markdown file already exists
- `--max-files 0` processes all PDFs in the input directory
- `--prompt-file path/to/prompt.md` uses a custom prompt template (if you make yours)

note that I have only tested the summary prompts thoroughly, the question paths are undefined. 

**NB**: You need to have openrouter credits if using Openrouter models.
contributions are highly welcome as well. 

See all options via...

```bash
uv run python yomu.py --help
```
