uv run yomu_all.py \
  --model qwen/qwen3.5-35b-a3b \
  --base-url https://openrouter.ai/api/v1 \
  --prompt-type summary \
  --input-dir ~/Downloads/materials_1404 \
  --output-dir ~/Downloads/materials_1404/md_summaries \
  --pdf-output-dir ~/Downloads/materials_1404/pdf_summaries \
  --max-files 50 \
  --convert-md-to-pdf \
  --name-suffix qwen\
  # --regenerate \ # if you want to regenerate summaries, uncomment this line e.g when switching models :)