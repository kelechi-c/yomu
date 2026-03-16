# edit args as needed. include api key in local .env file or export in shell

uv run yomu_all.py \
  --model qwen/qwen3.5-35b-a3b \
  --base-url https://openrouter.ai/api/v1 \
  --prompt-type summary \
  --input-dir ~/Downloads/materials_400_first_semester/update/ \
  --output-dir ~/Downloads/summaries_400_1/ \
  --pdf-output-dir ~/Downloads/pdf_summaries_1303/ \
  --max-files 50 \
  --convert-md-to-pdf \