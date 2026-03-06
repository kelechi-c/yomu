uv run yomu_all.py \
  --model qwen/qwen3.5-35b-a3b \
  --base-url https://openrouter.ai/api/v1 \
  --prompt-type summary \
  --max-files 1 \
  --regenerate \
  --input-dir test/ \
  --output-dir test_output/ \
  # --convert-md-to-pdf \
#   --difficulty high