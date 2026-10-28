cd /data/changzou/deepseek4robotics
conda activate ds4ro
set -a && source .env && set +a
export OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1
export OPENAI_COMPATIBLE_API_KEY="$DEEPSEEK_API_KEY"
python -m skillopt_libero train