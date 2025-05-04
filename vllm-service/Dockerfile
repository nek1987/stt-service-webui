# Используем официальный образ vLLM как базовый
FROM vllm/vllm-openai:latest

# Устанавливаем дополнительные зависимости для аудио
# --no-cache-dir для экономии места
# --upgrade для обновления vllm, если оно уже установлено, с добавлением аудио зависимостей
RUN pip install --no-cache-dir --upgrade "vllm[audio]"

# Устанавливаем ffmpeg и libsndfile1 для обработки аудио
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Нет необходимости в CMD или ENTRYPOINT, так как они уже определены в базовом образе
# и команда передается через docker-compose.yml