# Используем официальный образ vLLM как базовый
FROM vllm/vllm-openai:latest

# --- Исправление ошибок сборки и памяти ---

# 1) Отключаем оптимизации GnuTLS, чтобы метод https не падал SIGILL
# Это должно решить проблему с "Illegal Instruction" во время apt-get update
ENV GNUTLS_CPUID_OVERRIDE=0x1

# 2) Устанавливаем дополнительные зависимости для аудио и утилиты для аудиофайлов
# Используем опции для apt-get для устойчивости
# Разбиваем на два шага RUN для большей ясности и потенциально лучшей обработки кэша/ресурсов
RUN apt-get update -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::Timeout=30
RUN apt-get install -y -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::Timeout=30 ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# 3) Устанавливаем дополнительные зависимости vLLM для аудио
# --no-cache-dir для экономии места
# --upgrade для обновления vllm, если оно уже установлено, с добавлением аудио зависимостей
RUN pip install --no-cache-dir --upgrade "vllm[audio]"

# 4) Переменная окружения для улучшения управления памятью PyTorch/CUDA
# Рекомендовано в логах ошибки при фрагментации памяти
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Конец исправлений ---


# Нет необходимости в CMD или ENTRYPOINT, так как они уже определены в базовом образе
# и команда передается через docker-compose.yml
