# Используем официальный образ vLLM как базовый
FROM vllm/vllm-openai:latest

# 1) Отключаем оптимизации GnuTLS, чтобы метод https не падал SIGILL
# Это должно решить проблему с "Illegal Instruction"
ENV GNUTLS_CPUID_OVERRIDE=0x1

# Устанавливаем дополнительные зависимости для аудио и утилиты для аудиофайлов
# Сохраняем опции для apt-get для принудительного IPv4 и таймаутов/попыток
# на случай, если есть дополнительные сетевые факторы, хотя SIGILL указывает на GnuTLS/CPU
RUN apt-get update -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::Timeout=30 && \
    apt-get install -y -o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::Timeout=30 ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Установка vllm[audio] была отдельным шагом RUN ранее и успешно выполнилась или была в кеше.
# Если этот шаг был у вас в отдельной команде RUN ПЕРЕД apt-get,
# и apt-get все равно падал, то этот Dockerfile верен.
# Если apt-get был первым RUN после FROM, и он падал с этой ошибкой,
# то убедитесь, что установка vllm[audio] идет ПОСЛЕ этого шага в вашем полном Dockerfile.
# В стандартном подходе apt-get update/install идет до установки python-пакетов.

# Пример, если установка vllm[audio] должна идти после apt-get:
# RUN pip install --no-cache-dir --upgrade "vllm[audio]"

# Нет необходимости в CMD или ENTRYPOINT, так как они уже определены в базовом образе
# и команда передается через docker-compose.yml
