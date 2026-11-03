FROM ghcr.io/ggml-org/llama.cpp:server

WORKDIR /app
COPY docker-entrypoint.sh /usr/local/bin/crazeai-entrypoint
RUN chmod +x /usr/local/bin/crazeai-entrypoint

ENV LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_CTX_SIZE=1024 \
    LLAMA_ARG_N_PARALLEL=1 \
    LLAMA_ARG_THREADS=1 \
    LLAMA_ARG_BATCH=64 \
    LLAMA_ARG_UBATCH=32 \
    LLAMA_ARG_CACHE_TYPE_K=q8_0 \
    LLAMA_ARG_CACHE_TYPE_V=q8_0 \
    LLAMA_ARG_TEMP=0.2 \
    LLAMA_ARG_TOP_P=0.85 \
    LLAMA_ARG_TOP_K=20 \
    LLAMA_ARG_REPEAT_PENALTY=1.15 \
    LLAMA_ARG_REPEAT_LAST_N=128 \
    LLAMA_ARG_UI=false \
    LLAMA_ARG_ALIAS=Qwen/Qwen2.5-0.5B-Instruct

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/crazeai-entrypoint"]
