# syntax=docker/dockerfile:1
# Web UI by default: docker run -p 127.0.0.1:7860:7860 -v "$PWD:/data" <image>
# CLI:               docker run -v "$PWD:/data" <image> run photo.jpg -o bin.stl

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first, so source changes don't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --extra ui --no-install-project

COPY pyproject.toml uv.lock LICENSE ./
COPY gridfinity_shadowbox ./gridfinity_shadowbox
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ui --no-editable


# Same /usr/local Python as the builder image, so the venv's interpreter link stays valid.
FROM python:3.12-slim-bookworm

RUN groupadd --gid 1000 shadowbox \
    && useradd --uid 1000 --gid 1000 --create-home shadowbox \
    && mkdir /data \
    && chown shadowbox:shadowbox /data

COPY --from=builder /app/.venv /app/.venv

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SHADOWBOX_UI_HOST=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False

# The UI exports into its working directory; mount a host folder here.
WORKDIR /data
USER shadowbox
EXPOSE 7860

ENTRYPOINT ["shadowbox"]
CMD ["ui", "--no-browser"]
