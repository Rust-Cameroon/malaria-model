# Build stage
FROM rust:1.90-bullseye AS builder

# Install system deps
RUN apt-get update && apt-get install -y pkg-config libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*

# Install wasm target and trunk
RUN rustup target add wasm32-unknown-unknown && \
    cargo install trunk --locked && \
    cargo install static-web-server --locked

WORKDIR /app

# Cache dependencies first (copy manifest files only)
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p /app/src && echo "fn main() {}" > /app/src/lib.rs && cargo fetch

# Copy full source
COPY . .

# Build release assets
RUN trunk build --release

# Runtime stage
FROM debian:bullseye-slim AS runtime

# Create a non-root user
RUN useradd -m appuser

# Copy built assets and server binary
COPY --from=builder /usr/local/cargo/bin/static-web-server /usr/local/bin/static-web-server
COPY --from=builder /app/dist /srv/www

WORKDIR /srv/www
USER appuser

# Render provides $PORT; default to 8080 locally
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "static-web-server --port ${PORT:-8080} --host 0.0.0.0 --root . --page404 index.html"]
