FROM ubuntu:noble-20241118.1 AS builder

ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    mold && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /app
COPY src/ src/
COPY Cargo.toml Cargo.lock ./

RUN mkdir -p .cargo && echo '[target.x86_64-unknown-linux-gnu]\nlinker = "clang"\nrustflags = ["-C", "link-arg=-fuse-ld=/usr/bin/mold"]' > .cargo/config.toml

RUN cargo build --release

FROM ubuntu:noble-20241118.1 AS runtime

RUN apt-get update && apt-get install -y ca-certificates

COPY --from=builder /app/target/release/proxy /usr/local/bin/proxy

ENTRYPOINT ["/usr/local/bin/proxy"]
