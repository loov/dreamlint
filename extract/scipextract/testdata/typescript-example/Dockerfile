# Indexes a tiny TypeScript project with @sourcegraph/scip-typescript.
# No platform pin — the node image is multi-arch and scip-typescript is
# pure JS, so it runs natively on any host architecture.
FROM node:22-bookworm-slim

WORKDIR /project

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
