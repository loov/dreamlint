# Indexes a Go package with scip-go to produce a SCIP index.
FROM golang:1.25-bookworm

ARG SCIP_GO_VERSION=v0.2.3
RUN go install github.com/scip-code/scip-go/cmd/scip-go@${SCIP_GO_VERSION}

WORKDIR /project

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
