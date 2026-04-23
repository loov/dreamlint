# Indexes a Python project with scip-python to produce a SCIP index.
FROM --platform=linux/amd64 sourcegraph/scip-python:v0.6.6

WORKDIR /project

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
