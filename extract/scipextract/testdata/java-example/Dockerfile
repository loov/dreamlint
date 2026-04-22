# Indexes a Maven project with scip-java to produce a SCIP index.
FROM maven:3.9-eclipse-temurin-17-focal

ARG SCIP_JAVA_VERSION=0.12.3
RUN curl -fsSL -o /usr/local/bin/scip-java \
        "https://github.com/sourcegraph/scip-java/releases/download/v${SCIP_JAVA_VERSION}/scip-java-v${SCIP_JAVA_VERSION}" \
    && chmod +x /usr/local/bin/scip-java

WORKDIR /project

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
