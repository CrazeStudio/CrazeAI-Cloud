FROM gradle:8.10.2-jdk21 AS build
WORKDIR /workspace
COPY . .
RUN gradle --no-daemon clean installDist -x test

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /workspace/build/install/crazeai-cloud/ /app/
ENV PORT=8080
ENV MODEL_DIR=/var/data/models
ENV JAVA_TOOL_OPTIONS="--enable-preview --add-modules=jdk.incubator.vector -XX:MaxRAMPercentage=75.0"
EXPOSE 8080
CMD ["/app/bin/crazeai-cloud"]
