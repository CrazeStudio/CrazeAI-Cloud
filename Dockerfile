FROM gradle:8.10.2-jdk21 AS build
WORKDIR /workspace
COPY . .
RUN gradle --no-daemon clean build -x test
FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /workspace/build/libs/crazeai-cloud-1.0.0.jar /app/app.jar
ENV PORT=8080
ENV MODEL_DIR=/var/data/models
ENV JAVA_TOOL_OPTIONS="--enable-preview --add-modules=jdk.incubator.vector -XX:MaxRAMPercentage=75.0"
EXPOSE 8080
CMD ["java", "-jar", "/app/app.jar"]
