FROM maven:3.9.11-eclipse-temurin-25 AS build
WORKDIR /workspace
COPY pom.xml .
RUN mvn -B -q -DskipTests dependency:go-offline
COPY src ./src
RUN mvn -B -DskipTests package

FROM eclipse-temurin:25-jre
WORKDIR /app
ENV PORT=10000
ENV JINFER_MODELS=/var/cache/crazeai/jinfer
ENV JAVA_TOOL_OPTIONS="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xms32m -Xmx128m -XX:MaxDirectMemorySize=128m -XX:+ExitOnOutOfMemoryError"
RUN mkdir -p /var/cache/crazeai/jinfer
COPY --from=build /workspace/target/crazeai-cloud.jar /app/crazeai-cloud.jar
EXPOSE 10000
CMD ["java","-jar","/app/crazeai-cloud.jar"]
