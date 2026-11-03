package com.crazestudio.crazeai;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.JinferChatOptions;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

public final class Application {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String MODEL_ID = "crazeai-smollm2-360m";
    private static final String MODEL_REF = System.getenv().getOrDefault(
            "CRAZEAI_MODEL", "tensorblock/SmolLM2-360M-Instruct-GGUF:Q4_0");
    private static final int PORT = Integer.parseInt(System.getenv().getOrDefault("PORT", "10000"));
    private static final int CONTEXT = integerEnv("CRAZEAI_CONTEXT", 256, 128, 2048);
    private static final int DEFAULT_MAX_TOKENS = integerEnv("CRAZEAI_MAX_TOKENS", 192, 1, 512);
    private static final String API_KEY = System.getenv().getOrDefault("CRAZEAI_API_KEY", "");
    private static final AtomicReference<JinferChatModel> MODEL = new AtomicReference<>();
    private static final AtomicReference<String> MODEL_ERROR = new AtomicReference<>();
    private static final Object INFERENCE_LOCK = new Object();

    private Application() {}

    public static void main(String[] args) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", PORT), 0);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.createContext("/", Application::route);
        server.start();
        System.out.println("[CrazeAI] HTTP server listening on " + PORT);
        System.out.println("[CrazeAI] Model: " + MODEL_REF + " | context=" + CONTEXT);
        CompletableFuture.runAsync(Application::loadModel);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            JinferChatModel model = MODEL.getAndSet(null);
            if (model != null) {
                try { model.close(); } catch (Exception ignored) {}
            }
            server.stop(2);
        }));
    }

    private static void loadModel() {
        if (MODEL.get() != null) return;
        try {
            System.out.println("[CrazeAI] Loading GGUF into the JVM. First boot downloads the model.");
            JinferChatModel model = JinferChatModel.builder()
                    .model(MODEL_REF)
                    .contextCapacity(CONTEXT)
                    .options(JinferChatOptions.builder().maxTokens(DEFAULT_MAX_TOKENS).temperature(0.65).topP(0.9).build())
                    .build();
            MODEL.set(model);
            MODEL_ERROR.set(null);
            System.out.println("[CrazeAI] Model is ready.");
        } catch (Throwable error) {
            MODEL_ERROR.set(error.getClass().getSimpleName() + ": " + safeMessage(error));
            System.err.println("[CrazeAI] Model load failed: " + MODEL_ERROR.get());
            error.printStackTrace(System.err);
        }
    }

    private static void route(HttpExchange exchange) throws IOException {
        addCors(exchange.getResponseHeaders());
        String path = exchange.getRequestURI().getPath();
        String method = exchange.getRequestMethod();
        if ("OPTIONS".equalsIgnoreCase(method)) {
            exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Authorization, Content-Type");
            send(exchange, 204, "");
            return;
        }
        try {
            if ("GET".equals(method) && "/".equals(path)) {
                sendJson(exchange, 200, Map.of("name", "CrazeAI Cloud", "runtime", "Java 25", "engine", "Qxotic jinfer", "model", MODEL_ID));
            } else if ("GET".equals(method) && "/health".equals(path)) {
                Map<String, Object> body = new LinkedHashMap<>();
                body.put("status", "ok");
                body.put("service", "crazeai-cloud");
                body.put("runtime", Runtime.version().toString());
                body.put("engine", "jinfer");
                body.put("model", MODEL_ID);
                body.put("model_loaded", MODEL.get() != null);
                body.put("model_error", MODEL_ERROR.get());
                sendJson(exchange, 200, body);
            } else if ("GET".equals(method) && "/ready".equals(path)) {
                if (MODEL.get() == null) {
                    sendJson(exchange, 503, Map.of("status", "loading", "model_loaded", false,
                            "error", MODEL_ERROR.get() == null ? "" : MODEL_ERROR.get()));
                } else {
                    sendJson(exchange, 200, Map.of("status", "ready", "model_loaded", true, "model", MODEL_ID));
                }
            } else if ("GET".equals(method) && "/v1/models".equals(path)) {
                if (!authorized(exchange)) { sendUnauthorized(exchange); return; }
                sendJson(exchange, 200, Map.of("object", "list", "data", List.of(
                        Map.of("id", MODEL_ID, "object", "model", "created", Instant.now().getEpochSecond(),
                                "owned_by", "CrazeStudio"))));
            } else if ("POST".equals(method) && "/v1/chat/completions".equals(path)) {
                if (!authorized(exchange)) { sendUnauthorized(exchange); return; }
                handleCompletion(exchange);
            } else {
                sendJson(exchange, 404, Map.of("error", Map.of("message", "Route not found", "type", "not_found_error")));
            }
        } catch (RequestException e) {
            sendJson(exchange, e.status, Map.of("error", Map.of("message", e.getMessage(), "type", "invalid_request_error")));
        } catch (Throwable e) {
            System.err.println("[CrazeAI] Request error: " + safeMessage(e));
            if (exchange.getResponseCode() == -1) {
                sendJson(exchange, 500, Map.of("error", Map.of("message", "Internal server error", "type", "server_error")));
            } else {
                exchange.close();
            }
        }
    }

    private static void handleCompletion(HttpExchange exchange) throws Exception {
        byte[] body;
        try (InputStream in = exchange.getRequestBody()) { body = in.readNBytes(1_048_577); }
        if (body.length > 1_048_576) throw new RequestException(413, "Request body exceeds 1 MB");
        Map<String, Object> request = JSON.readValue(body, new TypeReference<>() {});
        Object messagesValue = request.get("messages");
        if (!(messagesValue instanceof List<?> rawMessages) || rawMessages.isEmpty()) {
            throw new RequestException(400, "messages must be a non-empty array");
        }
        List<Message> messages = new ArrayList<>();
        for (Object item : rawMessages) {
            if (!(item instanceof Map<?, ?> message)) throw new RequestException(400, "Each message must be an object");
            Object roleValue = message.get("role");
            String role = roleValue == null ? "" : roleValue.toString();
            Object contentValue = message.get("content");
            if (!(contentValue instanceof String content)) throw new RequestException(400, "Message content must be a string");
            if (content.length() > 32_000) throw new RequestException(400, "A message is too long");
            switch (role) {
                case "system" -> messages.add(new SystemMessage(content));
                case "user" -> messages.add(new UserMessage(content));
                case "assistant" -> messages.add(new AssistantMessage(content));
                default -> throw new RequestException(400, "Unsupported message role: " + role);
            }
        }
        if (MODEL.get() == null) {
            if (MODEL_ERROR.get() != null) throw new RequestException(503, "Model failed to load: " + MODEL_ERROR.get());
            throw new RequestException(503, "Model is still loading. Retry shortly.");
        }
        int maxTokens = request.containsKey("max_tokens")
                ? clampNumber(request.get("max_tokens"), 1, 512, DEFAULT_MAX_TOKENS) : DEFAULT_MAX_TOKENS;
        boolean streaming = Boolean.TRUE.equals(request.get("stream"));
        Prompt prompt = new Prompt(messages, JinferChatOptions.builder()
                .maxTokens(maxTokens)
                .temperature(number(request.get("temperature"), 0.65))
                .topP(number(request.get("top_p"), 0.9))
                .build());
        synchronized (INFERENCE_LOCK) {
            if (streaming) streamCompletion(exchange, prompt, maxTokens);
            else normalCompletion(exchange, prompt, maxTokens);
        }
    }

    private static void normalCompletion(HttpExchange exchange, Prompt prompt, int maxTokens) throws IOException {
        ChatResponse response = MODEL.get().call(prompt);
        String content = response == null || response.getResult() == null
                ? "" : response.getResult().getOutput().getText();
        if (content == null) content = "";
        Map<String, Object> choice = Map.of("index", 0, "message", Map.of("role", "assistant", "content", content),
                "finish_reason", "stop");
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("id", "chatcmpl-" + UUID.randomUUID());
        result.put("object", "chat.completion");
        result.put("created", Instant.now().getEpochSecond());
        result.put("model", MODEL_ID);
        result.put("choices", List.of(choice));
        sendJson(exchange, 200, result);
    }

    private static void streamCompletion(HttpExchange exchange, Prompt prompt, int maxTokens) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "text/event-stream; charset=utf-8");
        headers.set("Cache-Control", "no-cache");
        headers.set("Connection", "keep-alive");
        headers.set("X-Accel-Buffering", "no");
        exchange.sendResponseHeaders(200, 0);
        String id = "chatcmpl-" + UUID.randomUUID();
        try (OutputStream out = exchange.getResponseBody(); Stream<ChatResponse> chunks = MODEL.get().stream(prompt).toStream()) {
            chunks.forEach(chunk -> {
                try {
                    String text = chunk == null || chunk.getResult() == null ? "" : chunk.getResult().getOutput().getText();
                    if (text == null || text.isEmpty()) return;
                    Map<String, Object> delta = Map.of("role", "assistant", "content", text);
                    Map<String, Object> data = Map.of("id", id, "object", "chat.completion.chunk",
                            "created", Instant.now().getEpochSecond(), "model", MODEL_ID,
                            "choices", List.of(Map.of("index", 0, "delta", delta)));
                    out.write(("data: " + JSON.writeValueAsString(data) + "\n\n").getBytes(StandardCharsets.UTF_8));
                    out.flush();
                } catch (IOException e) { throw new RuntimeException(e); }
            });
            out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
            out.flush();
        } catch (RuntimeException e) {
            System.err.println("[CrazeAI] Stream interrupted: " + safeMessage(e));
        } finally {
            exchange.close();
        }
    }

    private static boolean authorized(HttpExchange exchange) {
        if (API_KEY.isBlank()) return true;
        String header = exchange.getRequestHeaders().getFirst("Authorization");
        if (header == null || !header.startsWith("Bearer ")) return false;
        return MessageDigest.isEqual(API_KEY.getBytes(StandardCharsets.UTF_8),
                header.substring(7).getBytes(StandardCharsets.UTF_8));
    }

    private static void sendUnauthorized(HttpExchange exchange) throws IOException {
        sendJson(exchange, 401, Map.of("error", Map.of("message", "Invalid or missing bearer API key", "type", "authentication_error")));
    }

    private static void sendJson(HttpExchange exchange, int status, Object value) throws IOException {
        send(exchange, status, JSON.writeValueAsString(value));
    }

    private static void send(HttpExchange exchange, int status, String value) throws IOException {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.getResponseHeaders().set("Cache-Control", "no-store");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) { out.write(bytes); }
    }

    private static void addCors(Headers headers) {
        headers.set("Access-Control-Allow-Origin", "*");
        headers.set("Vary", "Origin");
    }

    private static int integerEnv(String name, int fallback, int min, int max) {
        try { return Math.max(min, Math.min(max, Integer.parseInt(System.getenv().getOrDefault(name, String.valueOf(fallback))))); }
        catch (NumberFormatException ignored) { return fallback; }
    }

    private static int clampNumber(Object value, int min, int max, int fallback) {
        try { return Math.max(min, Math.min(max, ((Number) value).intValue())); }
        catch (Exception ignored) { return fallback; }
    }

    private static double number(Object value, double fallback) {
        try { return value instanceof Number n ? Math.max(0.0, Math.min(2.0, n.doubleValue())) : fallback; }
        catch (Exception ignored) { return fallback; }
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.isBlank() ? error.getClass().getSimpleName() : message;
    }

    private static final class RequestException extends Exception {
        final int status;
        RequestException(int status, String message) { super(message); this.status = status; }
    }
}
