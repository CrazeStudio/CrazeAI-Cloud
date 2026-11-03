package studio.crazeai.cloud;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.util.Downloader;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

public final class CrazeAiApplication {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String MODEL_ID = env("MODEL_ID", "HuggingFaceTB/SmolLM2-360M-Instruct");
    private static final String API_KEY = env("API_KEY", "");
    private static final int MAX_TOKENS = intEnv("MAX_TOKENS", 256);
    private static final Semaphore INFERENCE_SLOT = new Semaphore(1);
    private static volatile AbstractModel model;

    private CrazeAiApplication() {}

    public static void main(String[] args) throws Exception {
        if (API_KEY.isBlank()) throw new IllegalStateException("API_KEY is required. Set a long random secret in Render.");
        loadModel();
        int port = intEnv("PORT", 8080);
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", port), 0);
        server.createContext("/health", CrazeAiApplication::health);
        server.createContext("/v1/models", CrazeAiApplication::models);
        server.createContext("/v1/chat/completions", CrazeAiApplication::chatCompletions);
        server.setExecutor(Executors.newFixedThreadPool(4));
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try { server.stop(2); } catch (Exception ignored) {}
            AbstractModel current = model;
            if (current != null) current.close();
        }));
        server.start();
        System.out.println("CrazeAI Cloud listening on port " + port + " with model " + MODEL_ID);
    }

    private static void loadModel() throws Exception {
        String modelDir = env("MODEL_DIR", "./models");
        System.out.println("Downloading/checking Hugging Face model: " + MODEL_ID);
        Downloader downloader = new Downloader(modelDir, MODEL_ID);
        String hfToken = env("HF_TOKEN", "");
        if (!hfToken.isBlank()) downloader.withAuthToken(hfToken);
        java.io.File modelPath = downloader.huggingFaceModel();
        DType quantization = DType.valueOf(env("MODEL_QUANTIZATION", "Q4").toUpperCase());
        System.out.println("Loading Jlama model; model quantization=" + quantization);
        model = ModelSupport.loadModel(modelPath, null, DType.F32, DType.I8,
            java.util.Optional.of(quantization), java.util.Optional.empty());
        System.out.println("Model loaded: " + MODEL_ID + " (" + model.getModelType() + ")");
    }

    private static void health(HttpExchange exchange) throws IOException {
        if (!"GET".equals(exchange.getRequestMethod())) { methodNotAllowed(exchange); return; }
        sendJson(exchange, model == null ? 503 : 200, Map.of(
            "status", model == null ? "loading" : "ok", "service", "CrazeAI Cloud",
            "engine", "Jlama", "model", MODEL_ID, "model_loaded", model != null));
    }

    private static void models(HttpExchange exchange) throws IOException {
        if (!"GET".equals(exchange.getRequestMethod())) { methodNotAllowed(exchange); return; }
        if (!authorized(exchange)) { unauthorized(exchange); return; }
        sendJson(exchange, 200, Map.of("object", "list", "data", List.of(Map.of(
            "id", MODEL_ID, "object", "model", "created", Instant.now().getEpochSecond(), "owned_by", "crazestudio"))));
    }

    private static void chatCompletions(HttpExchange exchange) throws IOException {
        if (!"POST".equals(exchange.getRequestMethod())) { methodNotAllowed(exchange); return; }
        if (!authorized(exchange)) { unauthorized(exchange); return; }
        if (!INFERENCE_SLOT.tryAcquire()) { sendError(exchange, 429, "The model is busy. Retry shortly.", "rate_limit_error"); return; }
        try {
            JsonNode request = readJson(exchange.getRequestBody());
            JsonNode messages = request.path("messages");
            if (!messages.isArray() || messages.isEmpty()) { sendError(exchange, 400, "messages must be a non-empty array", "invalid_request_error"); return; }
            String requestedModel = request.path("model").asText(MODEL_ID);
            if (!requestedModel.equals(MODEL_ID)) { sendError(exchange, 404, "Model not found: " + requestedModel, "model_not_found"); return; }
            float temperature = (float) request.path("temperature").asDouble(0.7);
            if (temperature < 0 || temperature > 2) { sendError(exchange, 400, "temperature must be between 0 and 2", "invalid_request_error"); return; }
            int maxTokens = Math.max(1, Math.min(request.path("max_tokens").asInt(MAX_TOKENS), MAX_TOKENS));
            PromptContext prompt = buildPrompt(messages);
            boolean stream = request.path("stream").asBoolean(false);
            String id = "chatcmpl-" + UUID.randomUUID().toString().replace("-", "");
            long created = Instant.now().getEpochSecond();
            if (stream) {
                streamCompletion(exchange, id, created, temperature, maxTokens, prompt);
            } else {
                StringBuilder output = new StringBuilder();
                model.generate(UUID.randomUUID(), prompt, temperature, maxTokens, (token, timing) -> output.append(token));
                Map<String, Object> choice = Map.of("index", 0, "message", Map.of("role", "assistant", "content", output.toString()), "finish_reason", "stop");
                sendJson(exchange, 200, Map.of("id", id, "object", "chat.completion", "created", created,
                    "model", MODEL_ID, "choices", List.of(choice),
                    "usage", Map.of("prompt_tokens", 0, "completion_tokens", 0, "total_tokens", 0)));
            }
        } catch (IllegalArgumentException e) {
            sendError(exchange, 400, safeMessage(e), "invalid_request_error");
        } catch (Exception e) {
            e.printStackTrace(System.err);
            if (exchange.getResponseCode() == -1) sendError(exchange, 500, "Inference failed: " + safeMessage(e), "server_error");
        } finally { INFERENCE_SLOT.release(); }
    }

    private static PromptContext buildPrompt(JsonNode messages) {
        if (model.promptSupport().isPresent()) {
            var builder = model.promptSupport().get().builder();
            for (JsonNode message : messages) {
                String role = message.path("role").asText("user").toLowerCase();
                String content = message.path("content").asText("");
                switch (role) {
                    case "system", "developer" -> builder.addSystemMessage(content);
                    case "assistant" -> builder.addAssistantMessage(content);
                    case "user" -> builder.addUserMessage(content);
                    default -> throw new IllegalArgumentException("Unsupported message role: " + role);
                }
            }
            return builder.build();
        }
        StringBuilder prompt = new StringBuilder();
        for (JsonNode message : messages) prompt.append(message.path("role").asText("user")).append(": ").append(message.path("content").asText("")).append('\n');
        return PromptContext.of(prompt.toString());
    }

    private static void streamCompletion(HttpExchange exchange, String id, long created, float temperature, int maxTokens, PromptContext prompt) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "text/event-stream; charset=utf-8");
        headers.set("Cache-Control", "no-cache, no-transform");
        headers.set("Connection", "keep-alive");
        headers.set("X-Accel-Buffering", "no");
        exchange.sendResponseHeaders(200, 0);
        try (OutputStream out = exchange.getResponseBody()) {
            model.generate(UUID.randomUUID(), prompt, temperature, maxTokens, (token, timing) -> {
                try {
                    Map<String, Object> choice = new LinkedHashMap<>();
                    choice.put("index", 0); choice.put("delta", Map.of("content", token)); choice.put("finish_reason", null);
                    Map<String, Object> chunk = Map.of("id", id, "object", "chat.completion.chunk", "created", created, "model", MODEL_ID, "choices", List.of(choice));
                    out.write(("data: " + JSON.writeValueAsString(chunk) + "\n\n").getBytes(StandardCharsets.UTF_8));
                    out.flush();
                } catch (IOException io) { throw new java.io.UncheckedIOException(io); }
            });
            Map<String, Object> finalChoice = Map.of("index", 0, "delta", Map.of(), "finish_reason", "stop");
            out.write(("data: " + JSON.writeValueAsString(Map.of("id", id, "object", "chat.completion.chunk", "created", created, "model", MODEL_ID, "choices", List.of(finalChoice))) + "\n\n").getBytes(StandardCharsets.UTF_8));
            out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
            out.flush();
        } catch (java.io.UncheckedIOException e) {
            System.err.println("Client disconnected during streaming: " + e.getMessage());
        } finally { exchange.close(); }
    }

    private static JsonNode readJson(InputStream input) throws IOException {
        byte[] bytes = input.readNBytes(1_048_577);
        if (bytes.length > 1_048_576) throw new IllegalArgumentException("Request body exceeds 1 MB");
        return JSON.readTree(bytes);
    }

    private static boolean authorized(HttpExchange exchange) {
        String header = exchange.getRequestHeaders().getFirst("Authorization");
        if (header == null || !header.startsWith("Bearer ")) return false;
        return MessageDigest.isEqual(header.substring(7).getBytes(StandardCharsets.UTF_8), API_KEY.getBytes(StandardCharsets.UTF_8));
    }
    private static void unauthorized(HttpExchange e) throws IOException { sendError(e, 401, "Missing or invalid bearer API key", "authentication_error"); }
    private static void methodNotAllowed(HttpExchange e) throws IOException { e.getResponseHeaders().set("Allow", "GET, POST"); sendError(e, 405, "Method not allowed", "method_not_allowed"); }
    private static void sendError(HttpExchange e, int status, String message, String type) throws IOException {
        sendJson(e, status, Map.of("error", Map.of("message", message, "type", type, "param", "", "code", status)));
    }
    private static void sendJson(HttpExchange e, int status, Object value) throws IOException {
        byte[] body = JSON.writeValueAsBytes(value);
        e.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        e.getResponseHeaders().set("Cache-Control", "no-store");
        e.sendResponseHeaders(status, body.length);
        try (OutputStream out = e.getResponseBody()) { out.write(body); }
    }
    private static String safeMessage(Exception e) {
        String m = e.getMessage(); return m == null || m.isBlank() ? e.getClass().getSimpleName() : m.substring(0, Math.min(m.length(), 300));
    }
    private static String env(String key, String fallback) { String v = System.getenv(key); return v == null || v.isBlank() ? fallback : v.trim(); }
    private static int intEnv(String key, int fallback) { try { return Integer.parseInt(env(key, Integer.toString(fallback))); } catch (NumberFormatException e) { return fallback; } }
}
