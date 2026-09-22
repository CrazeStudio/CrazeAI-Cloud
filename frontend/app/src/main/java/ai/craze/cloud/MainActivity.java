package ai.craze.cloud;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends Activity {
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(180, TimeUnit.SECONDS)
            .build();
    private final Handler handler = new Handler();
    private EditText server;
    private EditText message;
    private TextView status;
    private TextView answer;
    private Button send;

    @Override public void onCreate(Bundle b) {
        super.onCreate(b);
        buildUi();
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(28, 32, 28, 20);
        root.setBackgroundColor(Color.rgb(8, 8, 8));

        TextView title = text("CrazeAI Cloud", 26, Color.WHITE);
        root.addView(title);

        status = text("Connected to CrazeAI Cloud", 14, Color.LTGRAY);
        root.addView(status);

        server = field("Server URL");
        server.setText("https://crazeai-cloud.onrender.com");
        root.addView(server);

        message = field("Ask CrazeAI...");
        message.setMinLines(4);
        root.addView(message, new LinearLayout.LayoutParams(-1, 0, 1));

        send = new Button(this);
        send.setText("Send");
        send.setOnClickListener(v -> sendMessage());
        root.addView(send);

        answer = text("Your CrazeAI response will appear here.", 16, Color.WHITE);
        answer.setPadding(0, 24, 0, 0);
        root.addView(answer);

        setContentView(root);
    }

    private TextView text(String s, int size, int color) {
        TextView v = new TextView(this);
        v.setText(s);
        v.setTextSize(size);
        v.setTextColor(color);
        v.setPadding(0, 8, 0, 8);
        return v;
    }

    private EditText field(String hint) {
        EditText e = new EditText(this);
        e.setHint(hint);
        e.setHintTextColor(Color.GRAY);
        e.setTextColor(Color.WHITE);
        e.setTextSize(16);
        e.setSingleLine(false);
        return e;
    }

    private void sendMessage() {
        final String base = server.getText().toString().trim().replaceAll("/+$", "");
        final String msg = message.getText().toString().trim();
        if (base.isEmpty() || msg.isEmpty()) return;

        send.setEnabled(false);
        answer.setText("");
        status.setText("Checking CrazeAI...");
        waitForBackend(base, msg, 0);
    }

    private void waitForBackend(String base, String msg, int attempt) {
        Request r = new Request.Builder().url(base + "/health").get().build();
        client.newCall(r).enqueue(new Callback() {
            @Override public void onFailure(Call c, IOException e) {
                retryOrTimeout(base, msg, attempt);
            }

            @Override public void onResponse(Call c, Response r) throws IOException {
                String raw = r.body() == null ? "" : r.body().string();
                boolean ready = r.isSuccessful() && raw.contains("\"model\":\"ready\"");
                r.close();

                if (ready) {
                    runOnUiThread(() -> status.setText("Ready — sending..."));
                    callChat(base, msg);
                } else {
                    retryOrTimeout(base, msg, attempt);
                }
            }
        });
    }

    private void retryOrTimeout(String base, String msg, int attempt) {
        if (attempt >= 30) {
            runOnUiThread(() -> {
                status.setText("Timed out. Try again.");
                send.setEnabled(true);
            });
            return;
        }

        runOnUiThread(() -> status.setText("Waking CrazeAI... " + ((attempt + 1) * 2) + "s"));
        handler.postDelayed(() -> waitForBackend(base, msg, attempt + 1), 2000);
    }

    private void callChat(String base, String msg) {
        try {
            JSONObject body = new JSONObject();
            body.put("message", msg);

            Request r = new Request.Builder()
                    .url(base + "/api/chat")
                    .post(RequestBody.create(body.toString(), JSON))
                    .build();

            client.newCall(r).enqueue(new Callback() {
                @Override public void onFailure(Call c, IOException e) {
                    finish("Network error: " + e.getMessage());
                }

                @Override public void onResponse(Call c, Response r) throws IOException {
                    String raw = r.body() == null ? "" : r.body().string();
                    try {
                        JSONObject obj = new JSONObject(raw);
                        finish(obj.optString("response", obj.optString("detail", raw)));
                    } catch (Exception e) {
                        finish(raw);
                    } finally {
                        r.close();
                    }
                }
            });
        } catch (Exception e) {
            finish(e.getMessage());
        }
    }

    private void finish(String result) {
        runOnUiThread(() -> {
            answer.setText(result);
            status.setText("Ready");
            send.setEnabled(true);
        });
    }
}
