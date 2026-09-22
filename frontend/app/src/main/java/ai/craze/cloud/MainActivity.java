package ai.craze.cloud;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.view.Gravity;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
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
    private static final int BG = Color.rgb(10,10,10);
    private static final int SURFACE = Color.rgb(20,20,20);
    private static final int USER = Color.rgb(43,43,43);
    private static final int ORANGE = Color.rgb(255,138,0);
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    private final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(180, TimeUnit.SECONDS).build();
    private final Handler handler = new Handler();

    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private TextView status;
    private View sendButton;

    @Override public void onCreate(Bundle b) {
        super.onCreate(b);
        buildUi();
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);

        // Header
        LinearLayout header = new LinearLayout(this);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(18, 14, 18, 12);
        header.setBackgroundColor(BG);

        ImageView logo = new ImageView(this);
        logo.setImageResource(ai.craze.cloud.R.drawable.ic_crazeai_logo);
        header.addView(logo, new LinearLayout.LayoutParams(42,42));

        LinearLayout names = new LinearLayout(this);
        names.setOrientation(LinearLayout.VERTICAL);
        names.setPadding(12,0,0,0);
        TextView title = label("CrazeAI", 19, Color.WHITE);
        status = label("Cloud • Ready", 12, Color.LTGRAY);
        names.addView(title);
        names.addView(status);
        header.addView(names, new LinearLayout.LayoutParams(0,-2,1));
        root.addView(header);

        // Chat area
        scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(16,12,16,18);
        scroll.addView(messages);

        addBotMessage("Hi! I'm CrazeAI. What can I help you with?");
        root.addView(scroll, new LinearLayout.LayoutParams(-1,0,1));

        // Composer
        LinearLayout composer = new LinearLayout(this);
        composer.setGravity(Gravity.CENTER_VERTICAL);
        composer.setPadding(10,8,10,10);
        composer.setBackgroundColor(BG);

        input = new EditText(this);
        input.setHint("Message CrazeAI...");
        input.setHintTextColor(Color.rgb(130,130,130));
        input.setTextColor(Color.WHITE);
        input.setTextSize(16);
        input.setSingleLine(false);
        input.setMaxLines(5);
        input.setPadding(18,10,14,10);
        input.setBackground(round(SURFACE, 28));
        input.setImeOptions(EditorInfo.IME_ACTION_SEND);
        input.setOnEditorActionListener((v, action, event) -> {
            if (action == EditorInfo.IME_ACTION_SEND) { sendMessage(); return true; }
            return false;
        });
        composer.addView(input, new LinearLayout.LayoutParams(0,-2,1));

        TextView send = label("➤", 22, Color.WHITE);
        send.setGravity(Gravity.CENTER);
        send.setBackground(round(ORANGE, 50));
        send.setPadding(0,0,0,2);
        sendButton = send;
        send.setOnClickListener(v -> sendMessage());
        LinearLayout.LayoutParams sp = new LinearLayout.LayoutParams(52,52);
        sp.setMargins(8,0,0,0);
        composer.addView(send, sp);

        root.addView(composer);
        setContentView(root);
    }

    private TextView label(String s, int size, int color) {
        TextView v = new TextView(this);
        v.setText(s); v.setTextSize(size); v.setTextColor(color);
        return v;
    }

    private GradientDrawable round(int color, float radius) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(color); g.setCornerRadius(radius);
        return g;
    }

    private void addBotMessage(String text) { addBubble(text, false); }
    private void addUserMessage(String text) { addBubble(text, true); }

    private void addBubble(String text, boolean user) {
        TextView bubble = label(text, 16, Color.WHITE);
        bubble.setTextIsSelectable(true);
        bubble.setPadding(16,12,16,12);
        bubble.setBackground(round(user ? USER : SURFACE, 22));

        LinearLayout row = new LinearLayout(this);
        row.setGravity(user ? Gravity.RIGHT : Gravity.LEFT);
        row.setPadding(0,5,0,5);
        row.addView(bubble, new LinearLayout.LayoutParams(
                user ? -2 : -1, -2));
        messages.addView(row);
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private void setBusy(boolean busy) {
        sendButton.setEnabled(!busy);
        sendButton.setAlpha(busy ? 0.45f : 1f);
        input.setEnabled(!busy);
    }

    private void sendMessage() {
        final String msg = input.getText().toString().trim();
        if (msg.isEmpty()) return;

        input.setText("");
        addUserMessage(msg);
        addBotMessage("Thinking…");
        final int thinkingIndex = messages.getChildCount() - 1;

        setBusy(true);
        status.setText("Cloud • Connecting...");

        waitForBackend("https://crazeai-cloud.onrender.com", msg, 0, thinkingIndex);
    }

    private void waitForBackend(String base, String msg, int attempt, int thinkingIndex) {
        Request r = new Request.Builder().url(base + "/health").get().build();
        client.newCall(r).enqueue(new Callback() {
            @Override public void onFailure(Call c, IOException e) {
                retry(base,msg,attempt,thinkingIndex);
            }
            @Override public void onResponse(Call c, Response r) throws IOException {
                String raw = r.body() == null ? "" : r.body().string();
                boolean ready = r.isSuccessful() && raw.contains("\"model\":\"ready\"");
                r.close();
                if (ready) {
                    runOnUiThread(() -> status.setText("Cloud • Online"));
                    callChat(base,msg,thinkingIndex);
                } else retry(base,msg,attempt,thinkingIndex);
            }
        });
    }

    private void retry(String base,String msg,int attempt,int thinkingIndex) {
        if (attempt >= 30) {
            replaceBubble(thinkingIndex, "The server took too long to wake up. Please try again.");
            runOnUiThread(() -> { status.setText("Cloud • Offline"); setBusy(false); });
            return;
        }
        runOnUiThread(() -> status.setText("Cloud • Waking model... " + ((attempt+1)*2) + "s"));
        handler.postDelayed(() -> waitForBackend(base,msg,attempt+1,thinkingIndex),2000);
    }

    private void callChat(String base,String msg,int thinkingIndex) {
        try {
            JSONObject body = new JSONObject();
            body.put("message",msg);
            Request r = new Request.Builder().url(base+"/api/chat")
                    .post(RequestBody.create(body.toString(),JSON)).build();

            client.newCall(r).enqueue(new Callback() {
                @Override public void onFailure(Call c, IOException e) {
                    replaceBubble(thinkingIndex,"Network error: "+e.getMessage());
                    runOnUiThread(() -> { status.setText("Cloud • Error"); setBusy(false); });
                }
                @Override public void onResponse(Call c, Response r) throws IOException {
                    String raw = r.body()==null ? "" : r.body().string();
                    String result;
                    try {
                        JSONObject obj = new JSONObject(raw);
                        result = obj.optString("response",obj.optString("detail",raw));
                    } catch(Exception e) { result=raw; }
                    r.close();
                    replaceBubble(thinkingIndex,result);
                    runOnUiThread(() -> { status.setText("Cloud • Online"); setBusy(false); });
                }
            });
        } catch(Exception e) {
            replaceBubble(thinkingIndex,e.getMessage());
            setBusy(false);
        }
    }

    private void replaceBubble(int index,String text) {
        runOnUiThread(() -> {
            if(index >= 0 && index < messages.getChildCount()) {
                View row=messages.getChildAt(index);
                if(row instanceof LinearLayout && ((LinearLayout)row).getChildCount()>0) {
                    ((TextView)((LinearLayout)row).getChildAt(0)).setText(text);
                }
                scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
            }
        });
    }
}
