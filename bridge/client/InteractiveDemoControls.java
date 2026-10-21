package communicationmod;

import basemod.BaseMod;
import basemod.interfaces.PreUpdateSubscriber;
import basemod.interfaces.PostRenderSubscriber;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.megacrit.cardcrawl.core.Settings;
import com.megacrit.cardcrawl.helpers.FontHelper;
import com.megacrit.cardcrawl.helpers.Hitbox;
import com.megacrit.cardcrawl.helpers.ImageMaster;
import com.megacrit.cardcrawl.helpers.SeedHelper;
import com.megacrit.cardcrawl.helpers.input.InputHelper;
import java.security.SecureRandom;
import java.util.HashMap;
import java.util.Map;

/** Explicit human control only. Does not execute gameplay commands or touch game RNG. */
public final class InteractiveDemoControls implements PreUpdateSubscriber, PostRenderSubscriber {
    public static final int VERSION = 1;
    private static InteractiveDemoControls instance;
    private static boolean connected, enabled;
    private static long requestId;
    private static String action = "NONE", seed = "", status = "LOADING", detail = "";
    private static String model = "等待模型握手";
    private static final SecureRandom random = new SecureRandom();
    private final Hitbox button = new Hitbox(290f * Settings.scale, 46f * Settings.scale);

    public static void install() {
        if (instance == null) {
            instance = new InteractiveDemoControls();
            BaseMod.subscribe(instance);
        }
    }

    public static void connection(boolean alive) {
        if (!alive && connected) {
            connected = enabled = false;
            status = "ERROR";
            detail = "模型进程已退出，请查看 bridge/client/agent_orchestrator.log";
        }
    }

    public static void augment(Map frame) {
        Map<String, Object> controls = new HashMap<String, Object>();
        controls.put("version", VERSION);
        controls.put("request_id", requestId);
        controls.put("action", action);
        controls.put("seed", seed);
        controls.put("enabled", enabled);
        frame.put("demo_controls", controls);
    }

    /** Dedicated UI receipt, intercepted before the ordinary gameplay dispatcher. */
    public static boolean acceptStatus(String command) {
        if (!command.startsWith("DEMO ")) return false;
        String[] parts = command.split(" ", 3);
        if (parts.length < 2 || !(parts[1].equals("READY") || parts[1].equals("RUNNING")
                || parts[1].equals("PAUSED") || parts[1].equals("STOPPED")
                || parts[1].equals("FINISHED") || parts[1].equals("ERROR")))
            throw new IllegalArgumentException("Unknown demo UI receipt");
        status = parts[1];
        connected = true;
        detail = parts.length == 3 ? parts[2] : "";
        if (status.equals("READY")) model = detail.equals("train47-step-000688914432-eval31")
                ? "TRAIN47 FINAL 688914432" : detail;
        if (status.equals("ERROR") || status.equals("STOPPED") || status.equals("FINISHED"))
            enabled = false;
        CommunicationMod.mustSendGameState = true;
        return true;
    }

    private static boolean inGame() { return CommandExecutor.isInDungeon(); }

    private float centerX() { return Settings.WIDTH - 175f * Settings.scale; }
    private float centerY() { return inGame() ? Settings.HEIGHT - 110f * Settings.scale : Settings.HEIGHT * .52f; }

    public void receivePreUpdate() {
        if (BaseMod.modSettingsUp || (!inGame() && !CommandExecutor.isStartCommandAvailable())) return;
        button.move(centerX(), centerY());
        button.update();
        if (!connected || !button.hovered || !InputHelper.justClickedLeft) return;
        InputHelper.justClickedLeft = false; // Do not also click a card/menu item below the overlay.
        if (inGame()) {
            enabled = !enabled;
            if (enabled) {
                requestId++;
                action = "ATTACH";
                seed = "";
                status = "RUNNING";
                detail = "等待接管当前画面";
            } else {
                status = "PAUSED";
                detail = "已发出的单个动作可能仍在结算";
            }
        } else if (!enabled) {
            requestId++;
            action = "START";
            seed = SeedHelper.getString(random.nextLong() & Long.MAX_VALUE);
            enabled = true;
            status = "RUNNING";
            detail = "请求新种子 " + seed;
        }
        CommunicationMod.mustSendGameState = true;
    }

    private static String statusText() {
        if (status.equals("READY")) return "模型就绪 · 等待点击";
        if (status.equals("RUNNING")) return enabled ? "AUTO 开启 · 模型操作中" : "手动操作";
        if (status.equals("PAUSED")) return "AUTO 关闭 · 手动操作";
        if (status.equals("FINISHED")) return "本局演示结束 · 不会自动再开";
        if (status.equals("STOPPED")) return "已停止 · 点击 AUTO 可重新接管";
        if (status.equals("ERROR")) return "已停止：接口/模型错误 · 见日志";
        return "正在加载模型，尚未接管";
    }

    public void receivePostRender(SpriteBatch sb) {
        if (BaseMod.modSettingsUp || (!inGame() && !CommandExecutor.isStartCommandAvailable())) return;
        float s = Settings.scale, x = centerX(), y = centerY();
        Color old = new Color(sb.getColor());
        sb.setColor(0.025f, .04f, .06f, .94f);
        if (inGame()) {
            sb.draw(ImageMaster.WHITE_SQUARE_IMG, x - 165*s, y - 77*s, 330*s, 110*s);
            sb.setColor(connected && button.hovered ? new Color(.18f,.42f,.42f,1f) : new Color(.10f,.23f,.28f,1f));
            sb.draw(ImageMaster.WHITE_SQUARE_IMG, x - 145*s, y - 23*s, 290*s, 46*s);
            FontHelper.renderFontCentered(sb, FontHelper.buttonLabelFont,
                    enabled ? "AUTO：开  ·  点击暂停" : "AUTO：关  ·  点击接管", x, y + 1*s,
                    connected ? Color.WHITE : Color.GRAY);
            FontHelper.renderFontCentered(sb, FontHelper.tipBodyFont, statusText(), x, y - 43*s,
                    status.equals("ERROR") ? Color.SALMON : Settings.CREAM_COLOR);
            FontHelper.renderFontCentered(sb, FontHelper.tipBodyFont, model, x, y - 68*s, Settings.GOLD_COLOR);
            sb.setColor(old);
            return;
        }
        sb.draw(ImageMaster.WHITE_SQUARE_IMG, x - 165*s, y - 130*s, 330*s, 275*s);
        FontHelper.renderFontCentered(sb, FontHelper.buttonLabelFont, "模型自动演示", x, y + 116*s, Settings.GOLD_COLOR);
        FontHelper.renderFontCentered(sb, FontHelper.tipBodyFont, model, x, y + 78*s, Settings.CREAM_COLOR);
        FontHelper.renderFontCentered(sb, FontHelper.tipBodyFont, statusText(), x, y + 43*s,
                status.equals("ERROR") ? Color.SALMON : Settings.CREAM_COLOR);
        sb.setColor(connected && button.hovered ? new Color(.18f,.42f,.42f,1f) : new Color(.10f,.23f,.28f,1f));
        sb.draw(ImageMaster.WHITE_SQUARE_IMG, x - 145*s, y - 23*s, 290*s, 46*s);
        String label = inGame() ? (enabled ? "AUTO：开  ·  点击暂停" : "AUTO：关  ·  点击接管") : "随机种子演示";
        FontHelper.renderFontCentered(sb, FontHelper.buttonLabelFont, label, x, y + 1*s,
                connected ? Color.WHITE : Color.GRAY);
        FontHelper.renderFontCentered(sb, FontHelper.tipBodyFont,
                inGame() ? "从当前画面接管；再次点击交还操作" : "只开一局；开始前备份现有战士存档", x, y - 53*s, Settings.CREAM_COLOR);
        FontHelper.renderFontCentered(sb, FontHelper.tipBodyFont,
                "战士 A0 / 前三幕 / 不支持的状态明确停止", x, y - 83*s, Settings.CREAM_COLOR);
        if (!detail.isEmpty()) FontHelper.renderFontCentered(sb, FontHelper.tipBodyFont,
                detail.length() > 36 ? detail.substring(0, 36) : detail, x, y - 113*s, Settings.GOLD_COLOR);
        sb.setColor(old);
    }
}
