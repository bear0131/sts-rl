// Offline CommunicationMod patcher for the complete simulator-visible state.
// Compile/run with local game/mod jars; only the exporter helper is bundled.
import javassist.*;
import javassist.expr.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.*;
import java.io.*;

public final class ExposeCurrentIntent {
    public static void main(String[] args) throws Exception {
        if (args.length != 5) throw new IllegalArgumentException(
            "input.jar output.jar desktop-1.0.jar BaseMod.jar ModTheSpire.jar");
        Path input = Paths.get(args[0]), output = Paths.get(args[1]);
        if (Files.exists(output)) throw new IOException("Output already exists");
        ClassPool pool = new ClassPool(true);
        for (int i : new int[]{0, 2, 3, 4}) pool.appendClassPath(args[i]);
        CtClass converter = pool.get("communicationmod.GameStateConverter");
        CtMethod gameMethod = converter.getDeclaredMethod("getGameState");
        CtMethod monsterMethod = converter.getDeclaredMethod("convertMonsterToJson");
        CtMethod combatMethod = converter.getDeclaredMethod("getCombatState");
        CtMethod gridMethod = converter.getDeclaredMethod("getGridState");
        CtMethod handMethod = converter.getDeclaredMethod("getHandSelectState");
        CtMethod cardRewardMethod = converter.getDeclaredMethod("getCardRewardState");
        CtMethod cardMethod = converter.getDeclaredMethod("convertCardToJson");
        Set<CtMethod> changedMethods = new HashSet<CtMethod>(Arrays.asList(
            gameMethod, monsterMethod, combatMethod, gridMethod, handMethod,
            cardRewardMethod, cardMethod));
        Map<String, byte[]> unchanged = new HashMap<>();
        for (CtMethod candidate : converter.getDeclaredMethods()) {
            if (!changedMethods.contains(candidate)) unchanged.put(candidate.getLongName(),
                candidate.getMethodInfo().getCodeAttribute().getCode().clone());
        }
        final int[] replacements = {0};
        monsterMethod.instrument(new ExprEditor() {
            public void edit(MethodCall call) throws CannotCompileException {
                if (call.getClassName().equals("com.megacrit.cardcrawl.characters.AbstractPlayer")
                        && call.getMethodName().equals("hasRelic")) {
                    replacements[0]++;
                    call.replace("{ $_ = \"Runic Dome\".equals($1) ? false : $proceed($$); }");
                }
            }
        });
        if (replacements[0] != 1) throw new IllegalStateException(
            "Unexpected serializer version: hasRelic calls=" + replacements[0]);
        gameMethod.insertAfter(
            "{ communicationmod.SimulatorParityExporter.augmentGameState($_); }");
        monsterMethod.insertAfter(
            "{ $_.put(\"current_intent_export\", communicationmod.SimulatorParityExporter.VERSION); " +
            "com.megacrit.cardcrawl.cards.AbstractCard stasis = communicationmod.SimulatorParityExporter.stasisCard($1); " +
            "if (stasis == null) $_.put(\"simulator_stasis_card\", null); " +
            "else $_.put(\"simulator_stasis_card\", convertCardToJson(stasis)); }");
        combatMethod.insertAfter(
            "{ communicationmod.SimulatorParityExporter.augmentCombatState($_); }");
        gridMethod.insertAfter(
            "{ communicationmod.SimulatorParityExporter.augmentGridState($_); }");
        handMethod.insertAfter(
            "{ communicationmod.SimulatorParityExporter.augmentHandSelectState($_); }");
        cardRewardMethod.insertAfter(
            "{ communicationmod.SimulatorParityExporter.augmentCardRewardState($_); }");
        cardMethod.insertAfter(
            "{ communicationmod.SimulatorParityExporter.augmentCardState($_, $1); }");
        byte[] patched = converter.toBytecode();
        for (CtMethod candidate : converter.getDeclaredMethods()) {
            if (!changedMethods.contains(candidate) && !Arrays.equals(
                    unchanged.get(candidate.getLongName()),
                    candidate.getMethodInfo2().getCodeAttribute().getCode()))
                throw new IllegalStateException("Unexpected method change: " + candidate.getLongName());
        }
        String target = "communicationmod/GameStateConverter.class";
        String[] helperTargets = new String[] {
            "communicationmod/SimulatorParityExporter.class",
            "communicationmod/SimulatorParityExporter$DamagePatch.class",
            "communicationmod/SimulatorParityExporter$PreBattlePatch.class",
        };
        Map<String, byte[]> helpers = new LinkedHashMap<String, byte[]>();
        for (String helperTarget : helperTargets) {
            try (InputStream stream = ExposeCurrentIntent.class.getClassLoader()
                    .getResourceAsStream(helperTarget)) {
                if (stream == null) throw new IOException("Missing compiled helper " + helperTarget);
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                byte[] helperBuffer = new byte[65536];
                int n;
                while ((n = stream.read(helperBuffer)) != -1) bytes.write(helperBuffer, 0, n);
                helpers.put(helperTarget, bytes.toByteArray());
            }
        }
        int changed = 0;
        try (ZipFile source = new ZipFile(input.toFile());
             ZipOutputStream dest = new ZipOutputStream(Files.newOutputStream(output))) {
            Enumeration<? extends ZipEntry> entries = source.entries();
            byte[] buffer = new byte[65536];
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                if (helpers.containsKey(entry.getName())) {
                    throw new IllegalStateException("Input jar already contains " + entry.getName());
                }
                ZipEntry copy = new ZipEntry(entry.getName());
                copy.setTime(entry.getTime());
                dest.putNextEntry(copy);
                if (entry.getName().equals(target)) { dest.write(patched); changed++; }
                else try (InputStream stream = source.getInputStream(entry)) {
                    int n; while ((n = stream.read(buffer)) != -1) dest.write(buffer, 0, n);
                }
                dest.closeEntry();
            }
            for (Map.Entry<String, byte[]> helper : helpers.entrySet()) {
                ZipEntry helperEntry = new ZipEntry(helper.getKey());
                dest.putNextEntry(helperEntry);
                dest.write(helper.getValue());
                dest.closeEntry();
            }
        }
        if (changed != 1) throw new IllegalStateException("Target count=" + changed);
        System.out.println(
            "Patched seven serializers and added read-only parity exporters; no gameplay or RNG modified");
    }
}
