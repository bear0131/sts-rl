// Offline CommunicationMod patcher: only current monster serialization changes.
// Compile/run with ModTheSpire's Javassist; dependencies are local, never bundled.
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
        CtMethod method = converter.getDeclaredMethod("convertMonsterToJson");
        Map<String, byte[]> unchanged = new HashMap<>();
        for (CtMethod candidate : converter.getDeclaredMethods()) {
            if (!candidate.equals(method)) unchanged.put(candidate.getLongName(),
                candidate.getMethodInfo().getCodeAttribute().getCode().clone());
        }
        final int[] replacements = {0};
        method.instrument(new ExprEditor() {
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
        method.insertAfter("{ $_.put(\"current_intent_export\", \"simulator-parity-v1\"); }");
        byte[] patched = converter.toBytecode();
        for (CtMethod candidate : converter.getDeclaredMethods()) {
            if (!candidate.equals(method) && !Arrays.equals(
                    unchanged.get(candidate.getLongName()),
                    candidate.getMethodInfo2().getCodeAttribute().getCode()))
                throw new IllegalStateException("Unexpected method change: " + candidate.getLongName());
        }
        String target = "communicationmod/GameStateConverter.class";
        int changed = 0;
        try (ZipFile source = new ZipFile(input.toFile());
             ZipOutputStream dest = new ZipOutputStream(Files.newOutputStream(output))) {
            Enumeration<? extends ZipEntry> entries = source.entries();
            byte[] buffer = new byte[65536];
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                ZipEntry copy = new ZipEntry(entry.getName());
                copy.setTime(entry.getTime());
                dest.putNextEntry(copy);
                if (entry.getName().equals(target)) { dest.write(patched); changed++; }
                else try (InputStream stream = source.getInputStream(entry)) {
                    int n; while ((n = stream.read(buffer)) != -1) dest.write(buffer, 0, n);
                }
                dest.closeEntry();
            }
        }
        if (changed != 1) throw new IllegalStateException("Target count=" + changed);
        System.out.println("Patched one serializer; no gameplay or RNG methods modified");
    }
}
