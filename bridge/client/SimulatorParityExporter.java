package communicationmod;

import basemod.ReflectionHacks;
import com.megacrit.cardcrawl.actions.AbstractGameAction;
import com.megacrit.cardcrawl.cards.AbstractCard;
import com.megacrit.cardcrawl.cards.DamageInfo;
import com.megacrit.cardcrawl.characters.AbstractPlayer;
import com.megacrit.cardcrawl.dungeons.AbstractDungeon;
import com.megacrit.cardcrawl.events.AbstractEvent;
import com.megacrit.cardcrawl.monsters.AbstractMonster;
import com.megacrit.cardcrawl.powers.AbstractPower;
import com.megacrit.cardcrawl.relics.AbstractRelic;
import com.megacrit.cardcrawl.screens.select.GridCardSelectScreen;
import com.megacrit.cardcrawl.core.Settings;
import com.evacipated.cardcrawl.modthespire.lib.SpirePatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePostfixPatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePrefixPatch;
import com.megacrit.cardcrawl.shop.ShopScreen;
import java.util.HashMap;
import java.util.ArrayList;

/** Extra state that the simulator observes but stock CommunicationMod omits. */
public final class SimulatorParityExporter {
    public static final String VERSION = "simulator-parity-v30";
    private static int lastAttackUnblockedDamage = 0;
    private static ShopScreen shopOwner;
    private static final AbstractCard[] shopCards = new AbstractCard[7];
    private static final com.megacrit.cardcrawl.shop.StoreRelic[] shopRelics = new com.megacrit.cardcrawl.shop.StoreRelic[3];
    private static final com.megacrit.cardcrawl.shop.StorePotion[] shopPotions = new com.megacrit.cardcrawl.shop.StorePotion[3];

    private SimulatorParityExporter() {}

    @SpirePatch(clz = ShopScreen.class, method = "init")
    public static class ShopInitPatch {
        @SpirePostfixPatch
        public static void Postfix(ShopScreen __instance) {
            if (__instance.coloredCards.size() != 5 || __instance.colorlessCards.size() != 2)
                throw new IllegalStateException("Unsupported initial shop card shelf");
            shopOwner = __instance;
            for (int i = 0; i < 5; i++) shopCards[i] = __instance.coloredCards.get(i);
            for (int i = 0; i < 2; i++) shopCards[5 + i] = __instance.colorlessCards.get(i);
            ArrayList<com.megacrit.cardcrawl.shop.StoreRelic> relics = ReflectionHacks.getPrivate(__instance, ShopScreen.class, "relics");
            ArrayList<com.megacrit.cardcrawl.shop.StorePotion> potions = ReflectionHacks.getPrivate(__instance, ShopScreen.class, "potions");
            if (relics.size() != 3 || potions.size() != 3)
                throw new IllegalStateException("Unsupported initial shop item shelf");
            for (int i = 0; i < 3; i++) {
                if (privateInt(relics.get(i), com.megacrit.cardcrawl.shop.StoreRelic.class, "slot") != i
                        || privateInt(potions.get(i), com.megacrit.cardcrawl.shop.StorePotion.class, "slot") != i)
                    throw new IllegalStateException("Unexpected initial shop item slot order");
                shopRelics[i] = relics.get(i);
                shopPotions[i] = potions.get(i);
            }
        }
    }

    private static int shopCardSlot(AbstractCard card) {
        for (int i = 0; i < shopCards.length; i++) if (shopCards[i] == card) return i;
        throw new IllegalStateException("Shop card has no observed physical slot");
    }

    @SpirePatch(clz = ShopScreen.class, method = "purchaseCard")
    public static class ShopPurchasePatch {
        private static AbstractCard original;
        private static ArrayList<AbstractCard> row;
        private static int slot, index, size;

        @SpirePrefixPatch
        public static void Prefix(ShopScreen __instance, AbstractCard hoveredCard) {
            if (shopOwner != __instance) throw new IllegalStateException("Unobserved shop initialization");
            original = hoveredCard;
            slot = shopCardSlot(original);
            row = slot < 5 ? __instance.coloredCards : __instance.colorlessCards;
            index = row.indexOf(original);
            size = row.size();
            if (index < 0) throw new IllegalStateException("Shop purchase source is absent");
        }

        @SpirePostfixPatch
        public static void Postfix(ShopScreen __instance) {
            // The game nulls its parameter locally; keep the actual prefix
            // object. Only record its removal/replacement, never perform it.
            if (row.contains(original)) return; // Unaffordable purchase did nothing.
            if (row.size() == size - 1) shopCards[slot] = null;
            else if (row.size() == size && AbstractDungeon.player.hasRelic("The Courier"))
                shopCards[slot] = row.get(index);
            else throw new IllegalStateException("Unmapped shop card purchase result");
        }
    }

    @SuppressWarnings("unchecked")
    public static void augmentShopState(HashMap<String, Object> state) {
        if (shopOwner != AbstractDungeon.shopScreen)
            throw new IllegalStateException("Shop shelf was not observed at initialization");
        ArrayList<HashMap<String, Object>> cards = (ArrayList<HashMap<String, Object>>)state.get("cards");
        ArrayList<AbstractCard> actualCards = ChoiceScreenUtils.getShopScreenCards();
        if (cards.size() != actualCards.size()) throw new IllegalStateException("Shop card export count mismatch");
        for (int i = 0; i < cards.size(); i++) cards.get(i).put("shop_slot", shopCardSlot(actualCards.get(i)));
        ArrayList<Integer> sold = new ArrayList<Integer>();
        for (int i = 0; i < shopCards.length; i++) if (shopCards[i] == null) sold.add(i);
        if (cards.size() + sold.size() != 7) throw new IllegalStateException("Incomplete shop card shelf");
        state.put("sold_card_slots", sold);
        ArrayList<HashMap<String, Object>> relics = (ArrayList<HashMap<String, Object>>)state.get("relics");
        ArrayList<com.megacrit.cardcrawl.shop.StoreRelic> actualRelics = ChoiceScreenUtils.getShopScreenRelics();
        if (relics.size() != actualRelics.size()) throw new IllegalStateException("Shop relic export count mismatch");
        for (int i = 0; i < relics.size(); i++) {
            com.megacrit.cardcrawl.shop.StoreRelic item = actualRelics.get(i);
            int slot = privateInt(item, com.megacrit.cardcrawl.shop.StoreRelic.class, "slot");
            if (slot < 0 || slot >= 3 || shopRelics[slot] != item)
                throw new IllegalStateException("Unobserved shop relic slot");
            relics.get(i).put("shop_slot", slot);
            relics.get(i).put("is_purchased", item.isPurchased);
        }
        ArrayList<HashMap<String, Object>> potions = (ArrayList<HashMap<String, Object>>)state.get("potions");
        ArrayList<com.megacrit.cardcrawl.shop.StorePotion> actualPotions = ChoiceScreenUtils.getShopScreenPotions();
        if (potions.size() != actualPotions.size()) throw new IllegalStateException("Shop potion export count mismatch");
        for (int i = 0; i < potions.size(); i++) {
            com.megacrit.cardcrawl.shop.StorePotion item = actualPotions.get(i);
            int slot = privateInt(item, com.megacrit.cardcrawl.shop.StorePotion.class, "slot");
            if (slot < 0 || slot >= 3 || shopPotions[slot] != item)
                throw new IllegalStateException("Unobserved shop potion slot");
            potions.get(i).put("shop_slot", slot);
            potions.get(i).put("is_purchased", item.isPurchased);
        }
        ArrayList<Integer> soldRelics = new ArrayList<Integer>(), soldPotions = new ArrayList<Integer>();
        for (int i = 0; i < 3; i++) {
            if (shopRelics[i].isPurchased) soldRelics.add(i);
            else if (!actualRelics.contains(shopRelics[i])) throw new IllegalStateException("Unsold relic disappeared");
            if (shopPotions[i].isPurchased) soldPotions.add(i);
            else if (!actualPotions.contains(shopPotions[i])) throw new IllegalStateException("Unsold potion disappeared");
        }
        state.put("sold_relic_slots", soldRelics);
        state.put("sold_potion_slots", soldPotions);
    }

    public static void augmentMonsterState(HashMap<String, Object> state, AbstractMonster monster) {
        // Stock CommunicationMod omits short history, whereas the strict
        // transport needs explicit absence. Read the actual history length;
        // never turn an unavailable/unknown move into the no-history sentinel.
        int historySize = monster.moveHistory.size();
        state.put("last_move_id", Integer.valueOf(historySize >= 2
            ? monster.moveHistory.get(historySize - 2).intValue() : -1));
        state.put("second_last_move_id", Integer.valueOf(historySize >= 3
            ? monster.moveHistory.get(historySize - 3).intValue() : -1));
        state.put("is_dying", Boolean.valueOf(monster.isDying));
        state.put("is_dead", Boolean.valueOf(monster.isDead));
        state.put("is_escaping", Boolean.valueOf(monster.isEscaping));
        state.put("escape_next", Boolean.valueOf(monster.escapeNext));
        Class<?> thiefClass = null;
        if (monster instanceof com.megacrit.cardcrawl.monsters.exordium.Looter)
            thiefClass = com.megacrit.cardcrawl.monsters.exordium.Looter.class;
        else if (monster instanceof com.megacrit.cardcrawl.monsters.city.Mugger)
            thiefClass = com.megacrit.cardcrawl.monsters.city.Mugger.class;
        HashMap<String, Object> thief = null;
        if (thiefClass != null) {
            thief = new HashMap<String, Object>();
            thief.put("stolen_gold", Integer.valueOf(privateInt(monster, thiefClass, "stolenGold")));
            thief.put("slash_count", Integer.valueOf(privateInt(monster, thiefClass, "slashCount")));
            thief.put("gold_amount", Integer.valueOf(privateInt(monster, thiefClass, "goldAmt")));
        }
        // Corpses retain their ledger; only a genuinely non-thief has null.
        state.put("simulator_thief_state", thief);
        HashMap<String, Object> lagavulin = null;
        if (monster instanceof com.megacrit.cardcrawl.monsters.exordium.Lagavulin) {
            Class<?> type = com.megacrit.cardcrawl.monsters.exordium.Lagavulin.class;
            lagavulin = new HashMap<String, Object>();
            // Gameplay sleep/wake state, not the constructor's unchanged asleep flag.
            lagavulin.put("sleeping", Boolean.valueOf(!privateBool(monster, type, "isOutTriggered")));
            lagavulin.put("idle_count", Integer.valueOf(privateInt(monster, type, "idleCount")));
        }
        state.put("simulator_lagavulin_state", lagavulin);
        state.put("simulator_wizard_charge", monster instanceof com.megacrit.cardcrawl.monsters.exordium.GremlinWizard
            ? Integer.valueOf(privateInt(monster, com.megacrit.cardcrawl.monsters.exordium.GremlinWizard.class, "currentCharge")) : null);
        state.put("simulator_writhing_mass_used_implant", monster instanceof com.megacrit.cardcrawl.monsters.beyond.WrithingMass
            ? Boolean.valueOf(privateBool(monster, com.megacrit.cardcrawl.monsters.beyond.WrithingMass.class, "usedMegaDebuff")) : null);
        state.put("simulator_move_state", monsterMoveState(monster));
        // UnawakenedPower is removed at the first lethal hit, while the room's
        // death protection remains until the queued CanLoseAction executes.
        state.put("unawakened", Boolean.valueOf(
            monster instanceof com.megacrit.cardcrawl.monsters.beyond.AwakenedOne
            && AbstractDungeon.getCurrRoom().cannotLose));
    }

    private static HashMap<String, Object> monsterMoveState(AbstractMonster monster) {
        HashMap<String, Object> result = new HashMap<String, Object>();
        if (monster instanceof com.megacrit.cardcrawl.monsters.beyond.TimeEater) {
            result.put("used_haste", Boolean.valueOf(privateBool(monster,
                com.megacrit.cardcrawl.monsters.beyond.TimeEater.class, "usedHaste")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.exordium.SlaverRed) {
            result.put("used_entangle", Boolean.valueOf(privateBool(monster,
                com.megacrit.cardcrawl.monsters.exordium.SlaverRed.class, "usedEntangle")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.city.BronzeOrb) {
            result.put("used_stasis", Boolean.valueOf(privateBool(monster,
                com.megacrit.cardcrawl.monsters.city.BronzeOrb.class, "usedStasis")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.beyond.Spiker) {
            result.put("thorns_count", Integer.valueOf(privateInt(monster,
                com.megacrit.cardcrawl.monsters.beyond.Spiker.class, "thornsCount")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.city.Champ) {
            Class<?> type = com.megacrit.cardcrawl.monsters.city.Champ.class;
            result.put("num_turns", Integer.valueOf(privateInt(monster, type, "numTurns")));
            result.put("forge_times", Integer.valueOf(privateInt(monster, type, "forgeTimes")));
            result.put("threshold_reached", Boolean.valueOf(privateBool(monster, type, "thresholdReached")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.city.BronzeAutomaton) {
            result.put("num_turns", Integer.valueOf(privateInt(monster,
                com.megacrit.cardcrawl.monsters.city.BronzeAutomaton.class, "numTurns")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.city.BookOfStabbing) {
            result.put("stab_count", Integer.valueOf(privateInt(monster,
                com.megacrit.cardcrawl.monsters.city.BookOfStabbing.class, "stabCount")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.exordium.LouseNormal) {
            result.put("bite_damage", Integer.valueOf(privateInt(monster,
                com.megacrit.cardcrawl.monsters.exordium.LouseNormal.class, "biteDamage")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.exordium.LouseDefensive) {
            result.put("bite_damage", Integer.valueOf(privateInt(monster,
                com.megacrit.cardcrawl.monsters.exordium.LouseDefensive.class, "biteDamage")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.beyond.Darkling) {
            result.put("nip_damage", Integer.valueOf(privateInt(monster,
                com.megacrit.cardcrawl.monsters.beyond.Darkling.class, "nipDmg")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.exordium.TheGuardian) {
            result.put("damage_threshold", Integer.valueOf(privateInt(monster,
                com.megacrit.cardcrawl.monsters.exordium.TheGuardian.class, "dmgThreshold")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.beyond.GiantHead) {
            result.put("count", Integer.valueOf(privateInt(monster,
                com.megacrit.cardcrawl.monsters.beyond.GiantHead.class, "count")));
        } else if (monster instanceof com.megacrit.cardcrawl.monsters.exordium.Hexaghost) {
            Class<?> type = com.megacrit.cardcrawl.monsters.exordium.Hexaghost.class;
            result.put("activated", Boolean.valueOf(privateBool(monster, type, "activated")));
            result.put("burn_upgraded", Boolean.valueOf(privateBool(monster, type, "burnUpgraded")));
            result.put("orb_active_count", Integer.valueOf(privateInt(monster, type, "orbActiveCount")));
            if (monster.damage.size() != 4)
                throw new IllegalStateException("Unexpected Hexaghost damage table");
            result.put("divider_damage", Integer.valueOf(monster.damage.get(2).base));
        } else {
            return null;
        }
        return result;
    }

    public static AbstractCard stasisCard(com.megacrit.cardcrawl.monsters.AbstractMonster monster) {
        if (monster.isDeadOrEscaped()) return null;
        for (AbstractPower power : monster.powers) {
            if ("Stasis".equals(power.ID)) {
                if (!(power instanceof com.megacrit.cardcrawl.powers.StasisPower))
                    throw new IllegalStateException("Unknown Stasis power implementation");
                AbstractCard card = ReflectionHacks.getPrivate(power,
                    com.megacrit.cardcrawl.powers.StasisPower.class, "card");
                if (card == null) throw new IllegalStateException("Active Stasis has no card");
                return card;
            }
        }
        return null;
    }

    public static void augmentCardState(HashMap<String, Object> state, AbstractCard card) {
        // Read actual instance fields only: no RNG, copies, applyPowers, or use.
        state.put("misc", Integer.valueOf(card.misc));
        HashMap<String, Object> extra = new HashMap<String, Object>();
        extra.put("version", Integer.valueOf(1));
        extra.put("base_cost", Integer.valueOf(card.cost));
        extra.put("free_to_play_once", Boolean.valueOf(card.freeToPlayOnce));
        extra.put("retain", Boolean.valueOf(card.retain));
        extra.put("self_retain", Boolean.valueOf(card.selfRetain));
        extra.put("innate", Boolean.valueOf(card.isInnate));
        extra.put("base_damage", Integer.valueOf(card.baseDamage));
        extra.put("base_block", Integer.valueOf(card.baseBlock));
        extra.put("bottled_flame", Boolean.valueOf(card.inBottleFlame));
        extra.put("bottled_lightning", Boolean.valueOf(card.inBottleLightning));
        extra.put("bottled_tornado", Boolean.valueOf(card.inBottleTornado));
        state.put("simulator_card_state", extra);
    }

    private static int relicCounter(String id) {
        AbstractRelic relic = AbstractDungeon.player.getRelic(id);
        return relic == null ? 0 : relic.counter;
    }

    private static AbstractPower power(String id) {
        for (AbstractPower power : AbstractDungeon.player.powers) {
            if (id.equals(power.ID)) return power;
        }
        return null;
    }

    private static int privateInt(Object value, Class<?> owner, String field) {
        return ((Integer)ReflectionHacks.getPrivate(value, owner, field)).intValue();
    }

    private static boolean privateBool(Object value, Class<?> owner, String field) {
        return ((Boolean)ReflectionHacks.getPrivate(value, owner, field)).booleanValue();
    }

    private static Object eventProgress(AbstractEvent event) {
        if (event == null) return null;
        String name = event.getClass().getName(), field;
        switch (name) {
            case "com.megacrit.cardcrawl.events.shrines.AccursedBlacksmith":
            case "com.megacrit.cardcrawl.events.exordium.GoldenIdolEvent":
            case "com.megacrit.cardcrawl.events.beyond.MoaiHead": field = "screenNum"; break;
            case "com.megacrit.cardcrawl.events.shrines.Bonfire":
            case "com.megacrit.cardcrawl.events.city.Colosseum":
            case "com.megacrit.cardcrawl.events.exordium.DeadAdventurer":
            case "com.megacrit.cardcrawl.events.exordium.GoldenWing":
            case "com.megacrit.cardcrawl.events.exordium.Sssserpent":
            case "com.megacrit.cardcrawl.events.exordium.ShiningLight":
            case "com.megacrit.cardcrawl.events.city.TheJoust":
            case "com.megacrit.cardcrawl.events.city.TheMausoleum":
            case "com.megacrit.cardcrawl.events.shrines.GremlinWheelGame": field = "screen"; break;
            default: return null;
        }
        Object phase = ReflectionHacks.getPrivate(event, event.getClass(), field);
        HashMap<String, Object> progress = new HashMap<String, Object>();
        progress.put("source", name);
        progress.put("phase", phase instanceof Enum ? ((Enum<?>)phase).name() : phase.toString());
        if (event instanceof com.megacrit.cardcrawl.events.exordium.DeadAdventurer)
            progress.put("rewards_taken", privateInt(event, event.getClass(), "numRewards"));
        if (event instanceof com.megacrit.cardcrawl.events.shrines.GremlinWheelGame) {
            for (String key : new String[]{"startSpin", "buttonPressed", "purgeResult"})
                progress.put(key, privateBool(event, event.getClass(), key));
        }
        // Only current execution progress; never export the wheel's hidden
        // result or Joust's ownerWins roll to the policy.
        return progress;
    }

    private static boolean pendingCardObtain() {
        for (java.util.ArrayList<com.megacrit.cardcrawl.vfx.AbstractGameEffect> queue :
                java.util.Arrays.asList(AbstractDungeon.effectList, AbstractDungeon.effectsQueue,
                    AbstractDungeon.topLevelEffects, AbstractDungeon.topLevelEffectsQueue)) {
            for (com.megacrit.cardcrawl.vfx.AbstractGameEffect effect : queue)
                if (!effect.isDone && effect instanceof com.megacrit.cardcrawl.vfx.cardManip.ShowCardAndObtainEffect)
                    return true;
        }
        return false;
    }

    public static void augmentGameState(HashMap<String, Object> state) {
        state.put("has_ruby_key", Boolean.valueOf(Settings.hasRubyKey));
        state.put("has_emerald_key", Boolean.valueOf(Settings.hasEmeraldKey));
        state.put("has_sapphire_key", Boolean.valueOf(Settings.hasSapphireKey));
        state.put("simulator_parity_export", VERSION);
        // Stock export omits this key outside an action. Emit observed absence
        // explicitly; the patched wire serializer must preserve the null.
        AbstractGameAction currentAction = AbstractDungeon.actionManager.currentAction;
        // Anonymous actions have an empty simple name. Preserve their actual
        // binary class identity, including the enclosing class and $ ordinal;
        // only a genuinely absent action is represented by null.
        state.put("current_action", currentAction == null ? null : currentAction.getClass().getName());
        state.put("current_action_is_anonymous", Boolean.valueOf(
            currentAction != null && currentAction.getClass().isAnonymousClass()));
        if (CommandExecutor.isChooseCommandAvailable()) {
            if (!(state.get("choice_list") instanceof java.util.ArrayList))
                throw new IllegalStateException("Available CHOOSE requires the actual choice list");
        } else {
            // This is a known unavailable command, not a missing list fallback.
            state.put("choice_list", new java.util.ArrayList<String>());
        }
        state.put("simulator_current_node", mapNodeIdentity(AbstractDungeon.getCurrMapNode()));
        java.util.ArrayList<HashMap<String, Object>> leaves = new java.util.ArrayList<HashMap<String, Object>>();
        for (java.util.ArrayList<com.megacrit.cardcrawl.map.MapRoomNode> row : AbstractDungeon.map) {
            for (com.megacrit.cardcrawl.map.MapRoomNode node : row) {
                // Stock convertMapToJson filters out every node without edges,
                // including the actual Heart and victory rooms in TheEnding.
                if (node.room != null && !node.hasEdges()) {
                    HashMap<String, Object> leaf = mapNodeIdentity(node);
                    leaf.put("children", new java.util.ArrayList<Object>());
                    leaves.add(leaf);
                }
            }
        }
        state.put("simulator_map_leaf_nodes", leaves);
        AbstractEvent event = AbstractDungeon.getCurrRoom().event;
        state.put("simulator_event_progress", eventProgress(event));
        state.put("simulator_obtain_pending", pendingCardObtain());
        if (event instanceof com.megacrit.cardcrawl.neow.NeowEvent) {
            com.megacrit.cardcrawl.neow.NeowEvent neow = (com.megacrit.cardcrawl.neow.NeowEvent)event;
            int screen = privateInt(neow, com.megacrit.cardcrawl.neow.NeowEvent.class, "screenNum");
            state.put("simulator_neow_screen", Integer.valueOf(screen));
            if (screen == 3) {
                java.util.ArrayList<com.megacrit.cardcrawl.neow.NeowReward> rewards =
                    ReflectionHacks.getPrivate(neow, com.megacrit.cardcrawl.neow.NeowEvent.class, "rewards");
                if (rewards == null || (rewards.size() != 2 && rewards.size() != 4)) {
                    throw new IllegalStateException("Neow must have two or four typed rewards");
                }
                java.util.ArrayList<HashMap<String, Object>> offers = new java.util.ArrayList<HashMap<String, Object>>();
                for (int index = 0; index < rewards.size(); index++) {
                    com.megacrit.cardcrawl.neow.NeowReward reward = rewards.get(index);
                    if (reward.type == null || reward.drawback == null) {
                        throw new IllegalStateException("Neow reward is missing its enum identity");
                    }
                    HashMap<String, Object> offer = new HashMap<String, Object>();
                    offer.put("choice_index", Integer.valueOf(index));
                    offer.put("bonus_type", reward.type.name());
                    offer.put("drawback_type", reward.drawback.name());
                    offer.put("hp_unit", Integer.valueOf(privateInt(reward,
                        com.megacrit.cardcrawl.neow.NeowReward.class, "hp_bonus")));
                    offers.add(offer);
                }
                state.put("simulator_neow_options", offers);
            }
        }
    }

    private static HashMap<String, Object> mapNodeIdentity(com.megacrit.cardcrawl.map.MapRoomNode node) {
        if (node == null || node.room == null)
            throw new IllegalStateException("Map identity requires the actual node and room");
        HashMap<String, Object> result = new HashMap<String, Object>();
        result.put("x", Integer.valueOf(node.x));
        result.put("y", Integer.valueOf(node.y));
        result.put("room_class", node.room.getClass().getSimpleName());
        result.put("symbol", node.getRoomSymbol(Boolean.TRUE));
        return result;
    }

    private static Object summonSlots() {
        ArrayList<AbstractMonster> monsters = AbstractDungeon.getMonsters().monsters;
        HashMap<String, Object> result = null;
        for (AbstractMonster leader : monsters) {
            AbstractMonster[] slots;
            if (leader instanceof com.megacrit.cardcrawl.monsters.city.GremlinLeader) {
                AbstractMonster[] gremlins = ((com.megacrit.cardcrawl.monsters.city.GremlinLeader)leader).gremlins;
                slots = new AbstractMonster[]{gremlins[2], gremlins[0], gremlins[1], leader};
            } else if (leader instanceof com.megacrit.cardcrawl.monsters.city.TheCollector) {
                HashMap<Integer, AbstractMonster> heads = ReflectionHacks.getPrivate(
                    leader, com.megacrit.cardcrawl.monsters.city.TheCollector.class, "enemySlots");
                slots = new AbstractMonster[]{heads.get(1), heads.get(0), leader};
            } else if (leader instanceof com.megacrit.cardcrawl.monsters.beyond.Reptomancer) {
                AbstractMonster[] daggers = ReflectionHacks.getPrivate(
                    leader, com.megacrit.cardcrawl.monsters.beyond.Reptomancer.class, "daggers");
                slots = new AbstractMonster[]{daggers[3], daggers[1], leader, daggers[2], daggers[0]};
            } else continue;
            if (result != null) throw new IllegalStateException("Multiple summon leaders in one encounter");
            result = new HashMap<String, Object>();
            result.put("leader", leader.id);
            ArrayList<Integer> indices = new ArrayList<Integer>();
            boolean pending = false;
            for (AbstractMonster monster : slots) {
                if (monster == null) indices.add(null); // Actual unoccupied slot.
                else {
                    int index = monsters.indexOf(monster);
                    // Constructors reserve the object before the queued spawn
                    // inserts it. Explicit pending prevents a settled restore.
                    indices.add(Integer.valueOf(index));
                    if (index < 0) pending = true;
                }
            }
            result.put("slots", indices);
            result.put("pending", pending);
        }
        return result;
    }

    public static void augmentCombatState(HashMap<String, Object> state) {
        state.put("simulator_summon_slots", summonSlots());
        AbstractPlayer player = AbstractDungeon.player;
        HashMap<String, Object> counters = new HashMap<String, Object>();
        counters.put("happy_flower", Integer.valueOf(relicCounter("Happy Flower")));
        counters.put("incense_burner", Integer.valueOf(relicCounter("Incense Burner")));
        counters.put("ink_bottle", Integer.valueOf(relicCounter("InkBottle")));
        counters.put("inserter", Integer.valueOf(relicCounter("Inserter")));
        counters.put("nunchaku", Integer.valueOf(relicCounter("Nunchaku")));
        counters.put("pen_nib", Integer.valueOf(relicCounter("Pen Nib")));
        counters.put("sundial", Integer.valueOf(relicCounter("Sundial")));
        int neow = relicCounter("NeowsBlessing");
        if (neow == -2) neow = 0;
        if (neow < 0 || neow > 3) throw new IllegalStateException("Neow remainder outside 0..3");
        counters.put("neow_lament", Integer.valueOf(neow));
        AbstractRelic puzzle = player.getRelic("Centennial Puzzle");
        counters.put("centennial_puzzle_active", Boolean.valueOf(puzzle != null &&
            !((Boolean)ReflectionHacks.getPrivateStatic(
                com.megacrit.cardcrawl.relics.CentennialPuzzle.class,
                "usedThisCombat")).booleanValue()));
        AbstractRelic tail = player.getRelic("Lizard Tail");
        if (tail != null && tail.counter != -1 && tail.counter != -2) {
            throw new IllegalStateException("Lizard Tail counter outside -1/-2 contract: "
                + tail.counter);
        }
        counters.put("lizard_tail_active", Boolean.valueOf(tail != null && tail.counter == -1));

        AbstractRelic necronomicon = player.getRelic("Necronomicon");
        counters.put("necronomicon_used", Boolean.valueOf(
            necronomicon != null && privateBool(
                necronomicon,
                com.megacrit.cardcrawl.relics.Necronomicon.class,
                "activated")));

        int attacks = 0;
        int skills = 0;
        for (AbstractCard card : AbstractDungeon.actionManager.cardsPlayedThisTurn) {
            if (card.type == AbstractCard.CardType.ATTACK) attacks++;
            if (card.type == AbstractCard.CardType.SKILL) skills++;
        }
        counters.put("cards_played", Integer.valueOf(
            AbstractDungeon.actionManager.cardsPlayedThisTurn.size()));
        counters.put("attacks_played", Integer.valueOf(attacks));
        counters.put("skills_played", Integer.valueOf(skills));

        AbstractRelic pellets = player.getRelic("OrangePellets");
        counters.put("orange_attack", Boolean.valueOf(pellets != null &&
            ((Boolean)ReflectionHacks.getPrivateStatic(
                com.megacrit.cardcrawl.relics.OrangePellets.class, "ATTACK")).booleanValue()));
        counters.put("orange_skill", Boolean.valueOf(pellets != null &&
            ((Boolean)ReflectionHacks.getPrivateStatic(
                com.megacrit.cardcrawl.relics.OrangePellets.class, "SKILL")).booleanValue()));
        counters.put("orange_power", Boolean.valueOf(pellets != null &&
            ((Boolean)ReflectionHacks.getPrivateStatic(
                com.megacrit.cardcrawl.relics.OrangePellets.class, "POWER")).booleanValue()));

        AbstractPower combust = power("Combust");
        counters.put("combust_hp_loss", Integer.valueOf(combust == null ? 0 : privateInt(
            combust, com.megacrit.cardcrawl.powers.CombustPower.class, "hpLoss")));
        AbstractPower deva = power("DevaForm");
        counters.put("deva_form_energy_per_turn", Integer.valueOf(deva == null ? 0 : privateInt(
            deva, com.megacrit.cardcrawl.powers.watcher.DevaPower.class, "energyGainAmount")));
        AbstractPower echo = power("Echo Form");
        counters.put("echo_form_cards_doubled", Integer.valueOf(echo == null ? 0 : privateInt(
            echo, com.megacrit.cardcrawl.powers.EchoPower.class, "cardsDoubledThisTurn")));
        AbstractPower panache = power("Panache");
        counters.put("panache_counter", Integer.valueOf(panache == null ? 0 : panache.amount));
        HashMap<String, Object> powerOverrides = new HashMap<String, Object>();
        if (panache != null) {
            // Panache.amount is the cards-remaining countdown.  The native
            // PANACHE status value is the private damage, so export both.
            powerOverrides.put("Panache", Integer.valueOf(privateInt(
                panache, com.megacrit.cardcrawl.powers.PanachePower.class, "damage")));
        }

        counters.put("cards_discarded", Integer.valueOf(
            com.megacrit.cardcrawl.actions.GameActionManager.totalDiscardedThisTurn));
        counters.put("times_damaged", Integer.valueOf(player.damagedThisCombat));

        int[] bombs = new int[] {0, 0, 0};
        ArrayList<HashMap<String, Object>> bombInstances = new ArrayList<HashMap<String, Object>>();
        for (AbstractPower candidate : player.powers) {
            if (!candidate.ID.startsWith("TheBomb")) continue;
            if (candidate.amount < 1 || candidate.amount > 3) {
                throw new IllegalStateException("The Bomb countdown outside simulator contract: "
                    + candidate.amount);
            }
            int damage = privateInt(candidate, com.megacrit.cardcrawl.powers.TheBombPower.class, "damage");
            if (damage < 0 || bombs[candidate.amount - 1] > Integer.MAX_VALUE - damage) {
                throw new IllegalStateException("The Bomb damage outside simulator integer range");
            }
            bombs[candidate.amount - 1] += damage;
            HashMap<String, Object> instance = new HashMap<String, Object>();
            instance.put("id", candidate.ID);
            instance.put("turns", Integer.valueOf(candidate.amount));
            instance.put("damage", Integer.valueOf(damage));
            bombInstances.add(instance);
        }
        state.put("simulator_bombs", bombInstances);
        counters.put("bomb_1", Integer.valueOf(bombs[0]));
        counters.put("bomb_2", Integer.valueOf(bombs[1]));
        counters.put("bomb_3", Integer.valueOf(bombs[2]));
        counters.put("last_attack_unblocked_damage", Integer.valueOf(lastAttackUnblockedDamage));
        state.put("simulator_combat_counters", counters);
        state.put("simulator_player_power_overrides", powerOverrides);
        AbstractPower berserk = power("Berserk");
        // EnergyManager.energyMaster already includes boss relics and temporary
        // Slaver's Collar combat changes. Berserk grants its amount separately
        // at turn start, while native folds that recurring grant into
        // energyPerTurn, so both parts are required for an exact root.
        state.put("simulator_energy_per_turn", Integer.valueOf(
            player.energy.energyMaster + (berserk == null ? 0 : berserk.amount)));
        state.put("simulator_parity_export", VERSION);
    }

    /**
     * Java's public lastDamageTaken is overwritten by HP-loss and self-damage.
     * The native field instead remembers only the latest NORMAL monster attack after
     * block/relic/power mitigation (it is consumed by Shelled Parasite's heal).
     * Track that exact subset without changing damage, queues, or RNG.
     */
    @SpirePatch(clz = AbstractPlayer.class, method = "damage")
    public static final class DamagePatch {
        @SpirePostfixPatch
        public static void afterDamage(AbstractPlayer __instance, DamageInfo info) {
            if (info.owner instanceof AbstractMonster && info.type == DamageInfo.DamageType.NORMAL) {
                lastAttackUnblockedDamage = __instance.lastDamageTaken;
            }
        }
    }

    @SpirePatch(clz = AbstractPlayer.class, method = "preBattlePrep")
    public static final class PreBattlePatch {
        @SpirePostfixPatch
        public static void afterPreBattle(AbstractPlayer __instance) {
            lastAttackUnblockedDamage = 0;
        }
    }

    private static String combatSelectionContext(AbstractGameAction action) {
        if (action == null) return null;
        String name = action.getClass().getSimpleName();
        if ("ArmamentsAction".equals(name)) return "ARMAMENTS";
        if ("CodexAction".equals(name)) return "CODEX";
        if ("DiscoveryAction".equals(name)) return "DISCOVERY";
        if ("ChooseOneColorless".equals(name)) return "DISCOVERY";
        if ("DualWieldAction".equals(name)) return "DUAL_WIELD";
        if ("ExhumeAction".equals(name)) return "EXHUME";
        if ("ForethoughtAction".equals(name)) return "FORETHOUGHT";
        if ("GamblingChipAction".equals(name)) return "GAMBLE";
        if ("DiscardPileToTopOfDeckAction".equals(name)) return "HEADBUTT";
        if ("MeditateAction".equals(name)) return "MEDITATE";
        if ("NightmareAction".equals(name)) return "NIGHTMARE";
        if ("RecycleAction".equals(name)) return "RECYCLE";
        if ("SkillFromDeckToHandAction".equals(name)) return "SECRET_TECHNIQUE";
        if ("AttackFromDeckToHandAction".equals(name)) return "SECRET_WEAPON";
        // The shipped Seek card uses BetterDrawPileToHandAction. SeekAction is
        // an unused legacy implementation; keep both source names explicit.
        if ("BetterDrawPileToHandAction".equals(name)) return "SEEK";
        if ("SeekAction".equals(name)) return "SEEK";
        if ("SetupAction".equals(name)) return "SETUP";
        if ("PutOnDeckAction".equals(name)) return "WARCRY";
        if ("BetterDiscardPileToHandAction".equals(name)) {
            boolean setCost = privateBool(
                action,
                com.megacrit.cardcrawl.actions.common.BetterDiscardPileToHandAction.class,
                "setCost");
            return setCost ? "LIQUID_MEMORIES_POTION" : "HOLOGRAM";
        }
        if ("ExhaustAction".equals(name)) {
            boolean anyNumber = privateBool(
                action, com.megacrit.cardcrawl.actions.common.ExhaustAction.class, "anyNumber");
            return anyNumber || action.amount > 1 ? "EXHAUST_MANY" : "EXHAUST_ONE";
        }
        return null;
    }

    private static boolean unfinishedRelic(String id, Class<?> owner, String field) {
        AbstractRelic relic = AbstractDungeon.player.getRelic(id);
        return relic != null && !privateBool(relic, owner, field);
    }

    private static String overworldSelectionContext(GridCardSelectScreen screen) {
        if (unfinishedRelic("Astrolabe", com.megacrit.cardcrawl.relics.Astrolabe.class,
                            "cardsSelected")) return "TRANSFORM_UPGRADE";
        if (unfinishedRelic("Bottled Flame", com.megacrit.cardcrawl.relics.BottledFlame.class,
                            "cardSelected")) return "BOTTLE";
        if (unfinishedRelic("Bottled Lightning", com.megacrit.cardcrawl.relics.BottledLightning.class,
                            "cardSelected")) return "BOTTLE";
        if (unfinishedRelic("Bottled Tornado", com.megacrit.cardcrawl.relics.BottledTornado.class,
                            "cardSelected")) return "BOTTLE";
        if (unfinishedRelic("DollysMirror", com.megacrit.cardcrawl.relics.DollysMirror.class,
                            "cardSelected")) return "DUPLICATE";

        // Bonfire uses a purge UI, but the simulator's selection also grants
        // a rarity-dependent reward. Its context must precede generic flags.
        if (AbstractDungeon.getCurrRoom().event instanceof com.megacrit.cardcrawl.events.shrines.Bonfire)
            return "BONFIRE_SPIRITS";
        if (screen.forUpgrade) return "UPGRADE";
        if (screen.forPurge) return "REMOVE";
        if (screen.forTransform) return "TRANSFORM";

        AbstractEvent event = AbstractDungeon.getCurrRoom().event;
        if (event != null) {
            String name = event.getClass().getSimpleName();
            // DrugDealer opens a two-card grid with every generic flag false;
            // the event, not GridCardSelectScreen, performs the transforms.
            if ("DrugDealer".equals(name)) return "TRANSFORM";
            if ("Bonfire".equals(name)) return "BONFIRE_SPIRITS";
            if ("Duplicator".equals(name)) return "DUPLICATE";
            if ("TheLibrary".equals(name) || "NoteForYourself".equals(name)
                    || "NeowEvent".equals(name)) return "OBTAIN";
        }
        return null;
    }

    public static void augmentGridState(HashMap<String, Object> state) {
        GridCardSelectScreen screen = AbstractDungeon.gridSelectScreen;
        // Single-card upgrade/transform/purge previews clear selectedCards.
        // The actual original is hoveredCard, not the animated replacement or
        // upgraded preview. Preserve this distinct, not-yet-confirmed identity.
        AbstractCard confirming = null;
        if (screen.confirmScreenUp && (screen.forUpgrade || screen.forTransform || screen.forPurge)) {
            confirming = ReflectionHacks.getPrivate(screen, GridCardSelectScreen.class, "hoveredCard");
            if (confirming == null || !screen.targetGroup.group.contains(confirming))
                throw new IllegalStateException("GRID confirmation has no actual source card");
        }
        state.put("confirmation_card_uuid", confirming == null ? null : confirming.uuid.toString());
        String context = combatSelectionContext(AbstractDungeon.actionManager.currentAction);
        if (context == null) context = overworldSelectionContext(AbstractDungeon.gridSelectScreen);
        state.put("selection_context", context == null ? "UNMAPPED" : context);
        AbstractGameAction action = AbstractDungeon.actionManager.currentAction;
        AbstractEvent event = AbstractDungeon.getCurrRoom().event;
        state.put("selection_source", action != null
            ? action.getClass().getName()
            : event != null ? event.getClass().getName() : AbstractDungeon.getCurrRoom().getClass().getName());
        state.put("simulator_parity_export", VERSION);
    }

    public static void augmentHandSelectState(HashMap<String, Object> state) {
        AbstractGameAction action = AbstractDungeon.actionManager.currentAction;
        String context = combatSelectionContext(action);
        state.put("selection_context", context == null ? "UNMAPPED" : context);
        state.put("selection_source", action == null ? "NONE" : action.getClass().getName());
        state.put("any_number", Boolean.valueOf(privateBool(
            AbstractDungeon.handCardSelectScreen,
            com.megacrit.cardcrawl.screens.select.HandCardSelectScreen.class, "anyNumber")));
        state.put("simulator_parity_export", VERSION);
    }

    public static void augmentCardRewardState(HashMap<String, Object> state) {
        AbstractGameAction action = AbstractDungeon.actionManager.currentAction;
        String context = combatSelectionContext(action);
        state.put("reward_item_present", AbstractDungeon.cardRewardScreen.rItem != null);
        state.put("selection_context", context == null ? "UNMAPPED" : context);
        state.put("selection_source", action == null ? "NONE" : action.getClass().getName());
        boolean discovery = action instanceof com.megacrit.cardcrawl.actions.unique.DiscoveryAction;
        boolean toolbox = action instanceof com.megacrit.cardcrawl.actions.utility.ChooseOneColorless;
        state.put("generated_count", discovery || toolbox ? Integer.valueOf(action.amount) : null);
        state.put("generated_free_for_turn", discovery || toolbox ? Boolean.valueOf(discovery) : null);
        state.put("simulator_parity_export", VERSION);
    }
}
