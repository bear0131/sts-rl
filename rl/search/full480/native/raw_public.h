#pragma once
#include <array>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <type_traits>
namespace {
constexpr int RAW_REFERENCE_DENSE=192;
std::atomic<std::uint64_t> rawPublicRows{0},rawPublicFallbacks{0};
static bool rawPublicVerify=std::getenv("VERIFY_RAW_PUBLIC") && std::atoi(std::getenv("VERIFY_RAW_PUBLIC"));
const int RAW_PUBLIC_WORDS=192+(rawPublicVerify?112:0);
inline void words64(int *p,std::uint64_t x){std::memcpy(p,&x,8);}
bool rawNormalSnapshot(const sts::BattleContext &bc,int owner,int offset,int cap,int *r,CUdeviceptr address){
    using namespace sts;const auto &p=bc.player;
    const auto orderCount=bc.onUseRelicOrder.size()+bc.onEndTurnRelicOrder.size()+bc.onPostDrawRelicOrder.size()+p.endTurnPowerOrder.size();
    r[0]=offset;r[1]=cap;r[2]=owner;r[3]=p.hasRelic<R::RUNIC_DOME>();r[4]=bc.potionCapacity;
    r[5]=int(p.cc);r[6]=int(bc.encounter);r[7]=int(orderCount);r[8]=int(p.bombs.size());r[9]=0;r[10]=0;
    if(p.bombs.size()>16 || orderCount>16 || p.statusMap.size()>24 || bc.potionCapacity>5 || cap>1024){r[9]=1;return false;}
    if(!address)throw std::logic_error("missing mapped state address");
    words64(r+16,address);r[15]=int(p.statusMap.size());
    r[11]=int(bc.onUseRelicOrder.size());r[12]=int(bc.onEndTurnRelicOrder.size());
    r[13]=int(bc.onPostDrawRelicOrder.size());r[14]=int(p.endTurnPowerOrder.size());
    if(!p.bombs.empty())std::memcpy(r+18,p.bombs.data(),p.bombs.size()*sizeof(Player::Bomb));
    int oi=0;auto orders=[&](const auto &v){for(auto id:v)r[50+oi++]=int(id);};
    orders(bc.onUseRelicOrder);orders(bc.onEndTurnRelicOrder);orders(bc.onPostDrawRelicOrder);orders(p.endTurnPowerOrder);
    int pi=0;for(const auto &entry:p.statusMap){r[66+pi*2]=int(entry.first);r[67+pi*2]=entry.second;++pi;}
    r[169]=int(p.orangePelletsCardTypesPlayed.to_ulong());
    for(int i=0;i<5;++i){
        const auto &m=bc.monsters.arr[i];
        r[129+i*4]=m.isTargetable();r[130+i*4]=m.isDeadOrEscaped();
        r[131+i*4]=m.isAlive();r[132+i*4]=m.isEscaping();
        if(m.id!=MonsterId::INVALID&&!r[3]){auto d=m.getMoveBaseDamage(bc);r[114+i*3]=d.damage;r[115+i*3]=m.getIntentDamage(bc,d.damage);r[116+i*3]=d.attackCount;}
        else r[114+i*3]=r[115+i*3]=r[116+i*3]=0;
        r[170+i*2]=(m.id==MonsterId::LOOTER||m.id==MonsterId::MUGGER)?m.thiefSlashCount:0;
        r[171+i*2]=(m.id==MonsterId::LOOTER||m.id==MonsterId::MUGGER)?m.thiefGoldAmount:0;
        r[149+i]=0;
        switch(m.id){
        case MonsterId::LOOTER:case MonsterId::MUGGER:case MonsterId::LAGAVULIN:
        case MonsterId::GREMLIN_WIZARD:case MonsterId::WRITHING_MASS:case MonsterId::TIME_EATER:
        case MonsterId::RED_SLAVER:case MonsterId::BRONZE_ORB:case MonsterId::SPIKER:
        case MonsterId::THE_CHAMP:case MonsterId::BRONZE_AUTOMATON:case MonsterId::BOOK_OF_STABBING:
        case MonsterId::THE_GUARDIAN:case MonsterId::GIANT_HEAD:case MonsterId::AWAKENED_ONE:
        case MonsterId::HEXAGHOST:r[149+i]=!r[3];break;
        default:break;
        }
        r[154+i*3]=m.wasJustApplied<MS::VULNERABLE>();r[155+i*3]=m.wasJustApplied<MS::WEAK>();r[156+i*3]=m.wasJustApplied<MS::RITUAL>();
    }
    return true;
}
void writeRawPublic(const sts::BattleContext &bc,int owner,int offset,int cap,int *r,int *facts,bool verify,CUdeviceptr address){
    bool raw=rawNormalSnapshot(bc,owner,offset,cap,r,address);
    if(!raw){rawPublicFallbacks.fetch_add(1,std::memory_order_relaxed);
        int used=sts::search::writeNormalFacts(bc,owner,facts,reinterpret_cast<float*>(r+16),cap);r[10]=used;
        for(int j=used;j<cap;++j)facts[j*5]=-1;
        if(verify)std::memcpy(r+RAW_REFERENCE_DENSE,r+16,112*sizeof(int));
    }else if(verify){
        int used=sts::search::writeNormalFacts(bc,owner,facts,reinterpret_cast<float*>(r+RAW_REFERENCE_DENSE),cap);r[10]=used;
        for(int j=used;j<cap;++j)facts[j*5]=-1;
    }
    if(verify)rawPublicRows.fetch_add(1,std::memory_order_relaxed);
}
py::dict rawPublicSchema(){
    using namespace sts;
    static_assert(int(PS::THE_BOMB)<128 && int(MS::INVALID)==42);
    // Ask the original accessor for the static status-to-field mapping once.
    Monster m;m.statusBits=~std::uint64_t(0);
    m.artifact=1000;m.blockReturn=1001;m.choked=1002;m.corpseExplosion=1003;m.lockOn=1004;m.mark=1005;
    m.metallicize=1006;m.platedArmor=1007;m.poison=1008;m.regen=1009;m.shackled=1010;m.strength=1011;
    m.vulnerable=1012;m.weak=1013;m.uniquePower0=1014;m.uniquePower1=1015;
    std::vector<int> offsets(64,-2);
    for(int k=0;k<int(MS::INVALID);++k){int v=m.getStatusInternal(MS(k));if(v==1)offsets[k]=-1;
        else if(v>=1000&&v<1016)offsets[k]=17+v-1000;
        else if(v!=0)throw std::logic_error("unknown monster status accessor");}
    py::dict d;d["version"]=3;d["words"]=RAW_PUBLIC_WORDS;d["player_last"]=int(PS::THE_BOMB);
    d["player_special"]=std::vector<int>{int(PS::ARTIFACT),int(PS::DEXTERITY),int(PS::FOCUS),int(PS::STRENGTH)};
    d["monster_invalid"]=int(MonsterId::INVALID);d["monster_status_invalid"]=int(MS::INVALID);
    d["monster_strength"]=int(MS::STRENGTH);d["monster_just"]=std::vector<int>{int(MS::VULNERABLE),int(MS::WEAK),int(MS::RITUAL)};
    d["monster_offsets"]=offsets;
    d["kind_ids"]=std::vector<int>{int(PlayoutKind::Battle),int(PlayoutKind::Player),int(PlayoutKind::Power),int(PlayoutKind::Relic),
        int(PlayoutKind::Card),int(PlayoutKind::Monster),int(PlayoutKind::MonsterPower),int(PlayoutKind::Move),int(PlayoutKind::Potion),
        int(PlayoutKind::Selection),int(PlayoutKind::Counter),int(PlayoutKind::Order)};
    return d;
}
}

namespace {
py::dict mappedStateSchema(){
    sts::BattleContext bc;
    const auto *base=reinterpret_cast<const char*>(&bc);
    std::vector<int> offsets(512,-1),widths(512,0),signs(512,0);
    auto add=[&](int index,const auto &v){
        using V=std::decay_t<decltype(v)>;size_t width=sizeof(V);
        const auto off=reinterpret_cast<const char*>(&v)-base;
        if(width!=1 && width!=2 && width!=4 && width!=8)throw std::logic_error("unsupported mapped field");
        if(off<0||off+width>sizeof(bc)||(off%4)+std::min(width,size_t(4))>4)throw std::logic_error("mapped field alignment");
        offsets[index]=off;widths[index]=std::min(width,size_t(4));signs[index]=std::is_signed_v<V>;
        if(width==8){offsets[index+1]=off+4;widths[index+1]=4;signs[index+1]=0;}
    };
    add(5,bc.player.cc);
    add(6,bc.encounter);
    add(16,bc.ascension);
    add(17,bc.floorNum);
    add(18,bc.turn);
    add(19,bc.inputState);
    add(20,bc.potionCapacity);
    add(21,bc.potionCount);
    add(22,bc.monsters.monsterCount);
    add(23,bc.monsters.monstersAlive);
    add(24,bc.postEnemyTurnBoundary);
    add(25,bc.pocketwatchCardsAtBoundary);
    add(26,bc.endTurnQueued);
    add(27,bc.turnHasEnded);
    add(28,bc.skipMonsterTurn);
    add(29,bc.cards.cardsInHand);
    add(30,bc.cards.knownDrawTop);
    add(31,bc.outcome);
    add(32,bc.player.curHp);
    add(33,bc.player.maxHp);
    add(34,bc.player.block);
    add(35,bc.player.energy);
    add(36,bc.player.gold);
    add(37,bc.player.energyPerTurn);
    add(38,bc.player.cardDrawPerTurn);
    add(39,bc.player.stance);
    add(40,bc.player.orbSlots);
    add(41,bc.player.lastTargetedMonster);
    add(42,bc.player.strength);
    add(43,bc.player.dexterity);
    add(44,bc.player.focus);
    add(45,bc.player.artifact);
    add(46,bc.player.bloodied);
    add(47,bc.player.redSkullActive);
    add(48,bc.player.happyFlowerCounter);
    add(49,bc.player.incenseBurnerCounter);
    add(50,bc.player.inkBottleCounter);
    add(51,bc.player.inserterCounter);
    add(52,bc.player.nunchakuCounter);
    add(53,bc.player.penNibCounter);
    add(54,bc.player.sundialCounter);
    add(55,bc.player.neowLamentRemaining);
    add(56,bc.player.haveUsedNecronomiconThisTurn);
    add(57,bc.player.combustHpLoss);
    add(58,bc.player.devaFormEnergyPerTurn);
    add(59,bc.player.echoFormCardsDoubled);
    add(60,bc.player.panacheCounter);
    add(61,bc.player.cardsPlayedThisTurn);
    add(62,bc.player.attacksPlayedThisTurn);
    add(63,bc.player.skillsPlayedThisTurn);
    add(65,bc.player.cardsDiscardedThisTurn);
    add(66,bc.player.lastAttackUnblockedDamage);
    add(67,bc.player.timesDamagedThisCombat);
    add(68,bc.player.feelNoPainBeforeDarkEmbrace);
    add(72,bc.player.statusBits0);
    add(74,bc.player.statusBits1);
    add(75,bc.player.justAppliedBits);
    add(76,bc.player.relicBits0);
    add(78,bc.player.relicBits1);
    add(215,bc.potions[0]);
    add(216,bc.potions[1]);
    add(217,bc.potions[2]);
    add(218,bc.potions[3]);
    add(219,bc.potions[4]);
    add(220,bc.monsters.arr[0].id);
    add(221,bc.monsters.arr[0].curHp);
    add(222,bc.monsters.arr[0].maxHp);
    add(223,bc.monsters.arr[0].block);
    add(224,bc.monsters.arr[0].strength);
    add(225,bc.monsters.arr[0].isEscapingB);
    add(226,bc.monsters.arr[0].isDyingB);
    add(227,bc.monsters.arr[0].isDeadB);
    add(228,bc.monsters.arr[0].halfDead);
    add(233,bc.monsters.arr[0].escapeNext);
    add(234,bc.monsters.arr[0].inactiveIntentDamage);
    add(235,bc.monsters.arr[0].statusBits);
    add(237,bc.monsters.arr[0].artifact);
    add(238,bc.monsters.arr[0].blockReturn);
    add(239,bc.monsters.arr[0].choked);
    add(240,bc.monsters.arr[0].corpseExplosion);
    add(241,bc.monsters.arr[0].lockOn);
    add(242,bc.monsters.arr[0].mark);
    add(243,bc.monsters.arr[0].metallicize);
    add(244,bc.monsters.arr[0].platedArmor);
    add(245,bc.monsters.arr[0].poison);
    add(246,bc.monsters.arr[0].regen);
    add(247,bc.monsters.arr[0].shackled);
    add(248,bc.monsters.arr[0].strength);
    add(249,bc.monsters.arr[0].vulnerable);
    add(250,bc.monsters.arr[0].weak);
    add(251,bc.monsters.arr[0].uniquePower0);
    add(252,bc.monsters.arr[0].uniquePower1);
    add(253,bc.monsters.arr[0].moveHistory[0]);
    add(254,bc.monsters.arr[0].moveHistory[1]);
    add(255,bc.monsters.arr[0].moveHistory[2]);
    add(260,bc.monsters.arr[0].miscInfo);
    add(261,bc.monsters.arr[0].thiefSlashCount);
    add(262,bc.monsters.arr[0].thiefGoldAmount);
    add(266,bc.monsters.arr[1].id);
    add(267,bc.monsters.arr[1].curHp);
    add(268,bc.monsters.arr[1].maxHp);
    add(269,bc.monsters.arr[1].block);
    add(270,bc.monsters.arr[1].strength);
    add(271,bc.monsters.arr[1].isEscapingB);
    add(272,bc.monsters.arr[1].isDyingB);
    add(273,bc.monsters.arr[1].isDeadB);
    add(274,bc.monsters.arr[1].halfDead);
    add(279,bc.monsters.arr[1].escapeNext);
    add(280,bc.monsters.arr[1].inactiveIntentDamage);
    add(281,bc.monsters.arr[1].statusBits);
    add(283,bc.monsters.arr[1].artifact);
    add(284,bc.monsters.arr[1].blockReturn);
    add(285,bc.monsters.arr[1].choked);
    add(286,bc.monsters.arr[1].corpseExplosion);
    add(287,bc.monsters.arr[1].lockOn);
    add(288,bc.monsters.arr[1].mark);
    add(289,bc.monsters.arr[1].metallicize);
    add(290,bc.monsters.arr[1].platedArmor);
    add(291,bc.monsters.arr[1].poison);
    add(292,bc.monsters.arr[1].regen);
    add(293,bc.monsters.arr[1].shackled);
    add(294,bc.monsters.arr[1].strength);
    add(295,bc.monsters.arr[1].vulnerable);
    add(296,bc.monsters.arr[1].weak);
    add(297,bc.monsters.arr[1].uniquePower0);
    add(298,bc.monsters.arr[1].uniquePower1);
    add(299,bc.monsters.arr[1].moveHistory[0]);
    add(300,bc.monsters.arr[1].moveHistory[1]);
    add(301,bc.monsters.arr[1].moveHistory[2]);
    add(306,bc.monsters.arr[1].miscInfo);
    add(307,bc.monsters.arr[1].thiefSlashCount);
    add(308,bc.monsters.arr[1].thiefGoldAmount);
    add(312,bc.monsters.arr[2].id);
    add(313,bc.monsters.arr[2].curHp);
    add(314,bc.monsters.arr[2].maxHp);
    add(315,bc.monsters.arr[2].block);
    add(316,bc.monsters.arr[2].strength);
    add(317,bc.monsters.arr[2].isEscapingB);
    add(318,bc.monsters.arr[2].isDyingB);
    add(319,bc.monsters.arr[2].isDeadB);
    add(320,bc.monsters.arr[2].halfDead);
    add(325,bc.monsters.arr[2].escapeNext);
    add(326,bc.monsters.arr[2].inactiveIntentDamage);
    add(327,bc.monsters.arr[2].statusBits);
    add(329,bc.monsters.arr[2].artifact);
    add(330,bc.monsters.arr[2].blockReturn);
    add(331,bc.monsters.arr[2].choked);
    add(332,bc.monsters.arr[2].corpseExplosion);
    add(333,bc.monsters.arr[2].lockOn);
    add(334,bc.monsters.arr[2].mark);
    add(335,bc.monsters.arr[2].metallicize);
    add(336,bc.monsters.arr[2].platedArmor);
    add(337,bc.monsters.arr[2].poison);
    add(338,bc.monsters.arr[2].regen);
    add(339,bc.monsters.arr[2].shackled);
    add(340,bc.monsters.arr[2].strength);
    add(341,bc.monsters.arr[2].vulnerable);
    add(342,bc.monsters.arr[2].weak);
    add(343,bc.monsters.arr[2].uniquePower0);
    add(344,bc.monsters.arr[2].uniquePower1);
    add(345,bc.monsters.arr[2].moveHistory[0]);
    add(346,bc.monsters.arr[2].moveHistory[1]);
    add(347,bc.monsters.arr[2].moveHistory[2]);
    add(352,bc.monsters.arr[2].miscInfo);
    add(353,bc.monsters.arr[2].thiefSlashCount);
    add(354,bc.monsters.arr[2].thiefGoldAmount);
    add(358,bc.monsters.arr[3].id);
    add(359,bc.monsters.arr[3].curHp);
    add(360,bc.monsters.arr[3].maxHp);
    add(361,bc.monsters.arr[3].block);
    add(362,bc.monsters.arr[3].strength);
    add(363,bc.monsters.arr[3].isEscapingB);
    add(364,bc.monsters.arr[3].isDyingB);
    add(365,bc.monsters.arr[3].isDeadB);
    add(366,bc.monsters.arr[3].halfDead);
    add(371,bc.monsters.arr[3].escapeNext);
    add(372,bc.monsters.arr[3].inactiveIntentDamage);
    add(373,bc.monsters.arr[3].statusBits);
    add(375,bc.monsters.arr[3].artifact);
    add(376,bc.monsters.arr[3].blockReturn);
    add(377,bc.monsters.arr[3].choked);
    add(378,bc.monsters.arr[3].corpseExplosion);
    add(379,bc.monsters.arr[3].lockOn);
    add(380,bc.monsters.arr[3].mark);
    add(381,bc.monsters.arr[3].metallicize);
    add(382,bc.monsters.arr[3].platedArmor);
    add(383,bc.monsters.arr[3].poison);
    add(384,bc.monsters.arr[3].regen);
    add(385,bc.monsters.arr[3].shackled);
    add(386,bc.monsters.arr[3].strength);
    add(387,bc.monsters.arr[3].vulnerable);
    add(388,bc.monsters.arr[3].weak);
    add(389,bc.monsters.arr[3].uniquePower0);
    add(390,bc.monsters.arr[3].uniquePower1);
    add(391,bc.monsters.arr[3].moveHistory[0]);
    add(392,bc.monsters.arr[3].moveHistory[1]);
    add(393,bc.monsters.arr[3].moveHistory[2]);
    add(398,bc.monsters.arr[3].miscInfo);
    add(399,bc.monsters.arr[3].thiefSlashCount);
    add(400,bc.monsters.arr[3].thiefGoldAmount);
    add(404,bc.monsters.arr[4].id);
    add(405,bc.monsters.arr[4].curHp);
    add(406,bc.monsters.arr[4].maxHp);
    add(407,bc.monsters.arr[4].block);
    add(408,bc.monsters.arr[4].strength);
    add(409,bc.monsters.arr[4].isEscapingB);
    add(410,bc.monsters.arr[4].isDyingB);
    add(411,bc.monsters.arr[4].isDeadB);
    add(412,bc.monsters.arr[4].halfDead);
    add(417,bc.monsters.arr[4].escapeNext);
    add(418,bc.monsters.arr[4].inactiveIntentDamage);
    add(419,bc.monsters.arr[4].statusBits);
    add(421,bc.monsters.arr[4].artifact);
    add(422,bc.monsters.arr[4].blockReturn);
    add(423,bc.monsters.arr[4].choked);
    add(424,bc.monsters.arr[4].corpseExplosion);
    add(425,bc.monsters.arr[4].lockOn);
    add(426,bc.monsters.arr[4].mark);
    add(427,bc.monsters.arr[4].metallicize);
    add(428,bc.monsters.arr[4].platedArmor);
    add(429,bc.monsters.arr[4].poison);
    add(430,bc.monsters.arr[4].regen);
    add(431,bc.monsters.arr[4].shackled);
    add(432,bc.monsters.arr[4].strength);
    add(433,bc.monsters.arr[4].vulnerable);
    add(434,bc.monsters.arr[4].weak);
    add(435,bc.monsters.arr[4].uniquePower0);
    add(436,bc.monsters.arr[4].uniquePower1);
    add(437,bc.monsters.arr[4].moveHistory[0]);
    add(438,bc.monsters.arr[4].moveHistory[1]);
    add(439,bc.monsters.arr[4].moveHistory[2]);
    add(444,bc.monsters.arr[4].miscInfo);
    add(445,bc.monsters.arr[4].thiefSlashCount);
    add(446,bc.monsters.arr[4].thiefGoldAmount);
    py::dict d;d["offsets"]=offsets;d["widths"]=widths;d["signed"]=signs;
    d["battle_bytes"]=sizeof(bc);d["version"]=1;d["packet_words"]=RAW_PUBLIC_WORDS;return d;
}
}
