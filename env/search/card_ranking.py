"""User-authored, context-free Ironclad reward priorities (best first).

Card IDs use the canonical observation vocabulary. Upgrades share a card's
priority; ties choose the first offered slot. This does not rank removals,
upgrades, shops, combat selections, or cards outside the Ironclad catalog.
"""
from dataclasses import dataclass

from env.core.ironclad_cards import IRONCLAD_CARDS

SKIP = "__skip__"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CardRanking:
    order: tuple[str, ...]

    def __post_init__(self):
        expected = set(IRONCLAD_CARDS) | {SKIP}
        if (not isinstance(self.order, tuple)
                or any(not isinstance(item, str) for item in self.order)
                or len(self.order) != len(expected)
                or set(self.order) != expected):
            raise ValueError("排名必须恰好包含全部战士卡牌和一个跳过选项，不能重复或遗漏")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
            raise ValueError("无效的排名文件格式")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("不支持此排名文件版本")
        if data.get("character") != "IRONCLAD" or data.get("upgrade_policy") != "shared":
            raise ValueError("排名必须为战士，且普通与升级版共用排名")
        if not isinstance(data.get("order"), list):
            raise ValueError("order 必须是卡牌 ID 列表")
        return cls(tuple(data["order"]))

    def to_dict(self):
        return dict(schema_version=SCHEMA_VERSION, character="IRONCLAD",
                    upgrade_policy="shared", order=list(self.order))

    def choose(self, cards, *, can_skip=True):
        """Return offered index, or None for skip; reject unknown IDs explicitly.

The caller supplies only legal offers and maps the resulting index back to
its own action codec. Never treat this index as a global action ID.
        """
        priority = {card_id: i for i, card_id in enumerate(self.order)}
        if any(not isinstance(card_id, str) or card_id not in IRONCLAD_CARDS
               for card_id in cards):
            raise ValueError("候选中存在未排名的卡牌；需由调用方显式处理")
        if not cards:
            if can_skip:
                return None
            raise ValueError("没有合法候选且不能跳过")
        best = min(range(len(cards)), key=lambda i: priority[cards[i]])
        return None if can_skip and priority[SKIP] < priority[cards[best]] else best
