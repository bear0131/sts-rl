"""Dashboard persistence and preview for the shared card ranking contract."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

from env.core.ironclad_cards import IRONCLAD_CARDS
from env.search.card_ranking import CardRanking, SKIP


class RankingStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def _read(self):
        if not self.path.exists():
            return CardRanking(tuple(IRONCLAD_CARDS) + (SKIP,)), None
        raw = self.path.read_bytes()
        ranking = CardRanking.from_dict(json.loads(raw))
        return ranking, hashlib.sha256(raw).hexdigest()

    def _state(self, ranking, revision):
        return dict(ranking=ranking.to_dict(), revision=revision,
                    saved=revision is not None, path=str(self.path),
                    cards=[dict(id=cid, name=info.name_zh, kind=info.kind,
                                cost=list(info.cost), effect=info.effect_zh)
                           for cid, info in IRONCLAD_CARDS.items()])

    def load(self):
        with self.lock:
            return self._state(*self._read())

    def save(self, body):
        ranking = CardRanking.from_dict(body.get("ranking"))
        with self.lock:
            _, revision = self._read()
            if "base_revision" not in body or body["base_revision"] != revision:
                raise ValueError("其他页面已保存新排名。请先导出当前草稿，再重新读取已保存排名")
            raw = (json.dumps(ranking.to_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Same-directory atomic replacement: an interrupted save cannot
            # leave half a JSON file. Caller-controlled file paths are not used.
            fd, temporary = tempfile.mkstemp(prefix=".ranking-", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            return self._state(ranking, hashlib.sha256(raw).hexdigest())


def preview_ranking(body):
    ranking = CardRanking.from_dict(body.get("ranking"))
    cards = body.get("cards")
    if not isinstance(cards, list) or len(cards) > 5:
        raise ValueError("试选需要至多五张候选牌")
    can_skip = body.get("can_skip", True)
    if type(can_skip) is not bool:
        raise ValueError("can_skip 必须为布尔值")
    index = ranking.choose(cards, can_skip=can_skip)
    return dict(index=index, card=SKIP if index is None else cards[index])
