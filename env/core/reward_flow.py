"""Sequential reward decisions over the unchanged native reward inventory.

Skipped items remain in native storage until final proceed. Cursors preserve
their physical indices; claiming an item removes it, so the cursor stays put.
Only the active category is revealed. A full potion belt uses a separate
replacement choice, leaving ordinary potion drinks available in its review.
"""
import numpy as np

from . import action_space as A, game_actions as ga


STAGES = ('relic', 'key', 'gold', 'potion', 'replace_potion', 'card', 'done')


class SequentialRewardFlow:
    def __init__(self):
        self.active = False
        self.stage = 'relic'
        self.cursors = dict(relic=0, gold=0, potion=0, key=0)

    def sync(self, env):
        phase = env._phase()
        if phase != 'combat_reward':
            # Bottled relics may open a card selection before returning here.
            if phase != 'card_select':
                self.active = False
            return
        if not self.active:
            self.__init__()
            self.active = True
        info = env.thick.gc.get_reward_info()
        while self.stage != 'done':
            stage = self.stage
            if stage == 'replace_potion':
                return
            if stage == 'card':
                exists = info['card_count'] > 0
            elif stage == 'key':
                exists = bool(info['keys']) and self.cursors['key'] == 0
            else:
                key = {'relic': 'relics', 'potion': 'potions', 'gold': 'gold'}[stage]
                exists = self.cursors[stage] < len(info[key])
            if exists:
                return
            self.stage = STAGES[STAGES.index(stage) + (2 if stage == 'potion' else 1)]

    def _replacement_slots(self, env, index):
        gc = env.thick.gc
        slots = [i for i, name in enumerate(gc.overworld_potions)
                 if name not in ('EMPTY_POTION_SLOT', 'Invalid', 'INVALID')]
        if not slots:
            return []
        # Ask the simulator whether clearing a slot makes THIS offer legal;
        # capacity and Sozu rules have one authoritative implementation.
        probe = gc.clone()
        probe.discard_potion(slots[0])
        allowed = any(str(kind) == 'potion' and int(idx) == index
                      for kind, idx, _ in probe.legal_reward_options())
        return slots if allowed else []

    def mask(self, env):
        self.sync(env)
        mask = np.zeros(env.action_space.n, dtype=bool)
        mask[A._PROCEED] = True  # skip this offer; only 'done' leaves the page
        gc, stage = env.thick.gc, self.stage
        legal = {(str(kind), int(idx)) for kind, idx, _ in gc.legal_reward_options()}
        if stage in ('gold', 'relic', 'potion'):
            index = self.cursors[stage]
            base = {'gold': A._RGOLD, 'relic': A._RRELIC, 'potion': A._RPOTION}[stage]
            if (stage, index) in legal:
                mask[base + index] = True
            if stage == 'potion':
                if self._replacement_slots(env, index):
                    mask[base + index] = True
                for potion in gc.legal_overworld_potions():
                    mask[A._POTION + int(potion['slot'])] = True
        elif stage == 'key' and ('key', 0) in legal:
            mask[A._RKEY] = True
        elif stage == 'replace_potion':
            for slot in self._replacement_slots(env, self.cursors['potion']):
                mask[A._POTION + slot] = True
        elif stage == 'card':
            for kind, _, item in gc.legal_reward_options():
                if str(kind) == 'card':
                    mask[A._RCARD + int(item)] = True
            mask[A._RSKIP] = True
            mask[A._PROCEED] = False  # use the existing explicit card-skip head
        # Sapphire competes with the last relic. Its identity is public here.
        if stage == 'relic':
            info = gc.get_reward_info()
            if (self.cursors['relic'] == len(info['relics']) - 1
                    and 'SAPPHIRE_KEY' in info['keys'] and ('key', 0) in legal):
                mask[A._RKEY] = True
        return mask

    def project(self, env, observation):
        self.sync(env)
        marker = np.zeros(len(STAGES), np.float32)
        if env._phase() == 'combat_reward':
            stage = self.stage
            marker[STAGES.index(stage)] = 1.
            active = 'potion' if stage == 'replace_potion' else stage
            for kind, keys in (
                ('gold', ('reward_gold',)), ('relic', ('reward_relic_ids',)),
                ('potion', ('reward_potion_ids',)),
                ('card', ('offer_card_ids', 'offer_card_upgrades')),
            ):
                for key in keys:
                    value = observation[key]
                    if kind != active:
                        value.fill(0)
                    elif kind != 'card':
                        index = self.cursors[kind]
                        value[:index] = 0
                        value[index + 1:] = 0
            if active != 'key':
                keep_sapphire = (active == 'relic'
                    and self.cursors['relic'] == len(env.thick.gc.get_reward_info()['relics']) - 1)
                observation['reward_keys'][0] = 0
                if not keep_sapphire:
                    observation['reward_keys'][1] = 0
        if 'reward_stage' in observation:
            observation['reward_stage'][...] = marker
        else:
            observation['reward_stage'] = marker
        return observation

    def step(self, env, action):
        mask = self.mask(env)
        if not 0 <= action < len(mask) or not mask[action]:
            raise ValueError('illegal sequential reward decision')
        stage = self.stage
        kind = 'potion' if stage == 'replace_potion' else stage
        event = dict(stage=stage, kind=kind, resolved=False, taken=False,
                     decision=stage in ('gold', 'relic', 'potion'),
                     available=bool(mask[{'gold': A._RGOLD, 'relic': A._RRELIC,
                                          'potion': A._RPOTION}.get(kind, 0)
                                         + self.cursors.get(kind, 0)]),
                     key_exchanged=False)
        if stage == 'replace_potion':
            event['available'] = bool(mask[A._POTION:A._POTION + A.P_MAX].any())
        semantic = None
        if stage == 'done':
            semantic = ga.proceed()
        elif stage == 'card':
            semantic = ga.skip_cards() if action == A._RSKIP else ga.reward_card(action - A._RCARD)
        elif action == A._PROCEED:
            self.cursors[kind] += 1
            event['resolved'] = True
            if stage == 'relic' and env.thick.gc.get_reward_info()['keys'] == ['SAPPHIRE_KEY']:
                if self.cursors[kind] == len(env.thick.gc.get_reward_info()['relics']):
                    self.cursors['key'] = 1
            if stage == 'replace_potion':
                self.stage = 'potion'
        elif action == A._RKEY:
            semantic = ga.claim(ga.CLAIM_KEY, 0)
            self.cursors['key'] = 1
            event.update(resolved=True, taken=stage == 'key', key_exchanged=stage == 'relic')
        elif stage == 'replace_potion':
            env.thick.gc.discard_potion(action - A._POTION)
            semantic = ga.claim(ga.CLAIM_POTION, self.cursors['potion'])
            self.stage = 'potion'
            event.update(resolved=True, taken=True, available=True)
        elif stage == 'potion' and A._POTION <= action < A._POTION + A.P_MAX:
            semantic = ga.use_potion(action - A._POTION, None)
        else:
            index = self.cursors[kind]
            legal = {(str(k), int(i)) for k, i, _ in env.thick.gc.legal_reward_options()}
            if stage == 'potion' and ('potion', index) not in legal:
                self.stage = 'replace_potion'
            else:
                semantic = ga.claim(kind, index)
                event.update(resolved=True, taken=True)
        if semantic is None:
            env._invalidate_backend_cache()
            self.sync(env)
            result = env._post_step(0., False, action_id=action)
            result[4]['simulator_transition'] = False
        else:
            _canon, reward, terminated, info = env._backend_step(semantic)
            self.sync(env)
            result = env._post_step(reward, terminated, info, action_id=action)
        result[4]['reward_flow'] = event
        return result

    def describe_action(self, action):
        if action == A._PROCEED:
            return '离开奖励界面' if self.stage == 'done' else '跳过当前奖励'
        if self.stage == 'replace_potion' and A._POTION <= action < A._POTION + A.P_MAX:
            return f'用奖励药水替换持有药水槽 {action - A._POTION + 1}'
        if self.stage == 'potion' and A._RPOTION <= action < A._RPOTION + A.REWARD_POTION:
            return '领取当前药水（满槽时选择替换）'
        return None

    def render(self, env):
        self.sync(env)
        info = env.thick.gc.get_reward_info()
        stage = self.stage
        label = dict(relic='遗物', key='钥匙', gold='金币', potion='药水',
                     replace_potion='选择替换的药水', card='卡牌', done='奖励处理完成')[stage]
        if stage == 'card':
            offer = env.thick.gc.get_card_reward()
        elif stage == 'key':
            offer = info['keys']
        elif stage == 'done':
            offer = ''
        else:
            kind = 'potion' if stage == 'replace_potion' else stage
            key = dict(relic='relics', gold='gold', potion='potions')[kind]
            offer = info[key][self.cursors[kind]]
        held = f'\n持有药水：{list(env.thick.gc.overworld_potions)}' if stage in ('potion', 'replace_potion') else ''
        return f'{label}：{offer}{held}'
