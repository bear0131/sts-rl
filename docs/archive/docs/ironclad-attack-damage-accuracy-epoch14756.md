# Ironclad Attack damage accuracy

- Checkpoint: `rl\runs\rules-anon-v5-structure-reweight-dist4-20260826-current\last-strike-plus.pt` (epoch 14756)
- Test state: energy=4, block=10, strength=0, enemy block=0, vulnerable=0
- Fixed base-damage reads: 49/84 (58.33%)
- Base-version reads: 24/41 (58.54%); exact versions: 12/29 (41.38%)
- Upgraded-version reads: 25/43 (58.14%); exact versions: 11/29 (37.93%)
- Exact fixed-damage versions: 23/58 (39.66%)
- Exact fixed-damage cards (base and upgrade both correct): 4/29 (13.79%)
- Teacher-forced HP-damage outputs: 90/90 (100.00%)
- Exact HP-damage versions: 64/64 (100.00%)
- Dynamic-formula versions without READ_BASE_DAMAGE: 6

`READ_BASE_DAMAGE` is the primary metric: it tests whether the card-surface damage was copied into the local calculation. HP-damage output is auxiliary because teacher forcing supplies the gold calculation prefix before it.

| Card | Version | Surface | Base target | Base prediction | Base | HP damage target | HP prediction | HP |
|---|---:|---:|---:|---:|:---:|---:|---:|:---:|
| Bash | base | 8 | 8 | 0 | FAIL | 8 | 8 | PASS |
| Bash | + | 10 | 10 | 10 | PASS | 10 | 10 | PASS |
| Strike_R | base | 6 | 6 | 0 | FAIL | 6 | 6 | PASS |
| Strike_R | + | 9 | 9 | 0 | FAIL | 9 | 9 | PASS |
| Anger | base | 6 | 6 | 6 | PASS | 6 | 6 | PASS |
| Anger | + | 8 | 8 | 2 | FAIL | 8 | 8 | PASS |
| Body Slam | base | dynamic | - | - | n/a | 10 | 10 | PASS |
| Body Slam | + | dynamic | - | - | n/a | 10 | 10 | PASS |
| Clash | base | 14 | 14 | 0 | FAIL | 14 | 14 | PASS |
| Clash | + | 18 | 18 | 0 | FAIL | 18 | 18 | PASS |
| Cleave | base | 8 | 8 | 0 | FAIL | 8 | 8 | PASS |
| Cleave | + | 11 | 11 | 0 | FAIL | 11 | 11 | PASS |
| Clothesline | base | 12 | 12 | 0 | FAIL | 12 | 12 | PASS |
| Clothesline | + | 14 | 14 | 0 | FAIL | 14 | 14 | PASS |
| Headbutt | base | 9 | 9 | 9 | PASS | 9 | 9 | PASS |
| Headbutt | + | 12 | 12 | 0 | FAIL | 12 | 12 | PASS |
| Heavy Blade | base | 14 | 14 | 14 | PASS | 14 | 14 | PASS |
| Heavy Blade | + | 14 | 14 | 14 | PASS | 14 | 14 | PASS |
| Iron Wave | base | 5 | 5 | 5 | PASS | 5 | 5 | PASS |
| Iron Wave | + | 7 | 7 | 7 | PASS | 7 | 7 | PASS |
| Perfected Strike | base | 6 + dynamic | - | - | n/a | 26 | 26 | PASS |
| Perfected Strike | + | 6 + dynamic | - | - | n/a | 36 | 36 | PASS |
| Pommel Strike | base | 9 | 9 | 0 | FAIL | 9 | 9 | PASS |
| Pommel Strike | + | 10 | 10 | 10 | PASS | 10 | 10 | PASS |
| Sword Boomerang | base | 3 x 3 | 3, 3, 3 | 3, 3, 3 | PASS | 3, 3, 3 | 3, 3, 3 | PASS |
| Sword Boomerang | + | 3 x 4 | 3, 3, 3, 3 | 0, 3, 3, 3 | FAIL | 3, 3, 3, 3 | 3, 3, 3, 3 | PASS |
| Thunderclap | base | 4 | 4 | 4 | PASS | 4 | 4 | PASS |
| Thunderclap | + | 7 | 7 | 0 | FAIL | 7 | 7 | PASS |
| Twin Strike | base | 5 x 2 | 5, 5 | 5, 5 | PASS | 5, 5 | 5, 5 | PASS |
| Twin Strike | + | 7 x 2 | 7, 7 | 0, 7 | FAIL | 7, 7 | 7, 7 | PASS |
| Wild Strike | base | 12 | 12 | 0 | FAIL | 12 | 12 | PASS |
| Wild Strike | + | 17 | 17 | 0 | FAIL | 17 | 17 | PASS |
| Blood For Blood | base | 18 | 18 | 8 | FAIL | 18 | 18 | PASS |
| Blood For Blood | + | 22 | 22 | 0 | FAIL | 22 | 22 | PASS |
| Carnage | base | 20 | 20 | 20 | PASS | 20 | 20 | PASS |
| Carnage | + | 28 | 28 | 20 | FAIL | 28 | 28 | PASS |
| Dropkick | base | 5 | 5 | 5 | PASS | 5 | 5 | PASS |
| Dropkick | + | 8 | 8 | 8 | PASS | 8 | 8 | PASS |
| Hemokinesis | base | 15 | 15 | 5 | FAIL | 15 | 15 | PASS |
| Hemokinesis | + | 20 | 20 | 20 | PASS | 20 | 20 | PASS |
| Pummel | base | 2 x 4 | 2, 2, 2, 2 | 0, 2, 2, 2 | FAIL | 2, 2, 2, 2 | 2, 2, 2, 2 | PASS |
| Pummel | + | 2 x 5 | 2, 2, 2, 2, 2 | 0, 2, 2, 2, 2 | FAIL | 2, 2, 2, 2, 2 | 2, 2, 2, 2, 2 | PASS |
| Rampage | base | 8 | 8 | 5 | FAIL | 8 | 8 | PASS |
| Rampage | + | 8 | 8 | 8 | PASS | 8 | 8 | PASS |
| Reckless Charge | base | 7 | 7 | 0 | FAIL | 7 | 7 | PASS |
| Reckless Charge | + | 10 | 10 | 10 | PASS | 10 | 10 | PASS |
| Searing Blow | base | dynamic | - | - | n/a | 12 | 12 | PASS |
| Searing Blow | + | dynamic | - | - | n/a | 16 | 16 | PASS |
| Sever Soul | base | 16 | 16 | 16 | PASS | 16 | 16 | PASS |
| Sever Soul | + | 22 | 22 | 22 | PASS | 22 | 22 | PASS |
| Uppercut | base | 13 | 13 | 0 | FAIL | 13 | 13 | PASS |
| Uppercut | + | 13 | 13 | 0 | FAIL | 13 | 13 | PASS |
| Whirlwind | base | 5 x X | 5, 5, 5, 5 | 5, 5, 5, 5 | PASS | 5, 5, 5, 5 | 5, 5, 5, 5 | PASS |
| Whirlwind | + | 8 x X | 8, 8, 8, 8 | 0, 8, 8, 8 | FAIL | 8, 8, 8, 8 | 8, 8, 8, 8 | PASS |
| Bludgeon | base | 32 | 32 | 0 | FAIL | 32 | 32 | PASS |
| Bludgeon | + | 42 | 42 | 0 | FAIL | 42 | 42 | PASS |
| Feed | base | 10 | 10 | 10 | PASS | 10 | 10 | PASS |
| Feed | + | 12 | 12 | 0 | FAIL | 12 | 12 | PASS |
| Fiend Fire | base | 7 + dynamic | 7, 7, 7, 7 | 0, 7, 7, 7 | FAIL | 7, 7, 7, 7 | 7, 7, 7, 7 | PASS |
| Fiend Fire | + | 10 + dynamic | 10, 10, 10, 10 | 10, 10, 10, 10 | PASS | 10, 10, 10, 10 | 10, 10, 10, 10 | PASS |
| Immolate | base | 21 | 21 | 0 | FAIL | 21 | 21 | PASS |
| Immolate | + | 28 | 28 | 1 | FAIL | 28 | 28 | PASS |
| Reaper | base | 4 | 4 | 0 | FAIL | 4 | 4 | PASS |
| Reaper | + | 5 | 5 | 5 | PASS | 5 | 5 | PASS |
