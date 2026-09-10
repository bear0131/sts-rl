Random mixed-deck card surface sensitivity
card=Twin Strike version=base trials=3
Each row keeps the same random deck, state, shuffle seed, and simulator target.
Only the target card surface in the input prefix changes.

variant | frozen target READ_BASE_DAMAGE | edited surface damage | teacher-forced read | matches edited | matches frozen target | TF changed vs baseline | AR changed vs baseline
5x2 | [5, 5, 5] | 5 | [5, 5, 5] | 3/3 | 3/3 | 0/3 | 0/3
7x3 | [5, 5, 5] | 7 | [7, 7, 3] | 2/3 | 0/3 | 3/3 | 3/3
