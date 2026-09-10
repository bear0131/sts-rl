Random mixed-deck card surface sensitivity
card=Strike_R version=base trials=3
Each row keeps the same random deck, state, shuffle seed, and simulator target.
Only the target card surface in the input prefix changes.

variant | frozen target READ_BASE_DAMAGE | edited surface damage | teacher-forced read | matches edited | matches frozen target | TF changed vs baseline | AR changed vs baseline
6xNone | [6, 6, 6] | 6 | [6, 6, 6] | 3/3 | 3/3 | 0/3 | 0/3
11xNone | [6, 6, 6] | 11 | [3, 5, 7] | 0/3 | 0/3 | 3/3 | 3/3
