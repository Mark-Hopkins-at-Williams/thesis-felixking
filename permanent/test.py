import statistics
scores = []
scores.append((("abc_Latn", "bcd_Latn"), ("BLEU = 25.22 86.0/42.9/14.6/7.5 (BP = 1.000 ratio = 1.000 hyp_len = 43 ref_len = 43)", "chrF2 = 51.13")))
scores.append((("cde_Latn", "def_Latn"), ("BLEU = 25.22 86.0/42.9/14.6/7.5 (BP = 1.000 ratio = 1.000 hyp_len = 43 ref_len = 43)", "chrF2 = 51.13")))
scores.append((("efg_Latn", "fgh_Latn"), ("BLEU = 25.22 86.0/42.9/14.6/7.5 (BP = 1.000 ratio = 1.000 hyp_len = 43 ref_len = 43)", "chrF2 = 51.13")))
scores.append((("ghi_Latn", "hij_Latn"), ("BLEU = 25.22 86.0/42.9/14.6/7.5 (BP = 1.000 ratio = 1.000 hyp_len = 43 ref_len = 43)", "chrF2 = 51.13")))

average_bleu_len = statistics.mean([len(b) for ((s, t), (b, c)) in scores])
print(average_bleu_len)

# output = "heading" + ''.join([f"\n\t{t}-{s}:\n\t· {b}\n\t· {c}" for ((t, s), (b, c)) in scores])

# print(output)

