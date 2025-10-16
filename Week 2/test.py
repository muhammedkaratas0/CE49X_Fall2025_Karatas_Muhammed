dead_loads = [100, 120]     # kN
live_loads = [40, 50, 60]   # kN
load_factors = {'dead': 1.2, 'live': 1.6, 'combo': 1.4}

combos = [d*load_factors['dead'] + l*load_factors['live'] for d in dead_loads for l in live_loads] \
       + [(d+l)*load_factors['combo'] for d in dead_loads for l in live_loads]

print("Max:", max(combos))
print("Exceeding 200:", sum(c > 200 for c in combos))

