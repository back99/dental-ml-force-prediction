import pandas as pd
import numpy as np

dpa_025 = pd.read_csv('dpa_025.csv')
dpa_05  = pd.read_csv('dpa_05.csv')

dpa_025['tooth_num'] = dpa_025['tooth'].str.extract(r'(\d+)').astype(int)
dpa_05['tooth_num']  = dpa_05['tooth'].str.extract(r'(\d+)').astype(int)

targets = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
weights = [0.5, 0.8, 1.0, 1.2, 1.5]

for tooth_label, tooth_num in [('U6', 6), ('U7', 7)]:

    t025 = dpa_025[dpa_025['tooth_num'] == tooth_num].copy()
    t05  = dpa_05[dpa_05['tooth_num'] == tooth_num].copy()

    # 시간대별 평균 delta 계산
    mean_025 = t025.groupby('time_hours')[targets].mean()
    mean_05  = t05.groupby('time_hours')[targets].mean()
    delta    = (mean_05 - mean_025).reset_index()

    for w in weights:
        w_str = str(w).replace('.', '')

        # 개별 0.5mm 행에 delta 적용 → 실제 데이터와 같은 행 수!
        sim = t05.copy()
        sim = sim.merge(delta, on='time_hours', suffixes=('', '_delta'))

        for target in targets:
            sim[target] = sim[target] + w * sim[f'{target}_delta']

        sim = sim.drop(columns=[f'{t}_delta' for t in targets])
        sim['thickness'] = 0.75

        filename = f'sim_075_{tooth_label}_w{w_str}.csv'
        sim.to_csv(filename, index=False)
        print(f"Saved: {filename}  ({len(sim)} rows)")

print("\nDone!")
