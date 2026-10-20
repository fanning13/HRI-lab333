import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

source = Path(sys.argv[1] if len(sys.argv) > 1 else 'input.zip')
with zipfile.ZipFile(source) as archive:
    files = [name for name in archive.namelist() if name.lower().endswith('.csv')]
    df = pd.concat(
        [pd.read_csv(io.BytesIO(archive.read(filename)), encoding='utf-8-sig', keep_default_na=False)
         for filename in files], ignore_index=True
    )
assert files and set(df.session_mode) == {'Result'}, 'Refusing unlabeled/mixed dataset'
cp = df[df.row_type.eq('checkpoint')].copy()
ratings = df[df.row_type.eq('perception')].copy()
assert len(cp) == 48 and len(ratings) == 48
for name in ('initial_correct','final_correct','hint_offered','hint_delivered'):
    cp[name] = cp[name].astype(str).str.lower().map({'true': 1, 'false': 0})
for name in ('confusion_rating','explanation_score','response_latency_ms'):
    cp[name] = pd.to_numeric(cp[name], errors='coerce')
cp['improvement'] = cp['final_correct'] - cp['initial_correct']
ratings['perception_rating'] = pd.to_numeric(ratings.perception_rating, errors='coerce')
print('DATA PROVENANCE: Result')
print(f'{len(files)} CSVs | {df.participant_id.nunique()} IDs | {len(cp)} checkpoints | {len(ratings)} rating records')
print('\nPer-ID means and SD across 8 IDs by condition:')
measures = ['initial_correct','final_correct','improvement','explanation_score','confusion_rating','response_latency_ms']
per_id = cp.groupby(['participant_id','condition'])[measures].mean().reset_index()
print(per_id.groupby('condition')[measures].agg(['mean','std']).round(4).to_string())
print('\nPerception means and SD by condition:')
print(ratings.groupby(['condition','perception_item']).perception_rating.agg(['count','mean','std']).round(4).to_string())
print('\nCondition-level counts:')
print(cp.groupby('condition')[['initial_correct','final_correct','hint_offered','hint_delivered']].sum().to_string())
print('\nNudge counts:')
print(cp.groupby(['condition','nudge']).size().to_string())
