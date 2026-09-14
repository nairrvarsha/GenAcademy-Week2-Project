import json
from pathlib import Path
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
st.subheader('Local human-review queue')
st.caption('Demo queue only. No external team receives these requests.')
items=sorted((ROOT/'data/human_review_queue').glob('*.json'))
if not items: st.write('No queued requests.')
for path in items:
    item=json.loads(path.read_text())
    with st.expander(item['question']+' — '+item['status']):
        st.write(item['answer']); st.caption(item['reason'])
        resolution=st.text_area('Human response / resolution',value=item['resolution'],key=item['id'])
        if st.button('Mark resolved',key='resolve-'+item['id']):
            if not resolution.strip(): st.error('Enter a resolution first.')
            else:
                item.update(status='resolved',resolution=resolution)
                path.write_text(json.dumps(item,indent=2)+'\n'); st.success('Saved locally.')
