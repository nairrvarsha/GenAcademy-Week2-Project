"""Conservative fallback policy and local demo review queue (no external delivery)."""
import json
from datetime import datetime, timezone
from uuid import uuid4
from rag_store import ROOT


def assess_support(status, sources):
    # Evidence sufficiency, not a calibrated probability or a similarity threshold.
    if status == 'answered' and sources:
        return {'level': 'supported', 'needs_human': False,
                'reason': 'Answer passed citation checks; factual correctness still requires evaluation.'}
    return {'level': 'insufficient', 'needs_human': True,
            'reason': 'Missing evidence, clarification needed, or answer validation failed.'}


def queue_review(question, answer, assessment, directory=None):
    directory = directory or ROOT / 'data/human_review_queue'
    directory.mkdir(parents=True, exist_ok=True)
    item = {'id': uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'pending', 'question': question, 'answer': answer,
            'reason': assessment['reason'], 'resolution': ''}
    (directory / (item['id'] + '.json')).write_text(json.dumps(item, indent=2)+'\n')
    return item['id']
