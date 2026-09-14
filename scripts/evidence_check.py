"""An evidence-based model check, not a calibrated confidence probability."""
import json
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from rag_store import ROOT


class EvidenceVerdict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    supported: bool
    reason: str = Field(min_length=1, max_length=1000)


def check_evidence(question, answer, sources, llm):
    try:
        result = llm.invoke([
            SystemMessage(content=(ROOT/'prompts/evidence_check.txt').read_text()
                          + '\nSchema: '+json.dumps(EvidenceVerdict.model_json_schema())),
            HumanMessage(content=json.dumps({'question': question, 'answer': answer,
                                           'sources': sources}, ensure_ascii=False)),
        ])
        verdict = EvidenceVerdict.model_validate_json(result.content)
    except Exception:
        # A failed verifier must never approve an unchecked answer.
        return {'level': 'unverified', 'needs_human': True,
                'reason': 'Evidence verification could not complete.'}
    return {'level': 'supported' if verdict.supported else 'insufficient',
            'needs_human': not verdict.supported, 'reason': verdict.reason}
