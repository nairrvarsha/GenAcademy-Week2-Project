"""LLM intent planning followed by validated, deterministic dataset queries."""

import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from rag_store import ROOT
from hybrid_retriever import validate_filters


class QueryFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    locality: str | None = None
    cuisine: str | None = None
    max_cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    valet: bool | None = None


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["count", "list", "rag", "clarify"]
    filters: QueryFilters
    clarification: str | None = None


def structured_context(question, filters=None, *, llm):
    """Ask the model for intent only; never execute model-generated code or SQL."""
    filters = validate_filters(filters)
    rows = json.loads((ROOT / 'data/restaurants.json').read_text())
    catalog = {
        "locality": sorted({r['locality'] for r in rows if r.get('locality')}),
        "cuisine": sorted({c for r in rows for c in r.get('cuisines', [])}),
    }
    system = (ROOT / 'prompts/query_planner.txt').read_text()
    response = llm.invoke([
        SystemMessage(content=system + "\nJSON schema:\n" + json.dumps(QueryPlan.model_json_schema())),
        HumanMessage(content=json.dumps({"question": question, "active_filters": filters,
                                        "allowed_values": catalog}, ensure_ascii=False)),
    ])
    try:
        plan = QueryPlan.model_validate_json(response.content)
    except (ValidationError, TypeError):
        return _source({'clarification': 'I could not reliably interpret the question. Please rephrase it.'})
    if plan.operation == 'clarify':
        return _source({'clarification': plan.clarification or 'Please clarify the filters you want to apply.'})
    proposed = plan.filters.model_dump(exclude_none=True)
    for key in ('locality', 'cuisine'):
        if key in proposed:
            allowed = {v.casefold(): v for v in catalog[key]}
            value = allowed.get(proposed[key].casefold())
            if value is None:
                return _source({'clarification': f'I could not match the requested {key} to our dataset.'})
            proposed[key] = value
    for key, value in proposed.items():
        if key == 'max_cost' and key in filters:
            filters[key] = min(filters[key], value)
        elif key in filters and (str(filters[key]).casefold() != str(value).casefold()):
            return _source({'clarification': f'The requested {key} conflicts with the active filter.'})
        else:
            filters[key] = value
    filters = validate_filters(filters)
    if plan.operation == 'rag':
        return None
    def eligible(row):
        return (
            ('locality' not in filters or (row.get('locality') or '').casefold() == filters['locality'].casefold())
            and ('cuisine' not in filters or filters['cuisine'].casefold() in [c.casefold() for c in row['cuisines']])
            and ('valet' not in filters or row.get('valet') == filters['valet'])
            and ('max_cost' not in filters or (row.get('approx_cost_for_two_inr') is not None
                 and row['approx_cost_for_two_inr'] <= filters['max_cost']))
        )
    # Count outlets, not chunks or unique restaurant names.
    matched = {r['restaurant_id']: r for r in rows if eligible(r)}
    outlets = [{'restaurant_id': r['restaurant_id'], 'name': r['name'], 'locality': r['locality']}
               for r in sorted(matched.values(), key=lambda r: (r['name'], r['restaurant_id']))]
    return _source({'operation': plan.operation, 'count': len(outlets),
                    'filters': filters, 'outlets': outlets, 'scope': 'our dataset'})


def _source(content):
    sources = [{'label': 'S1', 'chunk_id': 'restaurants:structured-query',
                'title': 'Full restaurant dataset', 'content': content,
                'metadata': {'source': 'data/restaurants.json', 'query_type': 'structured'}}]
    return sources, json.dumps(sources, ensure_ascii=False)


def render_structured(source):
    data = source['content']
    if 'clarification' in data:
        return data['clarification'], 'needs_clarification', []
    locality = data['filters'].get('locality')
    where = f' in {locality}' if locality else ''
    cuisine = data['filters'].get('cuisine')
    serving = f' serving {cuisine} cuisine' if cuisine else ''
    text = f"Our dataset contains {data['count']} restaurant outlets{where}{serving} matching your filters. [S1]"
    if data['operation'] == 'list':
        text += '\n\n' + '\n'.join(f"{i}. {r['name']} — {r['locality'] or 'Locality unavailable'} ({r['restaurant_id']})"
                                     for i, r in enumerate(data['outlets'], 1))
    return text, 'answered', ['S1']
