"""Export review coverage and explicitly labelled offline resolution metrics."""
import json
from pathlib import Path


def export_report(report_path):
    path = Path(report_path)
    report = json.loads(path.read_text())
    rows = report['results']
    completed = [r for r in rows if r['run_status'] == 'completed']
    reviewed = [r for r in completed if all(isinstance(r['human_review'].get(k), bool)
                for k in ('correctness', 'grounding', 'appropriate_behavior', 'resolved_without_handoff'))]
    resolved = [r for r in reviewed if all(r['human_review'][k] for k in
                ('correctness', 'grounding', 'appropriate_behavior', 'resolved_without_handoff'))
                and not any((t.get('support_assessment') or {}).get('needs_human', False) for t in r['turns'])]
    def rate(n, d):
        return f'{n}/{d} ({100*n/d:.1f}%)' if d else 'Not measured: no reviewed cases'
    lines = ['# Restaurant assistant evaluation', '', f"Run: {report['run_id']}",
             f"Run status: {report['run_status']}",
             f"Completed: {len(completed)}/{report['cases_planned']}",
             f"Human review coverage: {len(reviewed)}/{len(completed)}", '',
             '## Metrics', '',
             '- Automatic screen pass: '+rate(sum(r['automatic_checks_pass'] for r in rows), len(rows)),
             '- Faithfulness (human-reviewed grounding): '+rate(sum(r['human_review']['grounding'] for r in reviewed),len(reviewed)),
             '- Offline first-contact resolution proxy: '+rate(len(resolved),len(reviewed)), '',
             'A case is one simulated contact, including its setup turns. Resolution requires correct, grounded, appropriate answers, reviewer-confirmed resolution, and no handoff. Unreviewed cases are excluded; coverage is shown above. This is not a production customer-resolution metric.', '',
             'Targets proposed for this demo: 90% faithfulness and 80% offline resolution. These are project goals, not measured achievements or calibrated thresholds.', '',
             'Fallback uses answer status, citations, and an LLM evidence check for generated RAG answers, not calibrated confidence. Pending cases enter a local demo queue; no external support service is connected.', '',
             '## Per-case outcomes and failure analysis', '']
    for row in rows:
        lines += [f"### {row['case']['id']}: {row['case']['question']}",
                  'Expected: '+row['case']['expected_answer'],
                  'Actual: '+(row['turns'][-1]['answer'] if row['turns'] else row.get('error_type','No answer')),
                  'Failed checks: '+', '.join(k for k,v in row.get('checks',{}).items() if v is False),
                  'Reviewer notes: '+(row['human_review'].get('notes') or 'Not supplied'), '']
    output = path.with_suffix('.md')
    output.write_text('\n\n'.join(lines))
    return output
