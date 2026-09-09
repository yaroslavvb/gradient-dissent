"""Publish a bounded snapshot of completed user-visible Codex messages.

Adapted from animated-groups-fable/tools/export_session_transcript.py.
Reads only event_msg/item_completed UserMessage and AgentMessage records.
Never exports tool activity, reasoning, system/developer input or raw rollout data.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from markdown_it import MarkdownIt

ROLE_TYPES = {'UserMessage': 'user', 'AgentMessage': 'assistant'}
PHASES = {'commentary': 'commentary', 'final_answer': 'final', 'final': 'final'}
PUBLIC_ROOT = 'https://yaroslavvb.github.io/gradient-dissent/'
REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_LOCAL = REPO_ROOT / 'output/pdf/depth-robustness-audit.pdf'
PAPER_PUBLIC = PUBLIC_ROOT + 'significance/depth-robustness-audit.pdf'
CUTOFF_REQUEST = 'export this session and host html on github pages, I had a similar question for the animated groups fable, so reuse what was done there.'


def clean_user_text(text):
    for tag in ['in-app-browser-context', 'environment_context', 'recommended_plugins']:
        text = re.sub(r'<' + tag + r'\b[^>]*>.*?</' + tag + '>', '', text, flags=re.S)
    if '## My request:' in text:
        text = text.split('## My request:', 1)[1]
    reply = re.fullmatch(r'\s*<send_user_message_question_reply>\s*(.*?)\s*</send_user_message_question_reply>\s*', text, re.S)
    if reply:
        entries = json.loads(reply[1])
        text = '\n\n'.join('**Question shown:** ' + x['question'] + '\n\n**Answer:** ' + x['answer'] for x in entries)
    return text.strip()


def visible_messages(path, cutoff=CUTOFF_REQUEST):
    seen = set()
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            payload = event.get('payload', {})
            if event.get('type') != 'event_msg' or payload.get('type') != 'item_completed':
                continue
            item = payload.get('item', {})
            kind = item.get('type')
            if kind not in ROLE_TYPES:
                continue
            identifier = item.get('id')
            if not identifier:
                raise ValueError('Visible item without an ID; cannot safely deduplicate.')
            if identifier in seen:
                continue
            role = ROLE_TYPES[kind]
            if role == 'assistant' and item.get('phase') not in PHASES:
                continue
            seen.add(identifier)
            # This session has text-only visible messages. Never silently copy an
            # unknown attachment or arbitrary local file into a public snapshot.
            parts = item.get('content', [])
            if any(p.get('type') not in ('text', 'Text') for p in parts):
                raise ValueError('Unsupported visible attachment; explicit handling required.')
            text = '\n'.join(p.get('text', '') for p in parts).strip()
            if role == 'user':
                text = clean_user_text(text)
            if text:
                yield {'sourceItemId': identifier, 'role': role,
                       'phase': 'request' if role == 'user' else PHASES[item['phase']],
                       'timestamp': event['timestamp'], 'text': text}
            if role == 'user' and text == cutoff.strip():
                return
    raise ValueError('The publication request was not found; refusing an unbounded export.')


def public_path(value):
    value = unquote(value)
    if value == str(PAPER_LOCAL):
        return PAPER_PUBLIC
    if value.startswith(str(REPO_ROOT / 'docs') + '/'):
        path = Path(value)
        if not path.is_file():
            raise ValueError('Local document link is unavailable.')
        return PUBLIC_ROOT + path.relative_to(REPO_ROOT / 'docs').as_posix()
    if value.startswith(str(REPO_ROOT) + '/'):
        path = Path(value)
        if not path.is_file() or path.relative_to(REPO_ROOT).parts[0] in ('tmp', 'research'):
            raise ValueError('Unresolved local/private source link.')
        return 'https://github.com/yaroslavvb/gradient-dissent/blob/main/' + path.relative_to(REPO_ROOT).as_posix()
    parsed = urlsplit(value)
    if parsed.hostname in ('localhost', '127.0.0.1') or (parsed.hostname or '').endswith('.ts.net'):
        raise ValueError('Unresolved private-host link.')
    if value.startswith('/Users/') or value.startswith('file:'):
        raise ValueError('Unresolved local link.')
    return value


def publish_links(text):
    def citation(match):
        raw = match[1]
        p = re.search(r'path="([^"]+)"', raw)
        if not p:
            raise ValueError('Unrecognized file citation.')
        target = public_path(p[1])
        return '[Research paper (PDF)](' + target + ')'
    text = re.sub(r':codex-file-citation\{([^}]+)\}', citation, text)
    text = re.sub(r'\]\(<?(/Users/[^)>]+)>?\)', lambda m: '](' + public_path(m[1]) + ')', text)
    text = re.sub(r'https?://(?:localhost|127\.0\.0\.1|[\w.-]+\.ts\.net):\d+[^\s<>)]*',
                  lambda m: public_path(m[0]), text)
    return text


def validate_public_text(text):
    if re.search(r'<(?:environment_context|recommended_plugins|system|developer|in-app-browser-context|skills_instructions|permissions)\b', text, re.I):
        raise ValueError('Runtime envelope in visible message.')
    if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:gh[pousr]_|github_pat_|sk-proj-|sk-svcacct-)[A-Za-z0-9_-]{20,}|\bAKIA[A-Z0-9]{16}\b', text):
        raise ValueError('Possible credential in visible message; review required.')
    if re.search(r'/Users/|file://|https?://(?:localhost|127\.0\.0\.1|[\w.-]+\.ts\.net)\b', text):
        raise ValueError('Local path/private host remains in public text.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cutoff-text', default=CUTOFF_REQUEST)
    parser.add_argument('--title', default='Review paper and test claims')
    parser.add_argument('--thread-id', default='01a0874d-fa1d-7ae3-93ee-da1be8e0692a')
    args = parser.parse_args()
    # Read the entire bounded selection and validate it before creating outputs.
    selected = list(visible_messages(args.rollout, args.cutoff_text))
    markdown = MarkdownIt('commonmark', {'html': False, 'breaks': True}).enable('table')
    messages = []
    for raw in selected:
        text = publish_links(raw['text'])
        validate_public_text(text)
        messages.append({**raw, 'id': 'message-%03d' % (len(messages) + 1),
                         'text': text, 'html': markdown.render(text), 'attachments': []})
    assert messages and messages[-1]['role'] == 'user'
    scope = ('User messages, assistant replies and progress updates from September 9, 2026, through the request '
             'to export this session. Tool logs, internal reasoning, runtime instructions and automatic browser '
             'context are omitted. The local PDF citation is rewritten to its public download. Messages are '
             'preserved as written, including provisional findings and cost estimates superseded later in the conversation.')
    document = {'title': args.title, 'threadId': args.thread_id,
                'exportedAt': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'startedAt': messages[0]['timestamp'], 'endedAt': messages[-1]['timestamp'],
                'messageCount': len(messages), 'scopeNote': scope, 'messages': messages}
    lines = ['# ' + document['title'], '', scope, '']
    for m in messages:
        label = 'User' if m['role'] == 'user' else 'Assistant'
        if m['phase'] == 'commentary':
            label += ' · progress update'
        lines.extend(['## ' + label + ' · ' + m['timestamp'], '', m['text'], ''])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'transcript.json').write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'transcript.md').write_text('\n'.join(lines))
    manifest = {'threadId': args.thread_id, 'cutoffRequest': args.cutoff_text,
                'messages': len(messages), 'users': sum(m['role'] == 'user' for m in messages),
                'assistant': sum(m['role'] == 'assistant' for m in messages),
                'progressUpdates': sum(m['phase'] == 'commentary' for m in messages),
                'startedAt': document['startedAt'], 'endedAt': document['endedAt'],
                'selection': 'event_msg/item_completed; UserMessage plus AgentMessage commentary/final_answer/final only; unique item IDs',
                'excluded': ['tool calls and results', 'reasoning', 'system/developer instructions', 'runtime envelopes', 'events after export request'],
                'adaptedFrom': 'animated-groups-fable/tools/export_session_transcript.py and docs/transcripts/gray-scott-session/',
                'rawRolloutPublished': False,
                'files': {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ['transcript.json','transcript.md']}}
    (args.output / 'export-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:manifest[k] for k in ['messages','users','assistant','progressUpdates','startedAt','endedAt']}))


if __name__ == '__main__':
    main()
