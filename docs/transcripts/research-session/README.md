# Research-session transcript

[Read the HTML transcript](https://yaroslavvb.github.io/gradient-dissent/transcripts/research-session/).

This fixed snapshot contains 111 completed user-visible messages: 5 user messages and 106 assistant messages, including 101 progress updates. It starts with the paper-review request and ends with the request to publish this session. Earlier provisional findings remain as written. Tool activity, internal reasoning, system/developer instructions and automatic context are excluded. No raw session log is published.

The exporter and searchable viewer were adapted from the local `animated-groups-fable` implementation (`tools/export_session_transcript.py` and `docs/transcripts/gray-scott-session/`). The CSS is reused; the HTML title/navigation and JavaScript data-version key are updated for this project. Old project-specific video copying was removed. The PDF file citation now points to the published paper.

## Rebuild from the private local rollout

From the repository root:

```sh
python3 -m venv /tmp/gradient-dissent-export-venv
/tmp/gradient-dissent-export-venv/bin/pip install -r requirements-session-export.txt
/tmp/gradient-dissent-export-venv/bin/python scripts/export_session_transcript.py \
  /path/to/this-session.jsonl docs/transcripts/research-session
/tmp/gradient-dissent-export-venv/bin/python -m unittest discover \
  -s scripts -p 'test_export_session_transcript.py'
npm run check:transcript
```

The publication request is an exact cutoff. The exporter fails if that request is absent; it will not silently include future turns. It accepts only completed `UserMessage` and `AgentMessage` items, with assistant phases explicitly restricted to progress/final replies. It fails on unsupported attachments, unresolved local links, remaining runtime envelopes or recognized credential patterns. Markdown rendering disables raw HTML and unsafe URL schemes. The public JSON and Markdown contain only that filtered selection.

`export-manifest.json` records the cutoff, counts and output digests. GitHub Pages serves the committed directory from `main` under `/docs`; no backend or CDN is required. Search, progress visibility, permalinks, the PDF link and internal links are checked programmatically.
