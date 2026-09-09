"""Publication-boundary checks adapted from the animated-groups-fable exporter."""
import json
from pathlib import Path
import tempfile
import unittest
from markdown_it import MarkdownIt
from export_session_transcript import CUTOFF_REQUEST, visible_messages, publish_links, validate_public_text, PAPER_LOCAL, PAPER_PUBLIC

class TranscriptVisibilityTests(unittest.TestCase):
    def item(self, kind, identifier, text, phase=None):
        return {'type':'event_msg','timestamp':'2026-09-09T20:52:50Z','payload':{'type':'item_completed','item':{
            'type':kind,'id':identifier,'phase':phase,'content':[{'type':'Text','text':text}]}}}

    def read(self, events):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.jsonl';p.write_text('\n'.join(json.dumps(e) for e in events))
            return list(visible_messages(p))

    def test_only_completed_visible_items_and_stop_at_wrapped_request(self):
        private=[self.item(k,k,'PRIVATE') for k in ['Reasoning','CommandExecution','McpToolCall','FileChange','ContextCompaction']]
        private += [self.item('AgentMessage','analysis','PRIVATE','analysis'),
                    {'type':'response_item','payload':{'type':'message','role':'system','content':'PRIVATE'}}]
        cutoff='<in-app-browser-context>PRIVATE</in-app-browser-context>\n## My request:\n'+CUTOFF_REQUEST
        events=private+[self.item('UserMessage','u','hello'),self.item('AgentMessage','p','working','commentary'),
                        self.item('AgentMessage','a','done','final_answer'),self.item('AgentMessage','a','done','final_answer'),
                        self.item('UserMessage','cutoff',cutoff),self.item('AgentMessage','later','NOT IN SNAPSHOT','commentary')]
        result=self.read(events)
        self.assertEqual([m['text'] for m in result],['hello','working','done',CUTOFF_REQUEST])
        self.assertEqual([m['phase'] for m in result],['request','commentary','final','request'])
        self.assertNotIn('PRIVATE',json.dumps(result))

    def test_missing_boundary_fails(self):
        with self.assertRaisesRegex(ValueError,'publication request was not found'):
            self.read([self.item('UserMessage','u','hello')])

    def test_unknown_attachments_fail(self):
        x=self.item('UserMessage','u','hello');x['payload']['item']['content'].append({'type':'image','image_url':'file:///private.png'})
        with self.assertRaisesRegex(ValueError,'Unsupported visible attachment'):self.read([x])

    def test_file_citation_resolves_without_copying_private_files(self):
        raw='Paper: :codex-file-citation{path="'+str(PAPER_LOCAL)+'" purpose="output"}'
        result=publish_links(raw);self.assertIn(PAPER_PUBLIC,result);self.assertNotIn('/Users/',result)
        validate_public_text(result)
        with self.assertRaises(ValueError):publish_links('[secret](/Users/example/private.txt)')

    def test_runtime_and_credentials_fail(self):
        for text in ['<developer>PRIVATE</developer>','ghp_'+'A'*30,'-----BEGIN PRIVATE KEY-----','http://localhost:1234/test','/Users/example/private']:
            with self.assertRaises(ValueError):validate_public_text(text)

    def test_markdown_disables_active_html_and_javascript_links(self):
        md=MarkdownIt('commonmark',{'html':False,'breaks':True}).enable('table')
        rendered=md.render('<script>alert(1)</script>\n[click](javascript:alert(1))')
        self.assertNotIn('<script>',rendered);self.assertNotIn('href="javascript:',rendered)

if __name__=='__main__':unittest.main()
