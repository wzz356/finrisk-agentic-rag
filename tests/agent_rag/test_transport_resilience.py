import errno
import json
import socket
import ssl
import threading
import urllib.error

import pytest

from agent_rag.rerank_backend import DeepSeekReranker, RerankTaskControl


def good_reply(body, wrong_quote=False):
    packet=json.loads(body['messages'][1]['content'])
    opinion={'pair_id':packet['pair_id'],'score':85,'level':'specific_direct',
        'source_quote':'bad quote' if wrong_quote else packet['unit_text'],
        'reason':'同机制','limitation':'条件仍须核对','human_approved':False}
    return {'model':'deepseek-flash','choices':[{'finish_reason':'stop','message':{'content':json.dumps(opinion)}}],
        'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}}


def one(number=0):return {'unit_id':f'U{number}','file_id':f'F{number}','text':'处罚事实与适用条件。'}


@pytest.mark.parametrize('error,category,code',[
    (urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN,'sk-secret-fake-value')),'dns','EAI_AGAIN'),
    (urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED,'Authorization: Bearer fake-secret')),'connect','ECONNREFUSED'),
    (urllib.error.URLError(ssl.SSLCertVerificationError('private-body')),'tls','CERTIFICATE_VERIFY_FAILED'),
    (TimeoutError('request body secret'),'timeout','ETIMEDOUT'),
    (urllib.error.URLError(socket.gaierror(socket.EAI_FAIL,'private-dns-detail')),'dns',None),
])
def test_attempt_diagnostic_is_classified_and_never_copies_exception_text(tmp_path,error,category,code):
    def fail(*_):raise error
    control=RerankTaskControl(tmp_path/'ledger.json',1,20)
    result=DeepSeekReranker('规则',fail).score_query('query',[one()],control)
    attempts=json.loads(control.attempt_audit_path.read_text(encoding='utf-8'))['attempts']
    diagnostic=attempts[0].get('diagnostic')
    assert diagnostic is not None, 'lower-level diagnostic is lost'
    assert diagnostic['category']==category and diagnostic['error_code']==code
    assert not any(secret in json.dumps(result)+json.dumps(attempts) for secret in ['sk-secret','Bearer fake','private-body','body secret'])
    assert result['meta']['reserved_cost_micro_cny']>0


def test_task_breaker_stops_after_three_failures_and_survives_restart(tmp_path):
    calls=[]
    def fail(*_):calls.append(1);raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET,'failed'))
    control=RerankTaskControl(tmp_path/'ledger.json',1,100)
    reranker=DeepSeekReranker('规则',fail)
    reranker.score_query('q',[one()],control)
    result=reranker.score_query('q',[one(1)],control)
    assert len(calls)==3, 'must stop the second pair retry when circuit opens'
    assert result['meta']['transport_circuit_open'] is True
    assert result['meta']['transport_failure_streak']==3
    assert result['meta']['stop_reason']=='transport_unavailable'
    retained=result['meta']['reserved_cost_micro_cny']
    restarted=RerankTaskControl(tmp_path/'ledger.json',1,100)
    assert not restarted.check()
    reranker.score_query('q',[one(2)],restarted)
    assert len(calls)==3 and restarted.meta()['reserved_cost_micro_cny']==retained
    before=restarted.meta()['transport_diagnostic'];restarted.cancel()
    assert restarted.meta()['transport_circuit_open'] and restarted.meta()['transport_diagnostic']==before


def test_http_success_and_quote_validation_reset_network_streak(tmp_path):
    count=0
    def send(body,_):
        nonlocal count
        count+=1
        if count in (1,2,5,6):raise TimeoutError('timeout')
        return good_reply(body,wrong_quote=count==3)
    control=RerankTaskControl(tmp_path/'ledger.json',1,100)
    reranker=DeepSeekReranker('规则',send)
    reranker.score_query('q',[one()],control)
    assert control.meta().get('transport_failure_streak')==2
    assert reranker.score_query('q',[one(1)],control)['complete']
    assert control.meta()['transport_failure_streak']==0
    reranker.score_query('q',[one(2)],control)
    assert control.meta()['transport_failure_streak']==2 and not control.meta()['transport_circuit_open']


def test_permanent_provider_rejection_stops_other_pairs_immediately(tmp_path):
    calls=[]
    def fail(*_):calls.append(1);raise urllib.error.HTTPError('https://api.deepseek.com',402,'hidden',{},None)
    control=RerankTaskControl(tmp_path/'ledger.json',1,100)
    reranker=DeepSeekReranker('规则',fail)
    result=reranker.score_query('q',[one()],control)
    reranker.score_query('q',[one(1)],control)
    assert len(calls)==1
    assert result['meta']['transport_circuit_open']
    assert result['meta']['transport_diagnostic']['http_status']==402


def test_parallel_failures_cannot_dispatch_the_entire_candidate_pool(tmp_path):
    calls=[];lock=threading.Lock();release=threading.Event()
    def fail(*_):
        with lock:
            calls.append(1)
            if len(calls)>=8:release.set()
        release.wait(1)
        raise TimeoutError('offline injected failure')
    control=RerankTaskControl(tmp_path/'ledger.json',1,240)
    result=DeepSeekReranker('规则',fail).score_query('q',[one(i) for i in range(100)],control)
    assert len(calls)<=10, 'three failures open circuit; at most seven other calls already in flight'
    assert result['meta']['transport_circuit_open']
    assert result['meta']['external_retrieval_requests']==len(calls)
    assert result['meta']['reserved_cost_micro_cny']>0


@pytest.mark.parametrize('outcome,category,code',[
    ('timeout','timeout','TRANSPORT_TIMEOUT'),('invalid_json','validation','INVALID_JSON')])
def test_response_body_timeout_is_not_a_semantic_validation_error(tmp_path,monkeypatch,outcome,category,code):
    from agent_rag import rerank_backend
    class Reply:
        status=200
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def read(self):
            if outcome=='timeout':raise TimeoutError('private-response-body')
            return b'invalid-json'
    class Opener:
        def open(self,*_,**__):return Reply()
    monkeypatch.setattr(rerank_backend.urllib.request,'build_opener',lambda *_:Opener())
    control=RerankTaskControl(tmp_path/'ledger.json',1,20)
    result=DeepSeekReranker('rules',key_loader=lambda:'sk-fake-01234567890123456789').score_query('q',[one()],control)
    rows=json.loads(control.attempt_audit_path.read_text(encoding='utf-8'))['attempts']
    assert all(row['diagnostic']['category']==category for row in rows)
    assert all(row['diagnostic']['stage']=='response' and row['diagnostic']['http_status']==200 for row in rows)
    assert all(row['error_code']==code for row in rows)
    assert result['meta']['transport_failure_streak']==(2 if outcome=='timeout' else 0)
    assert result['meta']['reserved_cost_micro_cny']>0
    assert 'private-response-body' not in json.dumps(rows)


@pytest.mark.parametrize('reason', [
    'UNEXPECTED_EOF_WHILE_READING', 'WRONG_VERSION_NUMBER',
    'TLSV1_ALERT_PROTOCOL_VERSION', 'TLSV1_ALERT_INTERNAL_ERROR',
])
def test_tls_structured_reason_is_allowlisted_through_url_wrapper(tmp_path, reason):
    error = ssl.SSLError('private URL https://private/?key=sk-test-secret')
    error.reason = reason
    def fail(*_): raise urllib.error.URLError(error)
    control = RerankTaskControl(tmp_path/'ledger.json', 1, 20)
    result = DeepSeekReranker('rules', fail).score_query('query', [one()], control)
    rows = json.loads(control.attempt_audit_path.read_text(encoding='utf-8'))['attempts']
    assert rows and all(row['diagnostic']['error_code'] == reason for row in rows)
    assert all(row['diagnostic']['category'] == 'tls' for row in rows)
    assert result['meta']['reserved_cost_micro_cny'] > 0
    assert 'sk-test-secret' not in json.dumps(rows)
    assert 'private URL' not in json.dumps(rows)


@pytest.mark.parametrize('reason', [None, 'unknown', 'UNEXPECTED_EOF_WHILE_READING private-key', 42])
def test_tls_unrecognized_structured_reason_stays_generic(reason):
    from agent_rag.transport_diagnostics import transport_diagnostic, valid_diagnostic
    error = ssl.SSLError('UNEXPECTED_EOF_WHILE_READING do not parse this message')
    if reason is not None: error.reason = reason
    diagnostic = transport_diagnostic(urllib.error.URLError(error))
    assert diagnostic['error_code'] == 'SSL_ERROR'
    assert valid_diagnostic(diagnostic)
