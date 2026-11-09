"""Allowlisted transport metadata. Never copy exception text or request data."""
import errno
import socket
import ssl
import urllib.error

STAGES={'fetch','response','stream'}
CATEGORIES={'dns','connect','tls','timeout','abort','http','validation','ok','unknown'}
TLS_REASON_CODES=frozenset({'UNEXPECTED_EOF_WHILE_READING','WRONG_VERSION_NUMBER',
    'TLSV1_ALERT_PROTOCOL_VERSION','TLSV1_ALERT_INTERNAL_ERROR'})
CODES={'EAI_AGAIN','EAI_NONAME','ECONNREFUSED','ECONNRESET','ECONNABORTED',
       'ENETUNREACH','EHOSTUNREACH','ETIMEDOUT','EPIPE','EINTR',
       'CERTIFICATE_VERIFY_FAILED','SSL_ERROR'} | TLS_REASON_CODES
FIELDS={'stage','category','error_code','http_status'}


def valid_diagnostic(value):
    return (isinstance(value,dict) and set(value)==FIELDS
        and value['stage'] in STAGES and value['category'] in CATEGORIES
        and (value['error_code'] is None or value['error_code'] in CODES)
        and (value['http_status'] is None or type(value['http_status']) is int
             and 100<=value['http_status']<=599))


def transport_diagnostic(error=None, http_status=None, *, validating=False):
    attached=getattr(error,'diagnostic',None)
    if valid_diagnostic(attached):return dict(attached)
    if isinstance(error,urllib.error.HTTPError):http_status=error.code
    if type(http_status) is not int or not 100<=http_status<=599:http_status=None
    result={'stage':'response' if http_status is not None or validating or error is None else 'fetch',
            'category':'validation' if validating else 'ok' if error is None else 'unknown',
            'error_code':None,'http_status':http_status}
    if http_status is not None and http_status>=400:
        result['category']='http';return result
    if validating:return result
    current=error;seen=set()
    for _ in range(5):
        if current is None or id(current) in seen:break
        seen.add(id(current))
        if isinstance(current,socket.gaierror):
            result.update(category='dns',error_code='EAI_AGAIN' if current.errno==socket.EAI_AGAIN else 'EAI_NONAME' if current.errno==socket.EAI_NONAME else None);break
        if isinstance(current,ssl.SSLCertVerificationError):
            result.update(category='tls',error_code='CERTIFICATE_VERIFY_FAILED');break
        if isinstance(current,ssl.SSLError):
            reason=getattr(current,'reason',None)
            code=reason if isinstance(reason,str) and reason in TLS_REASON_CODES else 'SSL_ERROR'
            result.update(category='tls',error_code=code);break
        if isinstance(current,TimeoutError):
            result.update(category='timeout',error_code='ETIMEDOUT');break
        number=getattr(current,'errno',None) or getattr(current,'winerror',None)
        numbers={errno.ECONNREFUSED:'ECONNREFUSED',errno.ECONNRESET:'ECONNRESET',
            errno.ECONNABORTED:'ECONNABORTED',errno.ENETUNREACH:'ENETUNREACH',
            errno.EHOSTUNREACH:'EHOSTUNREACH',errno.ETIMEDOUT:'ETIMEDOUT',
            errno.EPIPE:'EPIPE',errno.EINTR:'EINTR',10061:'ECONNREFUSED',
            10054:'ECONNRESET',10053:'ECONNABORTED',10060:'ETIMEDOUT',10051:'ENETUNREACH',10065:'EHOSTUNREACH'}
        if type(number) is int and number in numbers:
            code=numbers[number];result.update(error_code=code,category='timeout' if code=='ETIMEDOUT' else 'abort' if code=='EINTR' else 'connect');break
        reason=getattr(current,'reason',None)
        current=reason if isinstance(reason,BaseException) else getattr(current,'__cause__',None)
    return result


def default_health():
    return {'failure_limit':3,'failure_streak':0,'open':False,'reason':None,'last_diagnostic':None}


def valid_health(value):
    return (isinstance(value,dict) and set(value)==set(default_health())
        and type(value['failure_limit']) is int and value['failure_limit']==3
        and type(value['failure_streak']) is int and value['failure_streak']>=0
        and type(value['open']) is bool
        and value['reason'] in (None,'transport_unavailable','provider_rejected')
        and ((value['open'] and value['reason'] is not None) or (not value['open'] and value['reason'] is None))
        and (value['last_diagnostic'] is None or valid_diagnostic(value['last_diagnostic'])))


def update_health(value, diagnostic, error_code):
    result=dict(value)
    if result['open']:return result  # Late successes cannot close a tripped circuit.
    status=diagnostic['http_status']
    permanent=status in (400,401,402,403)
    failure=(error_code in ('TRANSPORT_ERROR','TRANSPORT_TIMEOUT')
             or status==429 or status is not None and status>=500)
    if permanent or failure:
        result.update(failure_streak=result['failure_streak']+1,last_diagnostic=diagnostic)
        if permanent or result['failure_streak']>=result['failure_limit']:
            result.update(open=True,reason='provider_rejected' if permanent else 'transport_unavailable')
    elif error_code is None or diagnostic['category']=='validation':
        result.update(failure_streak=0,last_diagnostic=diagnostic)
    return result
