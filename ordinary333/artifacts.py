"""Optional GitHub Release checkpoints. Tokens come only from the runtime secret store."""
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from urllib.parse import quote


def request(url,token,method='GET',body=None,content_type='application/json'):
    """Make a bounded authenticated request without printing credentials or response headers."""
    payload=json.dumps(body).encode() if isinstance(body,dict) else body
    req=Request(url,data=payload,method=method,headers={'Authorization':'Bearer '+token,
                'Accept':'application/vnd.github+json','Content-Type':content_type,'User-Agent':'ordinary333-research'})
    with urlopen(req,timeout=90) as response:return json.load(response)


def publish_checkpoint(path,run_id,step):
    """Publish one owned checkpoint asset; return its URL, or a nonfatal explicit failure."""
    token=os.environ.get('GITHUB_TOKEN','')
    repo=os.environ.get('RESULTS_REPOSITORY','cheldieva-l/ihes-13-methods')
    if not token:return dict(status='no_storage_token',url=None)
    base='https://api.github.com/repos/'+repo
    tag='ordinary333-'+run_id
    try:
        try:release=request(base+'/releases/tags/'+quote(tag),token)
        except HTTPError as error:
            if error.code!=404:raise
            release=request(base+'/releases',token,'POST',{'tag_name':tag,'name':run_id+' checkpoints','body':'Ordinary333 experimental weights; quality is measured separately.'})
        name=f'{run_id}-step{step}.pt'
        existing=next((a for a in release.get('assets',[]) if a['name']==name),None)
        if existing:return dict(status='published',url=existing['browser_download_url'])
        url=release['upload_url'].split('{')[0]+'?name='+quote(name)
        asset=request(url,token,'POST',Path(path).read_bytes(),'application/octet-stream')
        return dict(status='published',url=asset['browser_download_url'])
    except Exception as error:
        return dict(status='upload_failed',error_type=type(error).__name__,http_code=getattr(error,'code',None),url=None)
