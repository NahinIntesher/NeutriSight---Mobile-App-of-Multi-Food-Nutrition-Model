import io, os, tempfile
os.environ['NUTRISIGHT_DATA_DIR']=tempfile.mkdtemp(prefix='nutrisight-test-')
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app, pipeline, dietary_insights

def account(c,email):
    r=c.post('/api/auth/signup',json={'email':email,'password':'longpassword123','name':'Test User'})
    assert r.status_code==201,r.text
    return {'Authorization':'Bearer '+r.json()['token']}
def test_account_profile_private_history(monkeypatch):
    fake={'foods':[{'name':'egg','confidence':.8}],'nutrition':{'calories':200,'mass':100,'protein':12,'fat':10,'carbs':2},'warnings':[],'image_size':[10,10]}
    monkeypatch.setattr(pipeline,'analyze',lambda image:dict(fake))
    with TestClient(app) as c:
        a=account(c,'a@example.com');b=account(c,'b@example.com')
        assert c.get('/api/history').status_code==401
        assert c.put('/api/profile',headers=a,json={'name':'Alice','allergies':['egg']}).status_code==200
        out=io.BytesIO();Image.new('RGB',(10,10)).save(out,format='PNG')
        scan=c.post('/api/analyze',headers=a,files={'image':('meal.png',out.getvalue(),'image/png')})
        assert scan.status_code==200,scan.text
        r=scan.json();assert r['saved'] and r['alerts'][0]['level']=='warning'
        assert len(c.get('/api/history',headers=a).json())==1
        assert c.get('/api/history',headers=b).json()==[]
        assert c.delete('/api/history/'+r['id'],headers=b).status_code==404
        assert c.post('/api/analyze',files={'image':('bad.jpg',b'no','image/jpeg')}).status_code==422
        guest=c.post('/api/analyze',files={'image':('meal.png',out.getvalue(),'image/png')}).json()
        assert not guest['saved'] and guest['profile_required']
        assert c.delete('/api/me',headers=a).status_code==200
        assert c.get('/api/me',headers=a).status_code==401
        assert c.post('/api/auth/logout',headers=b).status_code==200
        assert c.get('/api/me',headers=b).status_code==401

def test_login_and_validation():
    with TestClient(app) as c:
        account(c,'login@example.com')
        assert c.post('/api/auth/login',json={'email':'login@example.com','password':'incorrect-password'}).status_code==401
        assert c.post('/api/auth/login',json={'email':'LOGIN@example.com','password':'longpassword123'}).status_code==200
        assert c.get('/health').status_code==200
        assert c.get('/api/me',headers={'Authorization':'Bearer invalid'}).status_code==401
        assert c.post('/api/auth/signup',json={'email':'invalid','password':'longpassword123','name':'Name'}).status_code==422

def test_matching_is_not_substring():
    assert dietary_insights([{'name':'eggplant'}],{'allergies':['egg']})[0]['level']=='info'
    assert dietary_insights([{'name':'milk'}],{'allergies':['dairy']})[0]['level']=='warning'

def test_password_rotation_dashboard_and_cors():
    with TestClient(app) as c:
        a=account(c,'rotation@example.com')
        assert c.get('/api/dashboard',headers=a).json()['total_scans']==0
        bad=c.put('/api/auth/password',headers=a,json={'current_password':'badpassword','new_password':'changedpassword123'})
        assert bad.status_code==400
        out=c.put('/api/auth/password',headers=a,json={'current_password':'longpassword123','new_password':'changedpassword123'})
        assert out.status_code==200
        assert c.get('/api/me',headers=a).status_code==401
        b={'Authorization':'Bearer '+out.json()['token']}
        assert c.get('/api/me',headers=b).status_code==200
        assert c.post('/api/auth/login',json={'email':'rotation@example.com','password':'longpassword123'}).status_code==401
        assert c.put('/api/profile',headers=b,json={'name':'  '}).status_code==422
        response=c.options('/api/auth/signup',headers={'Origin':'http://192.168.1.10:8081','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'Content-Type'})
        assert response.status_code==200
        assert response.headers['access-control-allow-origin']=='http://192.168.1.10:8081'

def test_profile_skip_and_raw_upload(monkeypatch):
    fake={'foods':[{'name':'rice','confidence':.9}],'nutrition':{'calories':300,'mass':200,'protein':8,'fat':4,'carbs':55},'warnings':[],'image_size':[10,10]}
    monkeypatch.setattr(pipeline,'analyze',lambda image:dict(fake))
    with TestClient(app) as c:
        auth=account(c,'raw@example.com')
        me=c.get('/api/me',headers=auth).json()
        assert me['profile']=={}
        skipped=c.post('/api/profile/skip',headers=auth,json={})
        assert skipped.status_code==200,skipped.text
        assert skipped.json()['profile']['onboarding_complete'] is True
        out=io.BytesIO();Image.new('RGB',(10,10)).save(out,format='JPEG')
        raw=c.post('/api/analyze/raw',headers={**auth,'Content-Type':'image/jpeg'},content=out.getvalue())
        assert raw.status_code==200,raw.text
        assert raw.json()['foods'][0]['name']=='rice'
        assert raw.json()['profile_required'] is False
