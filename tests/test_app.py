from fastapi.testclient import TestClient
from app import app #this will run app.py top to bottom (importing TF)
import predict
import base64


def make_fake_predict(vcdr): #use a factory to be able to set the return value as wanted
    def _fake_predict(*args): #accepts all the args but just ignores them
        return {"vertical_cdr": vcdr, "overlay_png":b"...somebytes..."}
    return _fake_predict

def fake_load_models(*args): #accepts all the args but just ignores them
    return (object(),object())


def test_health_check(): 
    client = TestClient(app) #plain mode (still routes requests fine, lifespan never runs though so it can't test anything that needs app.state)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status":"ok"}


def test_predict_endpoint_returns_200_and_cdr(monkeypatch):
 
    # for temporarily replacing an attribute during a test but automatically restoring the original when the test ends
    # we don't want to waste time on actual inference so we use these replacements
    monkeypatch.setattr(predict, "load_models",fake_load_models)
    monkeypatch.setattr(predict,"predict",make_fake_predict(0.5))

    with TestClient(app) as client: #you need this context manager so that you run lifespan's startup and so app.state gets populated
        response = client.post("/predict",files={"file": ("test.png",b"somebytes","image/png")})
        assert response.status_code == 200
        assert response.json()["vertical_cdr"] == 0.5
        assert base64.b64decode(response.json()["overlay_png"]) == b"...somebytes..."      # truthy = non-empty string
    
def test_predict_endpoint_invalid_returns_422(monkeypatch):
    monkeypatch.setattr(predict, "load_models",fake_load_models)
    monkeypatch.setattr(predict,"predict",make_fake_predict(None))

    with TestClient(app) as client: #you need this context manager so that you run lifespan's startup and so app.state gets populated
        
        response = client.post("/predict",files={"file": ("test.png",b"somebytes","image/png")})
        assert response.status_code == 422
        assert "disc/cup" in response.json()["detail"]

