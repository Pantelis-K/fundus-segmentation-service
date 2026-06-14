from fastapi import FastAPI #to declare endpoints, while handling routing/validation automatically
from fastapi import UploadFile #custom class to help /docs generate an "Upload File" button automatically
from fastapi import HTTPException #for handling a "None" result from ellipse fitting logic
import predict as run_predict #import model and inference logic (with ellipse fitting)
from contextlib import asynccontextmanager #to create variables in shared state accessible across requests
from pydantic import BaseModel #to define complex return type of the route (helps with /docs and output validation)
import base64 #to covert png bytes into JSON safe string so it can be returned with the request

class PredictResponse(BaseModel):
    #custom return type, declared for automated documentation
    vertical_cdr: float | None
    overlay_png: str #base64-encoded PNG, as a string


@asynccontextmanager
async def lifespan(app: FastAPI):
    #startup code
    disc, cup = run_predict.load_models()
    app.state.disc_model = disc
    app.state.cup_model = cup
    yield
    #shutdown code (none needed)
    

app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile):
    image_bytes = await file.read()
    result = run_predict.predict(image_bytes,app.state.disc_model,app.state.cup_model)
    vcdr = result["vertical_cdr"]
    if vcdr is None:
        raise HTTPException(status_code=422, detail="Could not locate a disc/cup - ensure the uploaded file is an ROI-cropped fundus image.")
    png_bytes = result["overlay_png"]
    overlay = base64.b64encode(png_bytes).decode("ascii")
    return PredictResponse(vertical_cdr=vcdr, overlay_png=overlay)



