import time
import uvicorn

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

@app.middleware('http')
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers['X-Process-Time'] = str(process_time)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# baseUrl = "/api/v1"
# app.include_router(questions.router, tags=["Questions"], prefix=baseUrl)
# app.include_router(choose.router, tags=["Choose"], prefix=baseUrl)

class PriceHouseSerializer(BaseModel):
    District: str
    Ward: str
    Month: int
    Year: int
    Area: int
    class Config:
        orm_mode = True
        arbitrary_types_allowed = True
        allow_population_by_field_name = True


@app.post("/")
def read_root(data: PriceHouseSerializer):
    X_test = jsonable_encoder(data)
    X_test["District"] = X_test['District'].replace('Quận ', '').replace(' ', '').strip()
    X_test['Ward'] = X_test['Ward'].replace('Phường ', '').replace(' ', '').strip()
    X_test['Month'] = X_test['Month']
    X_test['Year'] = X_test['Year']

    X_test = pd.DataFrame.from_dict([X_test])

    dummy_district = pd.get_dummies(X_test["District"], prefix="District")
    dummy_ward = pd.get_dummies(X_test["Ward"], prefix="Ward")
    dummy_month = pd.get_dummies(X_test["Month"], prefix="Month")
    dummy_year = pd.get_dummies(X_test["Year"], prefix="Year")

    X_test = pd.concat([X_test, dummy_month, dummy_year, dummy_district, dummy_ward], axis=1)
    X_test = X_test.drop(['District', 'Ward', 'Month', 'Year'], axis=1)
    df = pd.read_csv("data.csv")
    for item in X_test:
        df[item] = X_test[item]
    load_model = joblib.load("my_random_forest.joblib")
    print(df)
    RF_predictions = load_model.predict(df)
    return {
        "success": True,
        "Estimated": round(float(RF_predictions ), 2),
        "message": f"Estimated house price is {round(float(RF_predictions ), 2)} million VND/m2"
    }
if __name__ == "__main__":
    uvicorn.run(app="main:app", host="127.0.0.1", port=5000, reload=True, workers=1)
