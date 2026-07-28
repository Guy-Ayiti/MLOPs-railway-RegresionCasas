import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends
from io import StringIO
import pandas as pd
from joblib import load

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session

from datetime import datetime
import pytz
import os
#from dotenv import load_dotenv, find_dotenv

import json
from google.oauth2 import service_account
from google.cloud import bigquery


app = FastAPI()

#=====================================================================================#
# creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
# if creds_json:
#     creds_dict = json.loads(creds_json)
#     credentials = service_account.Credentials.from_service_account_info(creds_dict)
#     bq_client = bigquery.Client(credentials=credentials)
# else:
#     bq_client = bigquery.Client() 

#creds_dict = json.loads('Datapath-Mlops-GoogleCloud.json')
#credentials = service_account.Credentials.from_service_account_info(creds_dict)

#load_dotenv(find_dotenv())

gcp_secrets = json.loads(os.getenv("GCP_CREDENTIALS"))
##clientBQ = bigquery.Client.from_service_account_json(gcp_secrets)
clientBQ = bigquery.Client.from_service_account_info(gcp_secrets)
tableBQ = os.environ["BIGQUERY_TABLE"]


# Configurar la base de datos
SQLALCHEMY_DATABASE_URL = os.environ["SQLALCHEMY_DATABASE_URL"]

engine = create_engine(SQLALCHEMY_DATABASE_URL)
metadata = MetaData()

# Configurar la sesión de SQLAlchemy
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#=====================================================================================#


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"message": "Hello"}

@app.get("/health")
def health_check(db=Depends(get_db)):
    return {"status": "healthy",
            "message": "Se conectó de manera exitosa a la Base de Datos"}

@app.post("/predict")
async def predict_banknote(db: Session=Depends(get_db)):
    classifier = load("linear_regression.joblib")
    
    features_df = pd.read_csv('selected_features.csv')
    features = features_df['0'].to_list()

    query = f"SELECT * FROM `{tableBQ}`"    
    df = clientBQ.query(query).to_dataframe()
    df = df[features]

    predictions = classifier.predict(df)

    lima_tz = pytz.timezone('America/Lima')
    now = datetime.now(lima_tz)

    predictions_df = pd.DataFrame({
        #'file_name': file.filename,
        'prediction': predictions, 
        'created_at': now })

    #predictions_df.to_sql('predictions', con=engine, if_exists='replace', index=False)
    Datos = pd.concat([df,predictions_df], axis=1)
    Datos.to_sql('Predictions', con=engine, if_exists='replace', index=True)
    
    return {
        "predictions": Datos.to_html() #predictions.tolist()
    }
