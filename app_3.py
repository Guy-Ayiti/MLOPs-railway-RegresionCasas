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

app = FastAPI()



# Configurar la base de datos
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:GhNpqooGAmJAjfRorPMVjxGIzakEKcJf@altaria.proxy.rlwy.net:59533/railway"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
metadata = MetaData()

# Configurar la sesión de SQLAlchemy
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    return {"status": "healthy"}

@app.post("/predict")
async def predict_banknote(file: UploadFile = File(...), db: Session=Depends(get_db)):
    classifier = load("linear_regression.joblib")
    
    features_df = pd.read_csv('selected_features.csv')
    features = features_df['0'].to_list()

    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    df = df[features]

    predictions = classifier.predict(df)

    lima_tz = pytz.timezone('America/Lima')
    now = datetime.now(lima_tz)

    predictions_df = pd.DataFrame({
        'file_name': file.filename,
        'prediction': predictions, 
        'created_at': now })

    predictions_df.to_sql('predictions', con=engine, if_exists='append', index=False)
    
    return {
        "predictions": predictions.tolist()
    }
