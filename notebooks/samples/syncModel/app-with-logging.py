import json
from joblib import load
import pandas as pd
from neomaril_codex.logging import Logger


def score(data, base_path):
    logger = Logger('Sync')

    logger.debug("USING LOGGER")

    with open(base_path+"/model.pkl", 'rb') as f:
        model = load(f)

    df = pd.DataFrame(data=json.loads(data), index=[0])
    
    return logger.callback({"score": 1000 * (1-float(model.predict_proba(df)[0,1]))})
