import pandas as pd
import shap
from joblib import load

def parse(input_path, output_path):
    
    df_input = pd.read_csv(input_path)
    df_output = pd.read_csv(output_path)
    
    return (df_input, df_output)


def get_shap(data, model_path):

    model = load(model_path+"/model.pkl")

    explainer = shap.Explainer(model[-1])
    shap_values = explainer.shap_values(model[:-1].transform(data))

    return pd.DataFrame(data=shap_values, columns=data.columns)
