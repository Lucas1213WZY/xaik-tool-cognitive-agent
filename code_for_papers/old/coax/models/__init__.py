from .MLP.model import MLPEngine
from .xgboost.model import XGBoostEngine

def get_model(model_name, **kwargs):
	if model_name == 'mlp':
		return MLPEngine(**kwargs)
	elif model_name == 'xgboost':
		return XGBoostEngine(**kwargs)
	else:
		raise ValueError('Model not found')