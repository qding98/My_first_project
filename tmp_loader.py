from inspect import getsource
import importlib
m = importlib.import_module("src.classifier_models.loader")
print(getsource(m.load_classifier_model))
