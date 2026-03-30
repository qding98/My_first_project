import inspect, sys, pathlib
sys.path.append(str(pathlib.Path('safety-eval').resolve()))
from src.classifier_models import base
print(inspect.getsource(base.ResponseRefusal))
print(inspect.getsource(base.ResponseHarmfulness))
