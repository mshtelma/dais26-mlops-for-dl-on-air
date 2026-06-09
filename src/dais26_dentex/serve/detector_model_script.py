"""MLflow *models-from-code* entrypoint for the detector pyfunc.

Logged via ``mlflow.pyfunc.log_model(python_model=<this file>)`` so MLflow
stores this script as the model definition instead of a cloudpickle of a
``DetectorPyfunc()`` instance.

Why this exists
---------------
Pickling the instance at log time (in the training process) captured a
reference to the HuggingFace *dynamic* backbone class, which lives in the
runtime-generated ``transformers_modules.*`` package created by
``trust_remote_code=True`` (nvidia/C-RADIOv4-SO400M). The Model Serving
container has no ``transformers_modules`` package, so unpickling
``python_model.pkl`` failed with::

    ModuleNotFoundError: No module named 'transformers_modules'

Models-from-code sidesteps pickling entirely: at load time MLflow executes
this script, which constructs a fresh ``DetectorPyfunc`` and registers it via
``set_model``. The backbone is then materialized inside ``load_context`` from
the bundled offline HF cache — never serialized.
"""

from mlflow.models import set_model

from dais26_dentex.serve.detector_pyfunc import DetectorPyfunc

set_model(DetectorPyfunc())
