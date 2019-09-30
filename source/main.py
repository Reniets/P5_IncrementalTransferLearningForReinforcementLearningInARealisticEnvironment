from source.runner import Runner
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


# Make runner object
runner = Runner()

# Settings of model can be changed in settings.py file
runner.train()  # Train a model
# runner.evaluate()  # Evaluate model

