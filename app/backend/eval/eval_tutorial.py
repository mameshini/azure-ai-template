import os
from braintrust import Eval
from autoevals import Factuality
from load_azd_env import load_azd_env
import logging
logger = logging.getLogger("eval")
load_azd_env()

token = os.getenv('BRAINTRUST_API_KEY')
logger.info(f"Braintrust key: {token}")

Eval(
  "Dev",
  data=lambda: [
      {
          "input": "Foo",
          "expected": "Hi Foo",
      },
      {
          "input": "Bar",
          "expected": "Hello Bar",
      },
  ],  
  task=lambda input: "Hi " + input,  
  scores=[Factuality],
)