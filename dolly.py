from langchain import HuggingFacePipeline
import logging 

logger = logging.getLogger(__name__)
logger.info("Loading model")
llm = HuggingFacePipeline.from_model_id(
    model_id="databricks/dolly-v2-3b",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)
logger.info("Prompting model")

llm("What is the capital of Indonesia?")
