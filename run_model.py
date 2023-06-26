from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain
import os 
from genmodel import GenModel 

question = "What is the capital of Indonesia?"
setup = GenModel("bigscience/bloom-1b7")
ans = setup.run_prompt(question)
print(ans)
