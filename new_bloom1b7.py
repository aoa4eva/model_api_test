from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain
import os 

class GenModel():

    cache_dir = os.environ.get("HF_HOME")
    def __init__(self,model_id):    
#model_id = "bigscience/bloom-1b7"
        tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir=cache_dir)
        pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    self.hf = HuggingFacePipeline(pipeline=pipe)

    def run_prompt(question):
        template = """Question: {question}

        Answer: Let's think step by step."""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=hf)
        print(llm_chain.run(question))

question = "What is the capital of Indonesia?"
setup = GenModel("bigscience/boom-1b7")
setup.run_prompt(question)
