from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain
import os 

class GenModel():

    cache_dir = os.environ.get("HF_HOME")
    def __init__(self,model_id):    
        tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=self.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir=self.cache_dir)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
        self.hf = HuggingFacePipeline(pipeline=pipe)

    def run_prompt(self,question):
        template = """Answer this directly: {question}"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.hf)
        answer = llm_chain.run(question)
        return answer
