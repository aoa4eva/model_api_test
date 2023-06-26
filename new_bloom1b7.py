from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain

model_id = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir="/home/aoabenlizar/efs")
model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir="/home/aoabenlizar/efs")
pipe = pipeline(
"text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
)
hf = HuggingFacePipeline(pipeline=pipe)



template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=hf)

question = "What is the capital of Indonesia?"

print(llm_chain.run(question))
