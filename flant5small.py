from langchain import HuggingFacePipeline
import logging 

logger = logging.getLogger(__name__)
logger.info("Loading model")
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-small",
    task="text2text-generation",
    model_kwargs={"temperature": 0.02, "max_length": 64},
)
logger.info("Prompting model")

from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: This is the answer to the question:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

question = "What is the capital of the US?"

llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))
