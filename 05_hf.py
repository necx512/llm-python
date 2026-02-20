from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# hub_llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

# prompt = PromptTemplate(
#     input_variables=["question"],
#     template="Translate English to SQL: {question}"    
# )

# hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
# print(hub_chain.run("What is the average age of the respondents using a mobile device?"))

# Note: Original code used remote HuggingFace Inference API, which is incompatible with
# current package versions. Switched to local execution using HuggingFacePipeline.
# See commented-out code below for original remote API approach (non-functional).

# ============================================================================
# ORIGINAL APPROACH (REMOTE API) - COMMENTED OUT DUE TO COMPATIBILITY ISSUES
# ============================================================================
# from langchain_community.llms import HuggingFaceHub
#
# # Original remote API approach using deprecated HuggingFaceHub
# # Issues:
# # - HuggingFaceHub is deprecated and broken with huggingface_hub>=1.0
# # - HuggingFaceEndpoint (recommended replacement) has compatibility issues
# # - Requires HUGGINGFACEHUB_API_TOKEN in .env
#
# hub_llm = HuggingFaceHub(
#     repo_id='gpt2',
#     model_kwargs={'temperature': 0.7, 'max_length': 100}
# )
# ============================================================================

# ============================================================================
# WORKING APPROACH (LOCAL EXECUTION)
# ============================================================================
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

# Local execution using HuggingFacePipeline (downloads model on first run ~500MB)
model_id = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100,  # Total length (not new tokens)
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id  # Suppress pad_token warning
)
hub_llm = HuggingFacePipeline(pipeline=pipe)
# ============================================================================

prompt = PromptTemplate(
    input_variables=["profession"],
    template="You had one job 😡! You're the {profession} and you didn't have to be sarcastic"
)

# LCEL (pipe syntax) for LangChain 1.x compatibility
hub_chain = prompt | hub_llm
print(hub_chain.invoke({"profession": "customer service agent"}))
print(hub_chain.invoke({"profession": "politician"}))
print(hub_chain.invoke({"profession": "Fintech CEO"}))
print(hub_chain.invoke({"profession": "insurance agent"}))