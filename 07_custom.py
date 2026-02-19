"""
Optional: Change where pretrained models from huggingface will be downloaded (cached) to:
export TRANSFORMERS_CACHE=/whatever/path/you/want
"""

# import os
# os.environ["TRANSFORMERS_CACHE"] = "/media/samuel/UDISK1/transformers_cache"
import os
import time

import torch
from dotenv import load_dotenv
from typing import ClassVar, Any, Iterator
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core import (
    SummaryIndex,
    PromptHelper,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from transformers import pipeline

# load_dotenv()
os.environ["OPENAI_API_KEY"] = "random"


def timeit():
    """
    a utility decoration to time running time
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            args = [str(arg) for arg in args]

            print(f"[{(end - start):.8f} seconds]: f({args}) -> {result}")
            return result

        return wrapper

    return decorator


prompt_helper = PromptHelper(
    # maximum input size
    context_window=2048,
    # number of output tokens
    num_output=256,
    # the maximum overlap between chunks.
    chunk_overlap_ratio=0.1,
)


class LocalOPT(CustomLLM):
    # model_name = "facebook/opt-iml-max-30b" (this is a 60gb model)
    model_name: ClassVar[str] = "facebook/opt-iml-1.3b"  # ~2.63gb model
    # https://huggingface.co/docs/transformers/main_classes/pipelines
    _pipeline: ClassVar[Any] = pipeline(
        "text-generation",
        model=model_name,
        device="cpu",
        dtype=torch.float32,
    )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self._pipeline(prompt, max_new_tokens=256)[0]["generated_text"]
        # only return newly generated tokens
        return CompletionResponse(text=response[len(prompt):])

    def stream_complete(self, prompt: str, **kwargs) -> Iterator[CompletionResponse]:
        raise NotImplementedError


@timeit()
def create_index():
    print("Creating index")
    Settings.llm = LocalOPT()
    Settings.prompt_helper = prompt_helper

    docs = SimpleDirectoryReader("news").load_data()
    index = SummaryIndex.from_documents(docs)
    print("Done creating index", index)
    return index


@timeit()
def execute_query():
    query_engine = index.as_query_engine()
    response = query_engine.query(
        "Who does Indonesia export its coal to in 2023?",
    )
    return response


if __name__ == "__main__":
    """
    Check if a local cache of the model exists,
    if not, it will download the model from huggingface
    """
    if not os.path.exists("7_custom_opt"):
        print("No local cache of model found, downloading from huggingface")
        index = create_index()
        index.storage_context.persist(persist_dir="./7_custom_opt")
    else:
        print("Loading local cache of model")
        Settings.llm = LocalOPT()
        Settings.prompt_helper = prompt_helper
        storage_context = StorageContext.from_defaults(persist_dir="./7_custom_opt")
        index = load_index_from_storage(storage_context)

    response = execute_query()
    print(response)
    print(response.source_nodes)
