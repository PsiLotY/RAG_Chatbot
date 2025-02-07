llm_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
embedding_model = "sentence-transformers/distiluse-base-multilingual-cased-v1"

_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_llm_model_name():
    return llm_model

def get_tokenizer_name():
    return tokenizer

def get_embedding_model_name():
    return embedding_model

