import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import logger, current_memory
from chroma_functions import get_closest_document
from init import *

log = logger(__name__)
_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
tokenizer = None

def init_llm_model():
    log.info(f"Initializing model: {get_llm_model_name}")
    global model, tokenizer
    current_memory()
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_4bit_compute_dtype=torch.float16  # Use FP16 for compute instead of FP32
    )

    tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_name())
    model = AutoModelForCausalLM.from_pretrained(
        get_llm_model_name(),
        device_map="auto",
        quantization_config=quantization_config
    )
    log.info(f"Initialized model: {get_llm_model_name()}")
    current_memory()

def get_model():
    return model

def get_tokenizer():
    return tokenizer

def generate_text(model, tokenizer, query):
    prompt, source_url = create_prompt(query)
    input_ids = tokenizer(prompt, return_tensors="pt").to(_device)
    prompt_length = input_ids.input_ids.shape[1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    log.info(f"Generating an answer for the query: {query}")
    
    with torch.no_grad():
        output = model.generate(
            input_ids.input_ids,
            do_sample=True,
            temperature=1.0,
            top_p=0.7, # sampling rate for next word prediciton. randomly picks a word of a group of words which cumulative propability are top_p
            max_new_tokens=500,  # Reduce token count to save memory
            attention_mask=input_ids.attention_mask,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    log.info(f"Question: {query}")
    log.info(f"Answer: {generated_text}")
    return generated_text, source_url



def create_prompt(query):
    log.info(f"Adding user query to prompt object")
    prompt = [
        {"role": "system", "content": "Du bist Experte über die Hochschule der Medien in Stuttgart und hilfst Leuten bei spezifischen Fragen dazu aus."},
        {"role": "user", "content": query},
    ]
    results = get_closest_document(query)

    if results:
        retrieved_doc = results[0]  # Get the most relevant document
        document_text = retrieved_doc["document"]
        source_url = retrieved_doc["metadata"].get("url", "Quelle nicht verfügbar")

        prompt.append(
            {
                "role": "system",
                "content": f"Hier ist eine relevante Information aus einer Quelle der HdM: {document_text}",
            }
        )
        log.info("Found a corresponding document and adding it.")
        return f"{prompt[0]['content']}\n{prompt[2]['content']}\n{prompt[1]['content']}", source_url
    else:
        log.info("No corresponding document found.")
        return f"{prompt[0]['content']}\n{prompt[1]['content']}", None
