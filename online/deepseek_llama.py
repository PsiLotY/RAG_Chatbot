import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList
from huggingface_hub import login
from pathlib import Path
from chroma_functions import get_closest_document
from utils import logger, current_memory

log = logger("deepseek")

class SentenceStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return decoded_text.endswith((".", "!", "?"))  # Stop at sentence end



stopping_criteria = StoppingCriteriaList([SentenceStoppingCriteria(),])

# login(token="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")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)

llama_deep = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Use 8-bit quantization to reduce memory usage
quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # Use 4-bit if needed: load_in_4bit=True

current_memory()

# Load model with automatic device mapping and quantization
model = LlamaForCausalLM.from_pretrained(
    llama_deep,
    device_map="auto",  # Automatically distribute model between GPU and CPU if needed
    quantization_config=quantization_config
)

current_memory()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(llama_deep)
query = "Ist die HdM geeignet für ein Auslandssemester oder Austauschjahr?"
results = get_closest_document(query, log)

prompt = [
    {"role": "system", "content": "Du bist Experte über die Hochschule der Medien in Stuttgart und hilfst Leuten bei spezifischen Fragen dazu aus. Deine Antworten sollten auf der gleichen Sprache sein wie das geschriebene vom User. Gebe dem Nutzer immer die Quelle an woher du die Information bekommen hast."},
    {"role": "user", "content": query},
]


if results:
    retrieved_doc = results[0]  # Get the most relevant document
    document_text = retrieved_doc["document"]
    source_url = retrieved_doc["metadata"].get("url", "Quelle nicht verfügbar")

    prompt.append(
        {
            "role": "system",
            "content": f"Hier ist eine relevante Information aus einer Quelle der HdM: {document_text}... [Quelle: {source_url}]",
        }
    )

# Encode prompt
input_text = f"{prompt[0]['content']}\n{prompt[1]['content']}\n{prompt[2]['content']}"
input_ids = tokenizer(input_text, return_tensors="pt").to(device)
prompt_length = input_ids.input_ids.shape[1]

pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# A. Adjustable Response Length
#     Give users control over response length.
#     Example: “Would you like a brief or detailed answer?”

# B. Limit Based on Context
#     Simple questions → Shorter responses.
#     Complex queries → Allow longer explanations.

# C. Summary + Expandable Details
#     Provide a short answer first, then offer "More details?".

# Generate response
with torch.no_grad():
    output = model.generate(
        input_ids.input_ids,
        do_sample=True,
        temperature=1.0,
        top_p=0.7, # sampling rate for next word prediciton. randomly picks a word of a group of words which cumulative propability are top_p
        max_new_tokens=300,  # Reduce token count to save memory
        attention_mask=input_ids.attention_mask,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
       # stopping_criteria=stopping_criteria,
    )

generated_tokens = output[0][prompt_length:]

# Decode response
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

log.info("Generated Response:")
log.info(generated_text)