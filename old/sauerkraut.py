import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import logger, current_memory
from chroma_functions import get_closest_document


log = logger("sauerkraut")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sauerkraut = "VAGOsolutions/SauerkrautLM-7b-HerO"

query = "Ist die HdM geeignet für ein Auslandssemester oder Austauschjahr?"
results = get_closest_document(query, log)
prompt = [
    {"role": "system", "content": "Du bist Experte über die Hochschule der Medien in Stuttgart und hilfst Leuten bei spezifischen Fragen dazu aus. Gebe dem Nutzer immer die Quelle an woher du die Information bekommen hast."},
    {"role": "user", "content": query},
]

# zeugs = "Herzlich willkommen an der Hochschule der Medien - so heißt es zu Beginn jedes Semesters beim Begrüßungsfrühstück für unsere Austauschstudenten aus der ganzen Welt. Neben praxisnahem Unterricht in kleinen Gruppen und mit neuester technischer Ausstattung ist es das Betreuungsprogramm durch unser HdM Exchange Network, dass die Hochschule der Medien so beliebt als Erstwunsch für ein Auslandssemester macht. Für jeden Austauschstudenten findet sich ein HdM-Student, der sich als Buddy um einen sanften Start in der neuen Heimat kümmert. Erster Mailkontakt und die Abholung am Bahnhof oder Flughafen sowie die Begleitung bei den Behördengängen gehört ebenso dazu wie das gemeinsame Durchstreifen des Stuttgarter Nachtlebens oder zahlreiche Unternehmungen, die man so anderswo nicht erleben kann."

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
    input_text = f"{prompt[0]['content']}\n{prompt[1]['content']}\n{prompt[2]['content']}"
else:
    input_text = f"{prompt[0]['content']}\n{prompt[1]['content']}"

current_memory()

torch.cuda.empty_cache()

current_memory()

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # bnb_4bit_compute_dtype=torch.float16  # Use FP16 for compute instead of FP32
)

tokenizer = AutoTokenizer.from_pretrained(sauerkraut)
model = AutoModelForCausalLM.from_pretrained(
    sauerkraut,
    device_map="auto",
    quantization_config=quantization_config
    )

current_memory()

input_ids = tokenizer(input_text, return_tensors="pt").to(device)
prompt_length = input_ids.input_ids.shape[1]


pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

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

current_memory()

generated_tokens = output[0][prompt_length:]


# Decode response
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

log.info("Generated Response:")
log.info(generated_text)