"""This module contains the RAG class which is used to interact with the RAG model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import logger, current_memory
from chroma_functions import ChromaDB

log = logger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RAG:
    """A class to interact with the RAG model."""

    def __init__(self, model_name: str, tokenizer_name: str):
        """Constructor for the RAG class.

        Args:
            model_name (str): The huggingface name of the model to be used.
            tokenizer_name (str): The huggingface name of the tokenizer to be used.
        """
        current_memory()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        current_memory()
        log.info("Initialized model: %s", model_name)

    def generate_text(self, query: str):
        """Takes in a query, searches for the most relevant document, creates a prompt object
        and generates an answer using the RAG model.

        Args:
            query (str): The query of the user.

        Returns:
            generated_text (str), source_url (str): Only the answer part of the generated text
            and the source URL.
        """
        current_memory()
        prompt, source_url = self.create_prompt(query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        prompt_length = input_ids.input_ids.shape[1]
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        log.info("Generating an answer for the query: %s", query)
        current_memory()

        with torch.no_grad():
            output = self.model.generate(
                input_ids.input_ids,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,  # sampling rate for next word prediciton. randomly picks a word of a group of words which cumulative propability are top_p
                # repetition_penalty=1.1,  # penalize words that are already in the text
                max_new_tokens=200,  
                attention_mask=input_ids.attention_mask,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        log.info("Question: %s", query)
        log.info("Answer: %s", generated_text)
        current_memory()
        torch.cuda.empty_cache()
        return generated_text, source_url

    def create_prompt(self, query: str):
        """Looks for the closest document to the user query and creates a prompt object.
        If no document is found, the prompt object will not contain a document nor source URL.

        Args:
            query (str): The query of the user.

        Returns:
            prompt (dict), source_url (str): An prompt object containing
            instructions for the model, the user query and if found, a document and source URL.
        """
        log.info("Adding user query to prompt object")
        prompt = [
            {
                "role": "system",
                "content": """
                Du bist Experte über die Hochschule der Medien in Stuttgart und hilfst 
                Leuten bei spezifischen Fragen dazu aus.
                """,
            },
            {"role": "user", "content": query},
        ]
        chroma = ChromaDB("hdm_collection")
        results = chroma.get_closest_document(query)
        if results:
            retrieved_doc = results[0]
            document_text = retrieved_doc["document"]
            source_url = retrieved_doc["metadata"].get("url", "Quelle nicht verfügbar")
            
            prompt.append(
                {
                    "role": "system",
                    "content": f"Hier ist eine relevante Information aus einer Quelle der HdM: {document_text}",
                }
            )
            log.info("Found a corresponding document and adding it.")
            return (
                f"{prompt[0]['content']}\n{prompt[2]['content']}\n{prompt[1]['content']}",
                source_url,
            )

        log.info("No corresponding document found.")
        return f"{prompt[0]['content']}\n{prompt[1]['content']}", None
