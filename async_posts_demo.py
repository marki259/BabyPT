import torch
from nltk import word_tokenize

from dotenv import load_dotenv
import os
import requests

import asyncio
from aiohttp import ClientSession

load_dotenv()

# Get some word embedding
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = os.getenv("HF_TOKEN")


api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

texts = [
    "How do I get a replacement Medicare card?",
    "What is the monthly premium for Medicare Part B?",
    "How do I terminate my Medicare Part B (medical insurance)?",
    "How do I sign up for Medicare?",
    "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
    "How do I sign up for Medicare Part B if I already have Part A?",
    "What are Medicare late enrollment penalties?",
    "What is Medicare and who can get it?",
    "How can I get help with my Medicare Part A and Part B premiums?",
    "What are the different parts of Medicare?",
    "Will my Medicare premiums be higher because of my higher income?",
    "What is TRICARE ?",
    "Should I sign up for Medicare Part B if I have Veterans' Benefits?",
]


sentence_tokens = [word_tokenize(text) for text in texts]


async def request_word_embeddings(words: list[str], queue: asyncio.Queue = None):
    async with ClientSession() as session:
        async with session.post(
            api_url,
            headers=headers,
            json={"inputs": words, "options": {"wait_for_model": True}},
        ) as response:
            response = await response.json()

            tensor_response = torch.tensor(response)

            await queue.put(tensor_response)


async def request_all_embeddings(
    sentence_tokens: list[list[str]],
) -> list[list[list[float]]]:
    embeddings_list = []
    queue = asyncio.Queue()

    async with asyncio.TaskGroup() as group:
        for i in range(len(sentence_tokens)):
            group.create_task(request_word_embeddings(sentence_tokens[i], queue))

    while not queue.empty():
        embeddings_list.append(await queue.get())

    return embeddings_list


n_sentences = len(texts)
list_sentence_embeddings = []

try:
    embeddings_list = torch.load("./data/demo_embeddings.pckl")
except (FileNotFoundError, RuntimeError) as e:
    print(e)
    # Let's just request embeddings for one sentence to start with
    embeddings_list = asyncio.run(request_all_embeddings(sentence_tokens))

    torch.save(embeddings_list, "./data/demo_embeddings.pckl")
