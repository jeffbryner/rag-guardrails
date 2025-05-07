from agno.models.google import Gemini
from google import genai
from google.genai import types
import google.auth
from agno.knowledge.text import TextKnowledgeBase
from agno.agent import Agent
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.fastembed import FastEmbedEmbedder
from agno.embedder.google import GeminiEmbedder
import lancedb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

credentials, PROJECT_ID = google.auth.default()
GEMINI_PRO = "gemini-1.5-pro"
GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_EMBEDDING = "text-embedding-004"
LOCATION = "us-central1"

generation_config = types.GenerateContentConfig(
    temperature=0,
    top_p=0.1,
    top_k=1,
    max_output_tokens=4096,
)

safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]

# init for Vertex AI API
# client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

model_pro = Gemini(
    id=GEMINI_PRO,
    vertexai=True,
    project_id=PROJECT_ID,
    location="us-central1",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

model_flash = Gemini(
    id=GEMINI_FLASH,
    vertexai=True,
    project_id=PROJECT_ID,
    location="us-central1",
    generation_config=generation_config,
    safety_settings=safety_settings,
)


def create_embedder():
    client_params = {}
    client_params["vertexai"] = True
    client_params["project"] = PROJECT_ID
    client_params["location"] = LOCATION

    embedder = GeminiEmbedder(
        id=GEMINI_EMBEDDING, dimensions=768, client_params=client_params
    )
    return embedder


def create_lance_vector_db():
    lance_vector_db = LanceDb(
        table_name="subjects", uri="./lancedb", embedder=create_embedder()
    )
    return lance_vector_db


def create_knowledge_base(lance_vector_db):
    knowledge_base = TextKnowledgeBase(path="./subjects.txt", vector_db=lance_vector_db)
    return knowledge_base


def create_agent():
    lance_vector_db = create_lance_vector_db()
    knowledge_base = create_knowledge_base(lance_vector_db)
    agent = Agent(
        model=model_flash,
        knowledge=knowledge_base,
        search_knowledge=True,
    )
    agent.knowledge.load(recreate=True)
    return agent


def allowed_subject(subject, embedder=None):
    db = lancedb.connect("./lancedb")
    table = db.open_table("subjects")
    if not embedder:
        embedder = create_embedder()

    # check for matches in our prepared embedding DB
    # adjust the distance range to allow looser correlation to items in subjects.txt
    results = (
        table.search(embedder.get_embedding(subject))
        .distance_type("cosine")
        .distance_range(0, 0.45)
        .to_pandas()
    )
    # debug
    logging.info(results)
    if len(results):
        return True
    else:
        return False
