## Why?

A proof of concept idea using RAG to act as guardrails for an app accepting YOLO input intended as a content for a LLM. 
Instead of choosing a 'deny list' approach, this uses an approach of creating an 'allow list' of subjects that are appropriate and only allow sending those further into the application. 

# What?

This uses an in-process embedding database provided by lanceDB and an allow-listed set of questions representing subjects that are appropriate. 

# Tech
This uses: 
- lancedb as a pre-generated set of embeddings
- Gemini in vertex/GCP for embeddings and LLM
- agno for the agent library
- rich as a simple UI

# Usage
Install using uv 

```
# mac
brew install uv

# windows
powershell -c "irm https://astral.sh/uv/install.ps1 | more"

# or install from https://github.com/astral-sh/uv/releases
```

Setup the code
```
git clone git@github.com:jeffbryner/rag-guardrails.git
cd rag-guardrails
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```


Use the example subjects.txt file to allow only discussion of ice cream or shakespeare. You can seed the file with other topics as desired. 

To test out the guardrails, create a lancedb using the topics in subjects.txt by running:
```
python create_embeddings.py
```

You can then test which topics pass/fail 
```
python guardrails.py
```
and you'll get a simple prompt for input with a verdict about whether it would be allowed or not. 

You can adjust the distance parater in the allowed_subject function to allow greater leeway in topics. 



