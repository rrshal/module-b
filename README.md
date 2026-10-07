Module B v1

Setup

1. Clone the repository

git clone <REPOSITORY_URL>
cd <REPOSITORY_FOLDER>

2. Create a virtual environment

Mac:
python3 -m venv .venv
source .venv/bin/activate

Windows:
python -m venv .venv
.venv\Scripts\activate


3. Install the requirements
pip install -r requirements.txt


4. Run the setup_project.py
python setup_project.py

Because it will:
- ask where to store the Wikipedia corpus and FAISS indexes
- download the required retrieval files from Hugging Face
- ask for your OpenAI API key
- save the selected data location automatically

The wikipedia corpus and fails indexes require about 65 gb of storage

To run the pipeline:

Example:
python run_pipeline.py \
  --input INPUT_FILE.json \
  --output-dir outputs/test \
  --source-dataset TruthfulQA \
  --id-field example_id \
  --response-field generated_response

