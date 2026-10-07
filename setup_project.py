from pathlib import Path
from getpass import getpass

from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "wikipedia"


print()
print("Module B setup")
print("=" * 50)

# Choose where to save the large Wikipedia files
print()
print("the wikipedia corpus and faiss indexes need about 65 GB.")
print("they can be stored on this computer or on an external ssd")
print()
print("press Enter to use the default location:")
print(DEFAULT_DATA_DIR)
print()

chosen_path = input("Data location: ").strip()

if chosen_path:
    data_dir = Path(chosen_path).expanduser()
else:
    data_dir = DEFAULT_DATA_DIR

data_dir.mkdir(parents=True, exist_ok=True)


# Download the corpus and indexes
print()
print("Downloading Wikipedia data...")
print("This can take a while and only needs to be done once")
print()

snapshot_download(
    repo_id="rannnran/module-b-wikipedia",
    repo_type="dataset",
    local_dir=str(data_dir)
)


# Save the data path for run_pipeline.py
location_file = BASE_DIR / "data_location.txt"
location_file.write_text(str(data_dir.resolve()))


# Save the OpenAI key locally
print()
api_key = getpass("Enter your openai API key: ").strip()

env_file = BASE_DIR / ".env"
env_file.write_text(f"OPENAI_API_KEY={api_key}\n")


print()
print("=" * 50)
print("Setup complete")
print()
print("Wikipedia data:")
print(data_dir.resolve())
print()
print("The pipeline will remember this location automatically.")
print("Your API key was saved locally in .env.")
print("=" * 50)
