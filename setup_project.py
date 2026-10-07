from pathlib import Path
from getpass import getpass

from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "wikipedia"


print("Module B setup")
print()
print("the wikipedia corpus and faiss indexes need about 65 gb")
print("press Enter to store them inside the project")
print("or enter another location like an external ssd")
print()

chosen_path = input(f"Data location [{DEFAULT_DATA_DIR}]: ").strip()

if chosen_path:
    data_dir = Path(chosen_path).expanduser()
else:
    data_dir = DEFAULT_DATA_DIR

data_dir.mkdir(parents=True, exist_ok=True)


# ask for the API key before the download starts
print()
api_key = ""

while not api_key:
    api_key = getpass("enter your openai API key: ").strip()

    if not api_key:
        print("api key cannot be empty please try again")

env_file = BASE_DIR / ".env"
env_file.write_text(f"OPENAI_API_KEY={api_key}\n")

print("api key saved")
print()
print("downloading wikipedia data...")
print("this may take a while")
print()


snapshot_download(
    repo_id="rannnran/module-b-wikipedia",
    repo_type="dataset",
    local_dir=str(data_dir)
)


location_file = BASE_DIR / "data_location.txt"
location_file.write_text(str(data_dir.resolve()))


print()
print("Setup complete.")
print("Wikipedia data location:")
print(data_dir.resolve())