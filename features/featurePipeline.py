from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import load_dataset, DatasetDict
from datasets import Audio
import hopsworks
from huggingface_hub import login
with open("../token.txt", "r") as file:
    token = file.readline().strip()


login(token=token, add_to_git_credential=True)
project = hopsworks.login()

common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True)
print(common_voice)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
print(common_voice)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

print(common_voice["train"][0])

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)
# common_voice.save_to_disk("common_voice")

#HuggingFace
common_voice.push_to_hub("dacavi/hindi-dataset")

#Hopsworks
# dataset_api = project.get_dataset_api()
# uploaded_file_path = dataset_api.upload(
#     local_path = "./common_voice/dataset_dict.json",
#     upload_path = "/Projects/Lab1_ID2222/Lab1_ID2222_Training_Datasets/common_voice", overwrite=True)
#
# uploaded_file_path = dataset_api.upload(
#     local_path = "./common_voice/train/state.json",
#     upload_path = "/Projects/Lab1_ID2222/Lab1_ID2222_Training_Datasets/common_voice/train/", overwrite=True)
#
# uploaded_file_path = dataset_api.upload(
#     local_path = "./common_voice/train/dataset_info.json",
#     upload_path = "/Projects/Lab1_ID2222/Lab1_ID2222_Training_Datasets/common_voice/train/", overwrite=True)
#
# uploaded_file_path = dataset_api.upload(
#     local_path = "./common_voice/train/data-00000-of-00001.arrow",
#     upload_path = "/Projects/Lab1_ID2222/Lab1_ID2222_Training_Datasets/common_voice/train/", overwrite=True)
#
# uploaded_file_path = dataset_api.upload(
#     local_path = "./common_voice/test/state.json",
#     upload_path = "/Projects/Lab1_ID2222/Lab1_ID2222_Training_Datasets/common_voice/test/", overwrite=True)
#
# file_path = dataset_api.upload(
#     local_path = "./common_voice/test/dataset_info.json",
#     upload_path = "/Projects/Lab1_ID2222/Lab1_ID2222_Training_Datasets/common_voice/test/", overwrite=True)
#
# uploaded_file_path = dataset_api.upload(
#     local_path = "./common_voice/test/data-00000-of-00001.arrow",
#     upload_path = "/Projects/Lab1_ID2222/Lab1_ID2222_Training_Datasets/common_voice/test/", overwrite=True)


