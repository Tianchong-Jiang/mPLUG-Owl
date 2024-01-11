# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="MAGAer13/mplug-owl2-llama2-7b", local_dir="/models/mplug-owl2-llama2-7b")
from huggingface_hub import snapshot_download
snapshot_download(repo_id="MAGAer13/mplug-owl-llama-7b-video", local_dir="/mount/mplug-owl-llama-7b-video", cache_dir="/mount/cache")