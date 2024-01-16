import os
os.environ['TRANSFORMERS_CACHE'] = '/mount/cache'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

pretrained_ckpt = '/mount/mplug-owl-llama-7b-video'

model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
    device_map={'': 0} # https://github.com/X-PLUG/mPLUG-Owl/issues/100#issuecomment-1622830827
)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

# We use a human/AI template to organize the context as a multi-turn conversation.
# <|video|> denotes an video placehold.
# prompts = [
# '''
# The following is an image of a interior of a room.
# Human: Describe the image
# AI: ''']

prompts = [ '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Can you describe the video?
AI: ''']

# video_list = ['/mount/data/franka_videos/opening_microwave_oven.mp4']
video_list = ['/mount/data/franka_videos/turning_on_light.mp4']

# generate kwargs (the same in transformers) can be passed in the do_generate()
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}
inputs = processor(text=prompts, videos=video_list, num_frames=100, return_tensors='pt')
# inputs["video_pixel_values"] = inputs["video_pixel_values"][:, :, :4]
inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    res = model.generate(**inputs, **generate_kwargs)
sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
print(sentence)