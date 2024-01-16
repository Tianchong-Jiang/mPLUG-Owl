import os
os.environ['TRANSFORMERS_CACHE'] = '/mount/cache'

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

prompt = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Can you describe the video?
AI: '''

outputs = [
'''The video shows an robot opening a microwave oven.''',
'''The video shows a human opening a microwave oven.''',
'''The video shows an robot opening a refrigerator door.''',
'''The video shows an robot making a beef burger.''',
'''The video shows cats playing with a toy.'''
]

# outputs = [
# '''The video shows an robot turnning on a light.''',
# '''The video shows a human opening a microwave oven.''',
# '''The video shows an robot opening a refrigerator door.''',
# '''The video shows an robot making a beef burger.''',
# '''The video shows cats playing with a toy.'''
# ]



texts = [prompt + outputs[i] for i in range(len(outputs))]

# video_list = ['/mount/data/franka_videos/opening_microwave_oven.mp4'] * 5
video_list = ['/mount/data/franka_videos/turning_on_light.mp4'] * 5

print(video_list)

inputs = processor(text=texts, videos=video_list, num_frames=100, return_tensors='pt')
# inputs["video_pixel_values"] = inputs["video_pixel_values"][:, :, :4]
inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    ppl = model.get_input_perplexity(**inputs)

print(ppl)