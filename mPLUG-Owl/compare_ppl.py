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
'''The image is a close-up of a wooden kitchen table set with a variety of items placed on it.''',
'''The image is a close-up of a wooden bedroom table set with a variety of items placed on it.''',
'''The image is a close-up of a wooden kitchen table set with a variety of computers placed on it.''',
'''The image is a close-up of a large lab computer set with a variety of items placed on it.''',
'''11111111111111111111111111111111''',
# '''The image is a close-up of a wooden kitchen table set with a variety of items placed on it. There are three cups, a bowl, a bottle, a knife, and a spoon. A cake is placed at the center of the table, and a vase is positioned on the right side. A clock is visible in the background, adding a touch of realism.
# In addition to the tableware, there is a potted plant situated on the left side of the scene. The overall arrangement of the items conveys the sense of a casual, everyday setting.'''
]



texts = [prompt + outputs[i] for i in range(len(outputs))]

video_list = ['/mount/data/video_20sec.mp4']

inputs = processor(text=texts, videos=video_list, num_frames=100, return_tensors='pt')
inputs["video_pixel_values"] = inputs["video_pixel_values"][:, :, :4]
inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    ppl = model.get_input_perplexity(**inputs)

print(ppl)