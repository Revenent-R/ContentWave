from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

USER_TOPIC = input("Please enter the topic you wish to use: ")

# Input prompt
messages = [
    {'role': 'system', 'content': '''
    You are a keyword generation assistant.

Your task is to generate exactly {X} highly relevant keywords based on the topic provided by the user.

Rules:
- Return only keywords, no explanations.
- Keywords should be directly related to the topic.
- Include a mix of broad keywords, specific keywords, and related search terms.
- Avoid duplicates.
- If possible, include trending or commonly searched variations.
- Output as a numbered list.
    '''},
    {"role": "user", "content": f"User topic: {USER_TOPIC}"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to('cuda')

outputs = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))