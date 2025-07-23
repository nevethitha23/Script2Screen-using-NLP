
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
print("🔄 Loading GPT-2 model...")
model_name = "gpt2"  # or use "gpt2-medium" or "gpt2-large" for better results
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
os.makedirs("output", exist_ok=True)
print("\n🎬 Enter a one-line story theme (e.g., 'A girl and a wolf in snowy mountains - make it emotional'):")
theme = input("➤ ")
prompt = (
    f"You are a professional storyteller. Your job is to write a touching, highly emotional, and creative story.\n"
    f"Theme: {theme}\n\n"
    f"Write a full short story with rich details, emotions, scenes, and dialogue. Make it sound like a cinematic experience.\n\n"
)
print("\n⏳ Generating offline story. Please wait...")
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs,
    max_length=800,
    temperature=0.9,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

story = tokenizer.decode(outputs[0], skip_special_tokens=True)
output_path = "output/generated_story.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("📜 THEME PROMPT:\n")
    f.write(theme + "\n\n")
    f.write("📖 GENERATED STORY:\n")
    f.write(story)
print(f"\n✅ Story saved to: {output_path}")
