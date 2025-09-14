import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random, sys
import nltk
from nltk.corpus import words

# Per Colab: simulazione input
# sys.argv = ['script.py', '3']

# --- Parametri ---
if len(sys.argv) < 2:
    print("Usage: python script.py <num_runs>")
    sys.exit(1)
num_runs = int(sys.argv[1])

max_input_tokens = 512
max_output_tokens = 64

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
model.eval()

nltk.download('words')
word_list = words.words()
english_words = [w for w in word_list if w.isalpha() and len(w) > 3]
random.shuffle(english_words)

all_inputs = []
all_outputs = []

# --- Esecuzioni ---
for i in range(num_runs):
    print(f"\n=== ESECUZIONE {i+1}/{num_runs} ===")
    
    # Costruisci un prompt breve e realistico
    prompt_words = ' '.join(english_words[i*100:(i+1)*100])
    inputs = tokenizer(prompt_words, return_tensors='pt', truncation=True, max_length=max_input_tokens).to(device)

    outputs = model.generate(**inputs, max_new_tokens=max_output_tokens, do_sample=True)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input:\n{prompt_words[:300]}... [troncato]")
    print(f"Output:\n{generated_text[len(prompt_words):len(prompt_words)+300]}... [troncato]")
    
    all_inputs.append(inputs)
    all_outputs.append(outputs)

print("\n✅ Completato. Libero la memoria GPU.")
del all_inputs
del all_outputs
torch.cuda.empty_cache()
