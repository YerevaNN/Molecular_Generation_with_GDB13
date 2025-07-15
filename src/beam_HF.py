import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
inputs = tokenizer("", return_tensors="pt").to("cuda")

model_path = "/auto/home/knarik/Molecular_Generation_with_GDB13/src/checkpoints/checkpoints_code/Llama-3-1B_tit_hf_4_epochs/step-3126"

model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float16).to("cuda")
model.eval()


outputs = model.generate(**inputs, max_new_tokens=45, num_beams=6, num_beam_groups=3, diversity_penalty=1.0, do_sample=False)
tokenizer.batch_decode(outputs, skip_special_tokens=True)