# app.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import secrets

app = FastAPI()
security = HTTPBasic()

USERNAME = "admin"
PASSWORD = "secret"

# Auth
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# Load base model and LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, "monsimas/medgemma-4b-it-sft-lora-autocompletion")
tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")

@app.post("/predict")
def predict(prompt: str, username: str = Depends(authenticate)):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"completion": result}
