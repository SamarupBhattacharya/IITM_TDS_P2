from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import re
import subprocess
from google import genai
import tempfile
import os
import time
import shutil
import uuid
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 API KEYS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models
gemini_model = "gemini-2.5-flash"
openai_model = "gpt-5-nano"  # Or gpt-5 when available


# ---------------- Gemini task breakdown ----------------
def task_breakdown(task: str):
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(
        "task_breakdown_prompt.txt", "r", encoding="utf-8", errors="replace"
    ) as f:
        task_breakdown_prompt = f.read()

    response = client.models.generate_content(
        model=gemini_model,
        contents=[task, task_breakdown_prompt],
    )
    return response.text


# ---------------- Helpers ----------------
def extract_last_json_block(text: str):
    matches = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if not matches:
        return None
    return json.loads(matches[-1])


def extract_python_code(text: str):
    matches = re.findall(r"```python\s*(.*?)\s*```", text, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1]


# ---------------- OpenAI calls (httpx) ----------------
async def openai_chat_request(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://aipipe.org/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": openai_model,
                "messages": messages,
                "temperature": 0,
            },
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


async def generate_code_from_breaked_text(breaked_text):
    breaked_content = breaked_text

    metadata = (
        "Metadata:\n"
        "Section 1: Lists all required data sources for the task.\n"
        "Section 2: Provides a step-by-step breakdown for Python implementation.\n"
        "Section 3: Contains extracted parameters and values from the question.\n"
        "Section 4: Specifies machine-readable JSON blocks, including output schema and example.\n"
    )

    instruction = (
        "Instruction:\n"
        "Using the information in breaked_text.txt and the metadata, generate a complete Python script "
        "that performs all described tasks and outputs the answer in the exact format specified in Section 4. "
        "The code must be enclosed entirely within triple backticks and start with ```python.\n"
        "Do not include explanations outside the code block."
    )

    response_text = await openai_chat_request(
        [
            {"role": "system", "content": "You are a Python code generator."},
            {"role": "user", "content": metadata},
            {"role": "user", "content": breaked_content},
            {"role": "user", "content": instruction},
        ]
    )
    return response_text


async def fix_code_with_llm(error_msg: str, old_code: str):
    whitelist = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            whitelist = [line.strip() for line in f if line.strip()]

    whitelist_text = "Whitelisted libraries:\n" + "\n".join(whitelist)

    instruction = (
        "The following Python code failed with an error. Fix the code so that the error is resolved, "
        "and ensure that only the whitelisted libraries are used. "
        "The code must be enclosed in ```python triple backticks, with no explanations outside the block.\n\n"
        f"Error:\n{error_msg}\n\nCode:\n```python\n{old_code}\n```\n\n{whitelist_text}"
    )

    response_text = await openai_chat_request(
        [
            {"role": "system", "content": "You are a senior Python engineer."},
            {"role": "user", "content": instruction},
        ]
    )

    return extract_python_code(response_text or "")


# ---------------- API Route ----------------
@app.post("/api/")
async def upload_files(request: Request):
    start_time = time.time()
    form = await request.form()

    if "questions.txt" not in form:
        return JSONResponse(status_code=400, content={"error": "Missing questions.txt"})

    # Create unique session directory
    session_dir = tempfile.mkdtemp(prefix=f"session_{uuid.uuid4().hex}_")

    # Save questions.txt
    q_file: UploadFile = form["questions.txt"]
    q_path = os.path.join(session_dir, "questions.txt")
    with open(q_path, "wb") as f:
        f.write(await q_file.read())

    # Save attachments
    attachments = []
    for key, value in form.items():
        if key == "questions.txt":
            continue
        if isinstance(value, UploadFile):
            file_path = os.path.join(session_dir, value.filename)
            with open(file_path, "wb") as f:
                f.write(await value.read())
            attachments.append(file_path)

    # Read question text
    with open(q_path, "r", encoding="utf-8", errors="replace") as f:
        question_text = f.read()

    # 1st LLM call → breakdown (Gemini)
    breakdown_text = task_breakdown(question_text)
    extracted_json = extract_last_json_block(breakdown_text)
    if extracted_json is None:
        shutil.rmtree(session_dir, ignore_errors=True)
        return JSONResponse(status_code=400, content={"error": "No JSON block found"})

    try:
        # 2nd LLM call → code generation (OpenAI)
        code_response = await generate_code_from_breaked_text(breakdown_text)
        python_code = extract_python_code(code_response)

        CODE_DIR = "generated_codes"
        os.makedirs(CODE_DIR, exist_ok=True)

        if not python_code:
            shutil.rmtree(session_dir, ignore_errors=True)
            return extracted_json

        # Save generated code (before running)
        gen_code_path = os.path.join(CODE_DIR, "generated_code.py")
        with open(gen_code_path, "w", encoding="utf-8") as f:
            f.write(python_code)

        def run_code(code: str):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".py", mode="w", encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(code)
                tmp_path = tmp_file.name
            try:
                proc = subprocess.run(
                    ["python", tmp_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=175,
                    env={**os.environ, "ATTACHMENTS_DIR": session_dir},
                )
                return proc, tmp_path
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # First run
        proc, _ = run_code(python_code)
        if proc.returncode == 0:
            shutil.rmtree(session_dir, ignore_errors=True)
            return json.loads(proc.stdout)

        print("Error during first run:\n", proc.stderr)

        # 3rd LLM call → fix code (OpenAI)
        fixed_code = await fix_code_with_llm(proc.stderr, python_code)
        if fixed_code:
            # Save fixed code
            fixed_code_path = os.path.join(CODE_DIR, "fixed_code.py")
            with open(fixed_code_path, "w", encoding="utf-8") as f:
                f.write(fixed_code)

            proc2, _ = run_code(fixed_code)
            if proc2.returncode == 0:
                shutil.rmtree(session_dir, ignore_errors=True)
                return json.loads(proc2.stdout)
            else:
                shutil.rmtree(session_dir, ignore_errors=True)
                return extracted_json

        return extracted_json

    except subprocess.TimeoutExpired as e:
        return extracted_json
    except Exception as e:
        return extracted_json
    finally:
        if time.time() - start_time > 160:
            shutil.rmtree(session_dir, ignore_errors=True)
            return extracted_json


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
