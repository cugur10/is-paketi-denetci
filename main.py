from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import os
from openai import OpenAI

app = FastAPI()

# OpenAI client (API key Render'da ENV olarak verilecek)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
Sen kurum içi bir İş Paketi Denetçi Ajanısın.
Kurumsal ve resmi dil kullan.
Hedeflerin ölçülebilirliğini denetle.
Ölçülebilir hedef; metrik, sayısal eşik, birim ve ölçüm yöntemini içermelidir.
Muğlak ifadeleri (optimize etmek, iyileştirmek, artırmak vb.) uygunsuz say.
Yöntem–hedef uyumunu, ispat/doğrulama durumunu ve kritik riskleri belirt.
Çıktıyı net, kısa ve karar verici şekilde üret.
"""

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        reader = PdfReader(file.file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # 🔴 KRİTİK: Metni sınırla (token patlamasını önler)
        MAX_CHARS = 12000
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Aşağıdaki metni denetle ve raporla:\n{text}"
                }
            ]
        )

        return {"result": response.output_text}

    except Exception as e:
        return {
            "error": "Denetim sırasında hata oluştu",
            "detail": str(e)
        }

