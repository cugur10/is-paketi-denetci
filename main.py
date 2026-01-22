from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
import os
from typing import List, Tuple

from openai import OpenAI

app = FastAPI(title="Proje Denetçi API", version="1.0.0")

# OpenAI client (API key ENV: OPENAI_API_KEY)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is missing.")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 1) KANONİK SİSTEM TALİMATI
# -----------------------------
AUDITOR_SYSTEM_PROMPT = r"""
SEN BİR “İŞ PAKETİ DENETÇİ AJANI”SIN.

Rolün bir chatbot olmak, soru cevaplamak veya kullanıcıyı yönlendirmek DEĞİLDİR.
Rolün; yüklenen belgeyi kurumsal Ar-Ge proje denetimi bakış açısıyla,
eleştirel, tarafsız ve kanıta dayalı biçimde incelemektir.

KULLANICIYA SORU SORMA.
KULLANICIDAN TALİMAT BEKLEME.
YALNIZCA BELGEYİ DENETLE.

---

TEMEL DENETİM YAKLAŞIMI:

Belgeyi aşağıdaki başlıklar altında incele:

1. Proje amacı açık, ölçülebilir ve somut mu?
2. İş paketleri (İP) net biçimde tanımlanmış mı?
3. Her iş paketinin:
   - çıktısı (deliverable),
   - başarım / kabul kriteri,
   - ölçülebilir hedefi
   açıkça belirtilmiş mi?
4. İş paketleri ile zaman planı uyumlu mu?
5. İş paketleri ile bütçe kalemleri tutarlı mı?
6. Personel, görev dağılımı ve yetkinlikler net mi?
7. Riskler tanımlanmış mı ve önlemler gerçekçi mi?
8. Önceki projelerle fark ve yenilik açık mı?
   - “Bu zaten önceki projenin hedefiydi” izlenimi var mı?
9. Ölçeklenebilirlik, tekrar üretilebilirlik ve ürünleşme potansiyeli net mi?
10. Pilot / saha / doğrulama aşamaları gerçekçi mi?

---

DENETÇİ DİLİ VE DAVRANIŞ KURALLARI:

- Belge içinde kanıt yoksa “BELGEDE BU BİLGİYE RASTLANMAMIŞTIR” de.
- Varsayım yapma.
- Yumuşatma yapma.
- “Bence”, “muhtemelen”, “olabilir” gibi ifadeler kullanma.
- Her bulgu için mümkünse sayfa/bölüm referansı ver.
- Majör ve minör uygunsuzluk ayrımı yap.

---

MAJÖR UYGUNSUZLUK ÖRNEKLERİ (BUNLAR CİDDİ HATA SAYILIR):

- İş paketinde çıktı tanımlanmamış olması
- Ölçülebilir hedef olmaması
- “Danışmanlık”, “genel çalışma”, “araştırılacaktır” gibi muğlak ifadeler
- Önceki projenin aynısının yeniden yazılmış olması
- Pilot / saha doğrulamasının belirsiz olması
- Ürünleşme yolunun tanımlanmaması

---

ÇIKTI FORMATIN (ZORUNLU):

1. DENETİM ÖZETİ
   - Genel değerlendirme (Uygun / Kısmen Uygun / Uygun Değil)

2. İŞ PAKETİ BAZLI BULGULAR
   - Her iş paketi için kısa değerlendirme

3. UYGUNSUZLUKLAR
   - Majör Uygunsuzluklar
   - Minör Uygunsuzluklar

4. DÜZELTME ÖNERİLERİ
   - Net, kısa, uygulanabilir maddeler

5. GENEL DENETÇİ YORUMU
   - Bu projenin bu haliyle kabul edilip edilemeyeceğine dair açık görüş

---

UNUTMA:
SEN BİR DENETÇİSİN.
AMACIN BEĞENMEK DEĞİL, UYGUNLUĞU KONTROL ETMEKTİR.
""".strip()


# -----------------------------
# 2) PDF METİN ÇIKARMA
# -----------------------------
def extract_text_with_pages(reader: PdfReader) -> List[Tuple[int, str]]:
    """
    Returns: [(page_no, text), ...] page_no starts at 1
    """
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        # normalize a bit
        t = t.replace("\x00", "").strip()
        pages.append((i, t))
    return pages


def build_audit_payload(pages: List[Tuple[int, str]], max_chars: int) -> str:
    """
    Packs page texts with page markers. Truncates safely to max_chars.
    """
    # Build full text with page references.
    chunks = []
    for page_no, text in pages:
        if not text:
            continue
        chunks.append(f"\n=== SAYFA {page_no} ===\n{text}\n")
    combined = "\n".join(chunks).strip()

    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "\n\n[TRUNCATED: metin limit nedeniyle kesildi]"
    return combined


# -----------------------------
# 3) OPSİYONEL OCR (istersen)
# -----------------------------
ENABLE_OCR = os.environ.get("ENABLE_OCR", "0") == "1"

def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = 25) -> List[Tuple[int, str]]:
    """
    Optional OCR. Requires: pdf2image + pytesseract + system deps (poppler, tesseract).
    Returns [(page_no, ocr_text), ...]
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except Exception as e:
        raise RuntimeError(
            "OCR is enabled but dependencies are missing. "
            "Install pdf2image + pytesseract and system packages (poppler, tesseract)."
        ) from e

    images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=max_pages)
    out = []
    for idx, img in enumerate(images, start=1):
        txt = pytesseract.image_to_string(img, lang="tur") or ""
        txt = txt.strip()
        out.append((idx, txt))
    return out


# -----------------------------
# 4) ENDPOINTS
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload a PDF. Returns an audit report.
    No Q&A. No user instructions. Always runs audit with fixed system prompt.
    """
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes or len(pdf_bytes) < 100:
            raise HTTPException(status_code=400, detail="Dosya boş veya okunamadı.")

        # Parse PDF
        reader = PdfReader(file=pdf_bytes)
        pages = extract_text_with_pages(reader)

        # Determine if text extraction is sufficient
        raw_text_len = sum(len(t) for _, t in pages)
        MIN_TEXT_CHARS = 500  # below this, it's likely scanned or protected

        # If no text, try OCR (optional) else return clear error
        if raw_text_len < MIN_TEXT_CHARS:
            if ENABLE_OCR:
                ocr_pages = ocr_pdf_bytes(pdf_bytes, max_pages=25)
                raw_text_len = sum(len(t) for _, t in ocr_pages)
                if raw_text_len < MIN_TEXT_CHARS:
                    raise HTTPException(
                        status_code=422,
                        detail="PDF’den yeterli metin elde edilemedi. OCR denendi ama sonuç yetersiz."
                    )
                pages_for_audit = ocr_pages
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "PDF içinden metin çıkarılamadı (muhtemelen tarama PDF / metin katmanı yok). "
                        "Bu uygulama denetim için metne ihtiyaç duyar. OCR gerekir. "
                        "Sunucuda OCR açmak için ENABLE_OCR=1 ve OCR bağımlılıkları gereklidir."
                    )
                )
        else:
            pages_for_audit = pages

        # Token/Cost control: cap chars
        MAX_CHARS = int(os.environ.get("MAX_AUDIT_CHARS", "24000"))
        audit_text = build_audit_payload(pages_for_audit, max_chars=MAX_CHARS)

        # Call OpenAI: fixed audit behavior
        response = client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": audit_text},
            ],
        )

        return {
            "file_name": file.filename,
            "extracted_chars": len(audit_text),
            "result": response.output_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Generic error
        raise HTTPException(status_code=500, detail=f"Denetim sırasında hata oluştu: {str(e)}")

