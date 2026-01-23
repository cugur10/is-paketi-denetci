import os
from io import BytesIO
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import faiss
import openai

st.set_page_config(
    page_title="İş Paketi Denetçi Ajanı",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
.small-muted { color: #6b7280; font-size: 0.92rem; }
.header {
  padding: 14px 16px;
  border: 1px solid rgba(120,120,120,.25);
  border-radius: 12px;
  background: rgba(240,240,240,.35);
}
.kpi {
  border: 1px solid rgba(120,120,120,.25);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(255,255,255,.6);
}
.card {
  border: 1px solid rgba(120,120,120,.25);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(240,240,240,.20);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 3, 1], vertical_alignment="center")
with c1:
    st.markdown("**LOGO**")
with c2:
    st.markdown("## İş Paketi Denetçi Ajanı")
    st.markdown(
        '<div class="small-muted">Demo • Kurum içi talimatlar sabit • PDF yüklenince otomatik denetim raporu üretilir</div>',
        unsafe_allow_html=True
    )
with c3:
    st.markdown('<div class="small-muted" style="text-align:right;">Sürüm: v2.0</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# Sabit talimat (sizinki)
SYSTEM_PROMPT = """
Sen kurum içi bir İş Paketi Denetçi Ajanısın.
Kurumsal ve resmi dil kullan.
Yalnızca yüklenen PDF içeriğine dayan.
PDF dışında bilgi uydurma.
Hedeflerin ölçülebilirliğini denetle.
Ölçülebilir hedef; metrik, sayısal eşik, birim ve ölçüm yöntemini içermelidir.
Muğlak ifadeleri (optimize etmek, iyileştirmek, artırmak vb.) uygunsuz say.
Yöntem–hedef uyumunu, ispat/doğrulama durumunu ve kritik riskleri belirt.
Çıktıyı net, kısa ve karar verici şekilde üret.
"""

# API key
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    api_key = None
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY bulunamadı. Streamlit Cloud > Secrets içine ekleyin.")
    st.stop()
openai.api_key = api_key

# Sidebar
with st.sidebar:
    st.markdown("### Çıktı")
    st.write("• Otomatik Denetim Raporu")
    st.divider()
    st.markdown("### Politika")
    st.caption("Yalnızca yüklenen PDF içeriği kullanılır. Harici bilgi üretilmez.")

# KPI row
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown('<div class="kpi"><b>1) Belge Yükle</b><br><span class="small-muted">Rapor/İP PDF dosyasını ekleyin.</span></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi"><b>2) Otomatik Denetim</b><br><span class="small-muted">Kriterler uygulanır, bulgular çıkarılır.</span></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi"><b>3) Standart Rapor</b><br><span class="small-muted">Bulgular/Eksikler/Riskler/Aksiyonlar.</span></div>', unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader("Rapor/PDF yükleyin (İP denetimi)", type=["pdf"])

def extract_pages_pymupdf(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [(p.get_text("text") or "") for p in doc]

def extract_pages_pdfplumber(pdf_bytes: bytes) -> list[str]:
    out = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return out

def make_chunks(pages: list[str]) -> list[str]:
    chunks = []
    for i, text in enumerate(pages):
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        for part in parts:
            if len(part) >= 40:
                chunks.append(f"(Sayfa {i+1}) {part}")
    return chunks

def safe_chat(messages):
    try:
        return openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        ).choices[0].message.content
    except Exception as e:
        msg = str(e)
        if "429" in msg or "insufficient_quota" in msg:
            st.error("OpenAI API kota/billing hatası (429). API anahtarınızda kredi/billing yok veya limit dolmuş.")
        else:
            st.error(f"OpenAI API hatası: {e}")
        st.stop()

def generate_audit_report(pages_text: str) -> str:
    # PDF çok uzunsa, LLM'e tamamını basmak yerine ilk N karakter/özet yaklaşımı gerekir.
    # Demo için güvenli limit: ilk 60k karakter.
    content = pages_text[:60000]

    user_prompt = f"""
Aşağıdaki içerik yüklenen PDF’den çıkarılmış metindir.

GÖREV:
- Bu metni iş paketi denetimi açısından değerlendir.
- “ölçülebilir hedef” kontrolünü uygula (metrik, eşik, birim, ölçüm yöntemi).
- Muğlak fiilleri tespit et (optimize/iyileştir/artır vb.) ve uygunsuz olarak işaretle.
- Yöntem–hedef uyumu, doğrulama/kanıt, kritik riskleri belirt.
- Sadece metinde olanı kullan, yoksa “PDF’de yok” de.
"""

    return safe_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )


if not uploaded_file:
    st.markdown('<div class="card"><b>Başlamak için</b><br><span class="small-muted">PDF yükleyin. Sistem otomatik denetim raporu üretecektir.</span></div>', unsafe_allow_html=True)
    st.stop()

pdf_bytes = uploaded_file.getvalue()

with st.spinner("PDF metni çıkarılıyor..."):
    pages = extract_pages_pymupdf(pdf_bytes)
    total_chars = sum(len(x) for x in pages)
    method_used = "PyMuPDF"

    if total_chars < 200:
        pages2 = extract_pages_pdfplumber(pdf_bytes)
        total_chars2 = sum(len(x) for x in pages2)
        if total_chars2 > total_chars:
            pages = pages2
            total_chars = total_chars2
            method_used = "pdfplumber"

st.markdown(
    f'<div class="card"><b>PDF Durumu</b><br>'
    f'<span class="small-muted">Metin çıkarma yöntemi: <b>{method_used}</b> • Toplam karakter: <b>{total_chars}</b></span></div>',
    unsafe_allow_html=True
)

pages_text = "\n".join(pages).strip()
if not pages_text:
    st.error("PDF içinden metin çıkarılamadı. (Metin katmanı yok / özel encoding). OCR gerekebilir.")
    st.stop()

with st.expander("Çıkan metin önizleme (ilk 300 karakter)"):
    st.code(pages_text[:300])

# Otomatik rapor üretimi
st.subheader("Otomatik Denetim Raporu")
with st.spinner("Denetim raporu hazırlanıyor (LLM)..."):
    report = generate_audit_report(pages_text)

st.markdown(report)

# İndirilebilir çıktı (opsiyonel)
st.download_button(
    label="Raporu indir (TXT)",
    data=report.encode("utf-8"),
    file_name="denetim_raporu.txt",
    mime="text/plain",
)


