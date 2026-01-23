import os
from io import BytesIO

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI


# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="İş Paketi Denetçi Ajanı",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="header">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 3, 1], vertical_alignment="center")
with c1:
    st.markdown("**LOGO**")
with c2:
    st.markdown("## İş Paketi Denetçi Ajanı")
    st.markdown(
        '<div class="small-muted">Demo • Kurum içi talimatlar sabit • PDF yüklenince otomatik denetim raporu üretilir</div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown('<div class="small-muted" style="text-align:right;">Sürüm: v2.2</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### Menü")
    st.write("• Belge Yükle")
    st.write("• Otomatik Denetim Raporu")
    st.divider()
    st.markdown("### Politika")
    st.caption("Uygulama yalnızca yüklenen PDF içeriğine dayanır. Harici bilgi üretilmez.")


# -----------------------------
# Sabit talimat (değişmeyecek)
# -----------------------------
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


# -----------------------------
# OpenAI client
# -----------------------------
def get_api_key() -> str | None:
    try:
        v = st.secrets.get("OPENAI_API_KEY")
        if v:
            return v
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error('OPENAI_API_KEY bulunamadı. Secrets içine şu formatla ekleyin:\n\nOPENAI_API_KEY = "sk-..."')
    st.stop()

client = OpenAI(api_key=api_key)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pages_pymupdf(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [(p.get_text("text") or "") for p in doc]


def extract_pages_pdfplumber(pdf_bytes: bytes) -> list[str]:
    out = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return out


def get_pdf_text(pdf_bytes: bytes) -> tuple[str, str, int]:
    pages = extract_pages_pymupdf(pdf_bytes)
    total_chars = sum(len(x) for x in pages)
    method = "PyMuPDF"

    if total_chars < 300:
        pages2 = extract_pages_pdfplumber(pdf_bytes)
        total2 = sum(len(x) for x in pages2)
        if total2 > total_chars:
            pages = pages2
            total_chars = total2
            method = "pdfplumber"

    text = "\n".join(pages).strip()
    return text, method, total_chars


# -----------------------------
# Report generator (final)
# -----------------------------
def generate_audit_report(pages_text: str) -> str:
    content = pages_text[:90000]  # sizde 46k, tamamı giriyor

    user_prompt = f"""
Aşağıdaki metin PDF’den çıkarılmıştır. SADECE BU METNE dayan.

ZORUNLU FORMAT:
- ÇIKTI ASLA tek satır “PDF’de yok.” olamaz.
- Her koşulda 1–9 başlıklar üretilecek.
- “PDF’de yok / belirtilmemiş” ifadesi SADECE ilgili madde/hücre için kullanılacak.
- Metinde açıkça yoksa “bütçe, çevresel etki, toksik solvent, regülasyon” gibi konuları EKLEME.

KANIT KURALI:
- 2) Kritik Uyarılar bölümünde her uyarı maddesinin sonunda, metinden dayanak 3–10 kelimelik mini alıntı ver (tırnak içinde).

ÇIKTI ŞABLONU (AYNEN):
1) Genel Uygunluk Özeti (0–100)
Skor: <0-100>/100

2) Kritik Uyarılar
- ...

3) Hedef Denetimi
İş Paketi | Hedef | Metrik | Değer | Birim | Ölçüm/İspat Yöntemi | Durum
Durum: UYGUN / İSPAT EKSİK / UYGUNSUZ

4) Sonuç–Hedef Karşılaştırması
İş Paketi | Değerlendirme | Gerekçe
Değerlendirme: Karşılandı / Kısmen Karşılandı / Karşılanmadı

5) Yöntem Uygunluğu / Kilit Riski

6) İspat ve Ölçüm Durumu

7) Karar Önerisi
Devam / Devam (Düzenleme koşuluyla) / Revizyon / Uygun Değil

8) Karar Gerekçesi

9) Düzeltilmesi Gereken Hususlar
- En az 2 muğlak hedefi ölçülebilir örneğe çevir

PUANLAMA:
- UYGUNSUZ: -10
- İSPAT EKSİK: -5
- Kritik uyarı: -2
- Skor 0–100 aralığında.

PDF METNİ:
<<<
{content}
>>>
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,  # stabil
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# Main
# -----------------------------
uploaded_file = st.file_uploader("Rapor/PDF yükleyin (İP denetimi)", type=["pdf"])

if not uploaded_file:
    st.markdown(
        '<div class="card"><b>Başlamak için</b><br><span class="small-muted">PDF yükleyin. Sistem otomatik denetim raporu üretecektir.</span></div>',
        unsafe_allow_html=True,
    )
    st.stop()

pdf_bytes = uploaded_file.getvalue()

with st.spinner("PDF metni çıkarılıyor..."):
    pages_text, method_used, total_chars = get_pdf_text(pdf_bytes)

st.markdown(
    f'<div class="card"><b>PDF Durumu</b><br>'
    f'<span class="small-muted">Metin çıkarma yöntemi: <b>{method_used}</b> • Toplam karakter: <b>{total_chars}</b></span></div>',
    unsafe_allow_html=True,
)

if not pages_text:
    st.error("PDF içinden metin çıkarılamadı. (Metin katmanı yok / özel encoding). OCR gerekebilir.")
    st.stop()

with st.expander("Çıkan metin önizleme (ilk 300 karakter)"):
    st.code(pages_text[:300])

st.subheader("Otomatik Denetim Raporu")
st.caption(f"LLM'e gönderilen karakter: {min(len(pages_text), 90000)}")

with st.spinner("Denetim raporu hazırlanıyor (LLM)..."):
    report = generate_audit_report(pages_text)

st.markdown(report)

st.download_button(
    label="Raporu indir (TXT)",
    data=report.encode("utf-8"),
    file_name="denetim_raporu.txt",
    mime="text/plain",
)
