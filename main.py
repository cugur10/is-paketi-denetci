import os
from io import BytesIO

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI


# -----------------------------
# UI / Page config
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

# Header
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
    st.markdown(
        '<div class="small-muted" style="text-align:right;">Sürüm: v2.1</div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### Çıktı")
    st.write("• Otomatik Denetim Raporu (Şablon 1–9)")
    st.divider()
    st.markdown("### Politika")
    st.caption("Yalnızca yüklenen PDF içeriği kullanılır. Harici bilgi üretilmez.")


# KPI row
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(
        '<div class="kpi"><b>1) Belge Yükle</b><br><span class="small-muted">Rapor/İP PDF dosyasını ekleyin.</span></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        '<div class="kpi"><b>2) Otomatik Denetim</b><br><span class="small-muted">Kriterler uygulanır, bulgular çıkarılır.</span></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        '<div class="kpi"><b>3) Standart Rapor</b><br><span class="small-muted">Skor + Uyarılar + Tablo + Karar.</span></div>',
        unsafe_allow_html=True,
    )

st.write("")


# -----------------------------
# Fixed system instruction (sizin sabit talimatınız)
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
# API key / client
# -----------------------------
def get_api_key() -> str | None:
    # Streamlit Cloud secrets
    try:
        v = st.secrets.get("OPENAI_API_KEY")
        if v:
            return v
    except Exception:
        pass
    # env fallback
    return os.environ.get("OPENAI_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error('OPENAI_API_KEY bulunamadı. Streamlit Cloud > Settings > Secrets içine şu formatta ekleyin:\n\nOPENAI_API_KEY = "sk-..."')
    st.stop()

client = OpenAI(api_key=api_key)


# -----------------------------
# PDF text extraction
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


# -----------------------------
# LLM helper
# -----------------------------
def safe_chat(system_prompt: str, user_prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        msg = str(e)
        if "429" in msg or "insufficient_quota" in msg:
            st.error("OpenAI API kota/billing hatası (429). API anahtarınızda kredi/billing yok veya limit dolmuş.")
        else:
            st.error(f"OpenAI API hatası: {e}")
        st.stop()


# -----------------------------
# Report generator (SİZİN İSTEDİĞİNİZ FORMAT)
# -----------------------------
def generate_audit_report(pages_text: str) -> str:
    # Bu PDF'de 46k karakter var; tamamı rahatça girer.
    content = pages_text[:90000]

    user_prompt = f"""
Aşağıdaki metin yüklenen PDF’den çıkarılmıştır. BU METİN DIŞINDA hiçbir bilgi kullanma, uydurma.

KRİTİK KURALLAR:
- ÇIKTI ASLA tek satır “PDF’de yok.” olamaz.
- Her koşulda aşağıdaki 1–9 ŞABLONU üret.
- Bir alan PDF’de yoksa SADECE o alan/satır/hücre için “PDF’de yok / belirtilmemiş” yaz.
- İş paketi adları (İP1, İP2...) açıkça yoksa tabloyu yine üret; “İş Paketi” sütununa “Belirtilmemiş” yaz.

PUANLAMA (model içi):
- Her “UYGUNSUZ” hedef: -10
- Her “İSPAT EKSİK” hedef: -5
- Her kritik uyarı: -2
- Skor 0–100 aralığında kalsın.

ÇIKTI ŞABLONU (AYNEN BAŞLIKLARLA VE BU SIRAYLA):
1) Genel Uygunluk Özeti (0–100)
Skor: <0-100>/100
(2–4 cümle kısa özet)

2) Kritik Uyarılar
- 3–8 madde

3) Hedef Denetimi
Markdown TABLO üret:
İş Paketi | Hedef | Metrik | Değer | Birim | Ölçüm/İspat Yöntemi | Durum
Durum sadece şu üç değerden biri: UYGUN / İSPAT EKSİK / UYGUNSUZ

4) Sonuç–Hedef Karşılaştırması
Markdown TABLO:
İş Paketi | Değerlendirme | Gerekçe
Değerlendirme sadece: Karşılandı / Kısmen Karşılandı / Karşılanmadı

5) Yöntem Uygunluğu / Kilit Riski
- 3–6 madde

6) İspat ve Ölçüm Durumu
- Genel durum
- İstisnalar

7) Karar Önerisi
Devam / Devam (Düzenleme koşuluyla) / Revizyon / Uygun Değil

8) Karar Gerekçesi
- 3–6 madde

9) Düzeltilmesi Gereken Hususlar
- İş paketi bazlı madde madde
- Muğlak hedefleri ölçülebilir örneğe çevir (en az 2 örnek)

PDF METNİ:
{content}
"""

    return safe_chat(SYSTEM_PROMPT, user_prompt)


# -----------------------------
# Main app flow
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
    pages = extract_pages_pymupdf(pdf_bytes)
    total_chars = sum(len(x) for x in pages)
    method_used = "PyMuPDF"

    # fallback (çok düşük çıktıysa)
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
    unsafe_allow_html=True,
)

pages_text = "\n".join(pages).strip()
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


