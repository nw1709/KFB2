import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import pypdf
# --- 1. UI SETUP & STYLING ---
st.set_page_config(layout="wide", page_title="KFB3 Turbo", page_icon=" ")
st.markdown("""
 <style>
 .reportview-container { background: #fafafa; }
 .main { padding-top: 2rem; }
 </style>
""", unsafe_allow_html=True)
st.title(" KFB3 Turbo: Enterprise RAG-Light Edition")
# --- 2. API & CONFIGURATION ---
def get_client():
 key_name = "gemini_key" if "gemini_key" in st.secrets else 
"GOOGLE_API_KEY"
 if key_name not in st.secrets:
 st.error(f"API Key fehlt! Bitte '{key_name}' in den Secrets hinterlegen.")
 st.stop()
 return genai.Client(api_key=st.secrets[key_name])
client = get_client()
# --- 3. RAG-LIGHT: CHUNKING ENGINE (PDF Handling) ---
def process_pdfs_to_chunks(pdf_files):
 """Zerlegt PDFs in Chunks, um Token-Last zu senken (Punkt 1 & 3 vom 
Autor)."""
 chunks = []
 for pdf_file in pdf_files:
 try:
 reader = pypdf.PdfReader(pdf_file)
 for i, page in enumerate(reader.pages):
 text = page.extract_text()
 if text:
 # Wir erstellen Chunks pro Seite für präzise Adressierung
chunks.append(f"--- QUELLE: {pdf_file.name}, SEITE {i+1} 
---¥n{text}")
 except Exception as e:
 st.warning(f"Fehler beim Lesen von {pdf_file.name}: {e}")
 return chunks
# --- 4. MASTER SOLVER PIPELINE (Multi-Model & Verification) ---
def solve_with_architecture(image, chunks):
 """Implementiert Fallback, OCR-Sim und Verification (Punkt 1, 5, 6, 7 vom 
Autor)."""
 
 # Modell-Hierarchie für Fallback-Strategie (Punkt 6 & 7)
 # Wir meiden Preview-Modelle wegen der Instabilität am 8./9. März
 model_hierarchy = ["gemini-1.5-pro", "gemini-1.5-flash"]
 
 # OCR-Pipeline-Simulation: Wir fordern erst die Strukturierung (Punkt 1)
 ocr_sim_instr = "SCHRITT 0: Extrahiere alle sichtbaren Texte und Zahlen aus 
dem Bild deterministisch."
 
 sys_instr = f"""Du bist Auditor am Lehrstuhl für Internes Rechnungswesen 
(FernUni Hagen, Modul 31031).
 {ocr_sim_instr}
 
 RECHTSGRUNDLAGE: Nutze NUR die beigefügten PDF-Chunks. 
 LOGIK: 
 1. Koordinatenschätzung bei Graphen.
 2. Dominanzprüfung (z^a dominiert z^b wenn alle Inputs <= und mind. einer <).
 3. Multiple-Choice: Jede Option einzeln begründen.
 
 VERIFICATION LAYER (Punkt 5): Bevor du antwortest, prüfe deine Rechnung 
auf mathematische Konsistenz."""
 # Wir nehmen nur die relevantesten Chunks (RAG-Light Simulation)
 # Hier senden wir die ersten 15 Chunks, um das Kontextfenster stabil zu halten
 selected_context = "¥n¥n".join(chunks[:15]) 
 img_byte_arr = io.BytesIO()
 image.save(img_byte_arr, format='JPEG', quality=95)
 for model_name in model_hierarchy:
 try:
 parts = [
 
types.Part.from_text(text=f"KONTEXT-CHUNKS:¥n{selected_context}"),
 types.Part.from_bytes(data=img_byte_arr.getvalue(), 
mime_type="image/jpeg"),
 types.Part.from_text(text="Löse die Aufgabe auf dem Bild gemäß 
Modulstandard.")
 ]
 response = client.models.generate_content(
 model=model_name,
 contents=parts,
 config=types.GenerateContentConfig(
 system_instruction=sys_instr,
temperature=0.0,
top_p=0.1,
max_output_tokens=4000
 )
 )
 
 # Einfache Validierung der Antwortlänge/Inhalt
 if len(response.text) > 50 and "Aufgabe" in response.text:
 return response.text, model_name
 else:
 raise ValueError("Antwort unvollständig.")
 except Exception as e:
 st.toast(f"Fallback: {model_name} fehlgeschlagen, versuche 
nächstes...")
 continue
 
 return "Fehler: Alle Modelle (Pro & Flash) konnten keine valide Antwort liefern.", 
"None"
# --- 5. UI INTERACTION ---
with st.sidebar:
 st.header(" System-Status")
 pdf_files = st.file_uploader("1. Skripte hochladen (RAG-Input)", type=["pdf"], 
accept_multiple_files=True)
 if pdf_files:
 with st.spinner("Zerlege Dokumente in Chunks..."):
 st.session_state.chunks = process_pdfs_to_chunks(pdf_files)
 st.success(f"{len(st.session_state.chunks)} Chunks bereit.")
col1, col2 = st.columns([1, 1])
with col1:
 st.header(" Input")
 exam_file = st.file_uploader("2. Klausurblatt scannen", type=["png", "jpg", 
"jpeg"])
 if exam_file:
 img = Image.open(exam_file).convert('RGB')
 st.image(img, use_container_width=True)
with col2:
 st.header(" Lösung")
 if exam_file and pdf_files:
 if st.button(" PRÄZISIONS-LÖSUNG STARTEN", type="primary"):
 with st.spinner("Architektur-Pipeline läuft (RAG -> Multi-Model -> 
Verify)..."):
 result, used_model = solve_with_architecture(img, 
st.session_state.chunks)
 st.info(f"Genutztes Modell: {used_model}")
 st.markdown("---")
 st.write(result)
 else:
 st.warning("Bitte erst Skripte UND Klausurblatt hochladen.")
st.divider()
st.caption("KFB3 Turbo v2026.3.8 | Entspricht Architekturvorschlag Multi-Modal 
Solver v3")
