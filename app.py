import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import os
import json
from google.oauth2 import service_account

# --- 1. UI SETUP ---
st.set_page_config(layout="wide", page_title="KFB2", page_icon="🦊")

st.markdown(f'''
<link rel="apple-touch-icon" sizes="180x180" href="https://em-content.zobj.net/thumbs/120/apple/325/fox-face_1f98a.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#FF6600"> 
''', unsafe_allow_html=True)

st.title("🦊 KFB2")

def get_client():
    # 1. VERSUCH
    if 'gcp_service_account' in st.secrets:
        try:
            service_account_info = json.loads(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Retry-Logik für maximale Stabilität
            retry_options = types.HttpRetryOptions(
                initial_delay=2.0,
                attempts=6,
                exp_base=2.0,
                max_delay=30.0,
                http_status_codes=[429, 500, 502, 503, 504]
            )
            
            return genai.Client(
                vertexai=True, 
                project=service_account_info["project_id"], 
                location="europe-west3", 
                credentials=credentials,
                http_options=types.HttpOptions(retry_options=retry_options, timeout=300.0)
            )
        except Exception as e:
            st.warning(f"API fehlgeschlagen, versuche Fallback... ({e})")

    # 2. VERSUCH: STANDARD API KEY (Backup-Schiene)
    if 'gemini_key' in st.secrets:
        return genai.Client(api_key=st.secrets["gemini_key"])
        
    # Wenn beides fehlt:
    st.error("🚨 Keine Zugangsdaten gefunden! Bitte gcp_service_account oder gemini_key in den Secrets hinterlegen.")
    st.stop()

# Client initialisieren
client = get_client()

# --- 3. SESSION STATE (DAS CHAT-GEDÄCHTNIS) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    pdfs = st.file_uploader("PDF-Skripte hochladen", type=["pdf"], accept_multiple_files=True)
    if pdfs:
       st.success(f"{len(pdfs)} Skripte geladen.")
    st.divider()
    if st.button("🗑️ Chat-Verlauf löschen", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.info("model: Gemini 3.1 Pro Preview (mit Retry & Memory)")

# --- 5. DER MASTER-SOLVER (LOGIK) ---
def solve_everything(image, pdf_files, user_input):
    try:
        # DEIN ORIGINAL SYSTEM PROMPT
        sys_instr = """Du bist ein präziser Assistent für Modul 31031 
(Internes Rechnungswesen, FernUniversität Hagen).

PRIORITÄT 1 – DOKUMENTKONTEXT:
Wenn die relevante Information in den Workspace-Dokumenten 
vorhanden ist, beantworte ausschließlich darauf basierend.

PRIORITÄT 2 – FACHWISSEN MIT KENNZEICHNUNG:
Wenn der Dokumentkontext fehlt oder unvollständig ist, 
nutze dein Wissen zu Modul 31031 – kennzeichne diese 
Stellen mit [Fachwissen].

ABSOLUTES VERBOT:
Erfinde niemals fehlende Werte (z. B. fixe Kosten, 
Bestandswerte, Mengenvorgaben). Wenn Werte in der 
Aufgabe fehlen, weise explizit darauf hin und frage 
nach – rechne NICHT mit angenommenen Beispielwerten.

LÖSUNGSPROZESS:
1. Aufgabe analysieren – alle gegebenen Werte auflisten
2. Fehlende Werte sofort benennen – nicht ergänzen
3. Methode aus Modul 31031 anwenden
4. Schritt für Schritt rechnen
5. Ergebnis klar ausgeben

BEI MULTIPLE-CHOICE / WAHR-FALSCH (Prüfungsprotokoll):
Bewerte jede Option zwingend einzeln im folgenden Format:

Option [Buchstabe]:
1. Anomalie-Check: Fällt diese Aussage unter eine 
   bekannte FernUni-Hagen-Besonderheit? Ja/Nein.
2. Behauptung: Was behauptet die Option konkret?
3. Fakt laut Skript/Modul: Was ist die korrekte Aussage?
4. Abgleich: Stimmt Behauptung mit Fakt überein? Ja/Nein.
5. Bewertung: Wahr / Falsch
6. Begründung: Ein Satz.

Vollständigkeitspflicht: Alle Optionen müssen geprüft 
werden – auch wenn eine offensichtlich richtige Option 
bereits gefunden wurde.
Reduziere das Ergebnis NIEMALS nachträglich auf eine 
einzige Option, wenn mehrere korrekt sind.

AUSGABEFORMAT:
Aufgabe [Nr.]: [Ergebnis]
Begründung: [Ein Satz auf Basis der FernUni-Methode]

FORMAT: Deutsch, fachlich sauber, Schritt für Schritt."""

    # Multimodaler Input
        parts = []
        if pdf_files:
            for pdf in pdf_files:
                # Wir lesen die PDF-Daten einmal ein
                pdf_data = pdf.read()
                parts.append(types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"))
                # Zeiger zurücksetzen, falls die Funktion mehrfach aufgerufen wird
                pdf.seek(0)
        
        # Bildbytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        parts.append(types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/jpeg"))	
		
        parts.append("Löse ALLE Aufgaben auf dem Bild unter strikter Einhaltung deines Lösungsprozesses")
        

        # Historie hinzufügen für das "Gedächtnis"
        for m in st.session_state.messages:
            parts.append(f"{m['role']}: {m['content']}")
            
        # Neue Nachricht
        parts.append(f"user: {user_input}")

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=sys_instr,
                temperature=0,
                max_output_tokens=15000,
            )
        )
        return response.text
    except Exception as e:
        return f"Fehler: {str(e)}"

# --- 6. UI LAYOUT ---
col1, col2 = st.columns([1, 1.2])

with col1:
    uploaded_file = st.file_uploader("Klausurblatt hochladen...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        if "rot" not in st.session_state: st.session_state.rot = 0
        if st.button("🔄 Bild drehen"):
            st.session_state.rot = (st.session_state.rot + 90) % 360
            st.rerun()
        img = img.rotate(-st.session_state.rot, expand=True)
        st.image(img, width="stretch")

with col2:
    # Chat History anzeigen
    st.subheader("Analyse & Chat")
    chat_container = st.container(height=600)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# --- 7. CHAT INPUT (AM UNTEREN RAND) ---
if prompt := st.chat_input("Löse die Aufgaben oder gib mir eine Korrektur-Anweisung..."):
    if not uploaded_file:
        st.warning("Bitte lade zuerst ein Klausurblatt hoch!")
    else:
        # User Nachricht anzeigen
        st.session_state.messages.append({"role": "user", "content": prompt})
        with col2: # In der rechten Spalte anzeigen
             with chat_container:
                 with st.chat_message("user"):
                     st.markdown(prompt)
        
        with st.chat_message("assistant"):
                with st.spinner("Gemini löst..."):
                    answer = solve_everything(img, pdfs, prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
