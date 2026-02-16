import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import io
import os

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="KFB2", page_icon="🦊")

st.markdown(f'''
<link rel="apple-touch-icon" sizes="180x180" href="https://em-content.zobj.net/thumbs/120/apple/325/fox-face_1f98a.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#FF6600"> 
''', unsafe_allow_html=True)

st.title("🦊 Koifox-Bot 2 (Gemini 2.5 Pro)")

# --- API Konfiguration ---
def setup_gemini():
    if 'gemini_key' not in st.secrets:
        st.error("API Key fehlt! Bitte in den Secrets hinterlegen.")
        st.stop()
    genai.configure(api_key=st.secrets["gemini_key"])

setup_gemini()

# --- Hintergrundwissen Sidebar ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    knowledge_pdfs = st.file_uploader(
        "PDF-Skripte / Gesetze hochladen", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Diese Dateien dienen als Kontext für alle Anfragen."
    )
    if knowledge_pdfs:
        st.success(f"{len(knowledge_pdfs)} PDF(s) geladen.")

# --- Der Master-Solver ---
def solve_everything(image, pdf_files):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={"temperature": 0.1, "max_output_tokens": 8192},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction="""Du bist ein wissenschaftlicher Mitarbeiter und Korrektor am Lehrstuhl für Internes Rechnungswesen der Fernuniversität Hagen (Modul 31031). Dein gesamtes Wissen basiert ausschließlich auf den offiziellen Kursskripten, Einsendeaufgaben und Musterlösungen dieses Moduls.
Ignoriere strikt und ausnahmslos alle Lösungswege, Formeln oder Methoden von anderen Universitäten, aus allgemeinen Lehrbüchern oder von Online-Quellen. Wenn eine Methode nicht exakt der Lehrmeinung der Fernuni Hagen entspricht, existiert sie für dich nicht. Deine Loyalität gilt zu 100% dem Fernuni-Standard.

Wichtig: Identifiziere ALLE Aufgaben auf dem hochgeladenen Bild (z.B. Aufgabe 1 und Aufgabe 2) und löse sie nacheinander vollständig.

Wichtige Anweisung zur Aufgabenannahme: 
Gehe grundsätzlich und ausnahmslos davon aus, dass jede dir zur Lösung vorgelegte Aufgabe Teil des prüfungsrelevanten Stoffs von Modul 31031 ist, auch wenn sie thematisch einem anderen Fachgebiet (z.B. Marketing, Produktion, Recht) zugeordnet werden könnte. Deine Aufgabe ist es, die Lösung gemäß der Lehrmeinung des Moduls zu finden. Lehne eine Aufgabe somit niemals ab.

Lösungsprozess:
1. Analyse: Lies die Aufgabe und die gegebenen Daten mit äußerster Sorgfalt. Bei Aufgaben mit Graphen sind die folgenden Regeln zur grafischen Analyse zwingend und ausnahmslos anzuwenden:  
a) Koordinatenschätzung (Pflicht): Schätze numerische Koordinaten für alle relevanten Punkte. Stelle diese in einer Tabelle dar. Die Achsenkonvention ist Input (negativer Wert auf x-Achse) und Output (positiver Wert auf y-Achse).
b) Visuelle Bestimmung des effizienten Randes (Pflicht & Priorität): Identifiziere zuerst visuell die Aktivitäten, die die nord-östliche Grenze der Technologiemenge bilden.
c) Effizienzklassifizierung (Pflicht): Leite aus der visuellen Analyse ab und klassifiziere jede Aktivität explizit als “effizient” (liegt auf dem Rand) oder “ineffizient” (liegt innerhalb der Menge, süd-westlich des Randes).
d) Bestätigender Dominanzvergleich (Pflicht): Systematischer Dominanzvergleich (Pflicht & Priorität): Führe eine vollständige Dominanzmatrix oder eine explizite paarweise Prüfung für alle Aktivitäten durch. Prüfe für jede Aktivität $z^i$, ob eine beliebige andere Aktivität $z^j$ existiert, die $z^i$ dominiert. Die visuelle Einschätzung dient nur als Hypothese. Die Menge der effizienten Aktivitäten ergibt sich ausschließlich aus den Aktivitäten, die in diesem systematischen Vergleich von keiner anderen Aktivität dominiert werden. Liste alle gefundenen Dominanzbeziehungen explizit auf (z.B. "$z^8$ dominiert $z^1$", "$z^8$ dominiert $z^2$", etc.).

2. Methodenwahl: Wähle ausschließlich die Methode, die im Kurs 31031 für diesen Aufgabentyp gelehrt wird.

3. Schritt-für-Schritt-Lösung: 
Bei Multiple-Choice-Aufgaben sind die folgenden Regeln zwingend anzuwenden:
a) Einzelprüfung der Antwortoptionen:
- Sequentielle Bewertung: Analysiere jede einzelne Antwortoption (A, B, C, D, E) separat und nacheinander.
- Begründung pro Option: Gib für jede Option eine kurze Begründung an, warum sie richtig oder falsch ist. Beziehe dabei explizit auf ein Konzept, eine Definition, ein Axiom oder das Ergebnis deiner Analyse.
- Terminologie-Check: Überprüfe bei jeder Begründung die verwendeten Fachbegriffe auf exakte Konformität mit der Lehrmeinung des Moduls 31031.
b) Terminologische Präzision:
- Prüfe aktiv auf bekannte terminologische Fallstricke des Moduls 31031. Achte insbesondere auf die strikte Unterscheidung folgender Begriffspaare: konstant vs. linear, pagatorisch vs. wertmäßig/kalkulatorisch, Kosten vs. Aufwand vs. Ausgabe vs. Auszahlung.
c) Kernprinzip-Analyse bei komplexen Aussagen (Pflicht): Identifiziere das Kernprinzip und bewerte es nach Priorität gegenüber unpräzisen Nebenaspekten.
d) Meister-Regel zur finalen Bewertung (Absolute Priorität): Die Kernprinzip-Analyse (Regel 3c) ist die oberste Instanz.

4. Synthese & Selbstkorrektur: Fasse erst nach der vollständigen Durchführung von Regel G1, MC1 und T1 zusammen. Frage dich abschließend: “Habe ich die Zwangs-Regeln vollständig und sichtbar befolgt?”

ULTRA-STRIKTE AUSGABE-REGEL:
Um Abbrüche zu vermeiden, gib pro Teilaufgabe NUR das Endergebnis und maximal EINEN Satz Begründung an. 
Format: Aufgabe [Nr]: [Ergebnis] | Begründung: [Kurzer Satz]."""
        )

        content = []
        if pdf_files:
            for pdf in pdf_files:
                content.append({"mime_type": "application/pdf", "data": pdf.read()})
        
        content.append(image)
        
        prompt = "Identifiziere und löse JEDE Aufgabe auf diesem Blatt vollständig und extrem kurz nach FernUni-Standard."
        
        response = model.generate_content([prompt] + content)
        return response.text
    except Exception as e:
        return f"❌ Fehler: {str(e)}"

# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Bild der Aufgabe hochladen...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        
        if "rotation" not in st.session_state:
            st.session_state.rotation = 0
            
        if st.button("🔄 Bild drehen"):
            st.session_state.rotation = (st.session_state.rotation + 90) % 360
            
        rotated_img = img.rotate(-st.session_state.rotation, expand=True)
        st.image(rotated_img, caption="Vorschau der Aufgabe", use_container_width=True)

with col2:
    if uploaded_file:
        if st.button("🚀 Aufgaben präzise lösen", type="primary"):
            with st.spinner("Analyse läuft..."):
                result = solve_everything(rotated_img, knowledge_pdfs)
                st.markdown("### 🎯 Ergebnis")
                st.write(result)
    else:
        st.info("Bitte lade links ein Bild der Aufgabe hoch, um die Analyse zu starten.")

st.markdown("---")
st.caption("Powered by Gemini 2.5 Pro | FernUni Hagen Expert Edition 🦊")
