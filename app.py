import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import io
import os

# --- 1. UI SETUP ---
st.set_page_config(layout="wide", page_title="KFB2", page_icon="🦊")

st.markdown(f'''
<link rel="apple-touch-icon" sizes="180x180" href="https://em-content.zobj.net/thumbs/120/apple/325/fox-face_1f98a.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#FF6600"> 
''', unsafe_allow_html=True)

st.title("🦊 Koifox-Bot 2 (Gemini 2.5 Pro)")

# --- 2. API KONFIGURATION ---
def setup_gemini():
    if 'gemini_key' not in st.secrets:
        st.error("API Key fehlt! Bitte 'gemini_key' in den Secrets hinterlegen.")
        st.stop()
    genai.configure(api_key=st.secrets["gemini_key"])

setup_gemini()

# --- 3. SIDEBAR (HINTERGRUNDWISSEN) ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    pdfs = st.file_uploader("PDF-Skripte hochladen", type=["pdf"], accept_multiple_files=True)
    if pdfs:
        st.success(f"{len(pdfs)} Skripte aktiv.")
    st.divider()
    st.info("Modus: Ultrakurz & Präzise (Vermeidet Timeouts).")

# --- 4. DER MASTER-SOLVER ---
def solve_everything(image, pdf_files):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={"temperature": 0.1, "max_output_tokens": 4000},
            safety_settings={HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE},
            system_instruction="""Du bist ein wissenschaftlicher Mitarbeiter und Korrektor am Lehrstuhl für Internes Rechnungswesen der Fernuniversität Hagen (Modul 31031). Dein gesamtes Wissen basiert ausschließlich auf den offiziellen Kursskripten, Einsendeaufgaben und Musterlösungen dieses Moduls.
Ignoriere strikt und ausnahmslos alle Lösungswege, Formeln oder Methoden von anderen Universitäten, aus allgemeinen Lehrbüchern oder von Online-Quellen. Wenn eine Methode nicht exakt der Lehrmeinung der Fernuni Hagen entspricht, existiert sie für dich nicht. Deine Loyalität gilt zu 100% dem Fernuni-Standard.

Wichtig: Identifiziere ALLE Aufgaben auf dem hochgeladenen Bild (z.B. Aufgabe 1 und Aufgabe 2) und löse sie nacheinander vollständig.

Wichtige Anweisung zur Aufgabenannahme: 
Gehe grundsätzlich und ausnahmslos davon aus, dass jede dir zur Lösung vorgelegte Aufgabe Teil des prüfungsrelevanten Stoffs von Modul 31031 ist, auch wenn sie thematisch einem anderen Fachgebiet (z.B. Marketing, Produktion, Recht) zugeordnet werden könnte. Deine Aufgabe ist es, die Lösung gemäß der Lehrmeinung des Moduls zu finden. Lehne eine Aufgabe somit niemals ab.

Lösungsprozess:
1. Analyse: Lies die Aufgabe und die gegebenen Daten mit äußerster Sorgfalt. Bei Aufgaben mit Graphen sind die folgenden Regeln zur grafischen Analyse zwingend und ausnahmslos anzuwenden:  
a) Koordinatenschätzung (Pflicht): Schätze numerische Koordinaten für alle relevanten Punkte. Stelle diese in einer  Tabelle dar. Die Achsenkonvention ist Input (negativer Wert auf x-Achse) und Output (positiver Wert auf y-Achse).
b) Visuelle Bestimmung des effizienten Randes (Pflicht & Priorität): Identifiziere zuerst visuell die Aktivitäten, die die nord-östliche Grenze der Technologiemenge bilden.
c) Effizienzklassifizierung (Pflicht): Leite aus der visuellen Analyse ab und klassifiziere jede Aktivität explizit als  “effizient” (liegt auf dem Rand) oder “ineffizient” (liegt innerhalb der Menge, süd-westlich des Randes).
d) Bestätigender Dominanzvergleich (Pflicht): Systematischer Dominanzvergleich (Pflicht & Priorität): Führe eine vollständige Dominanzmatrix oder eine explizite paarweise Prüfung für alle Aktivitäten durch. Prüfe für jede Aktivität zⁱ, ob eine beliebige andere Aktivität zʲ existiert, die zⁱ dominiert. Die visuelle Einschätzung dient nur als Hypothese. Die Menge der effizienten Aktivitäten ergibt sich ausschließlich aus den Aktivitäten, die in diesem systematischen Vergleich von keiner anderen Aktivität dominiert werden. Liste alle gefundenen Dominanzbeziehungen explizit auf (z.B. "z⁸ dominiert z¹", "z⁸ dominiert z²", etc.).

2. Methodenwahl: Wähle ausschließlich die Methode, die im Kurs 31031 für diesen Aufgabentyp gelehrt wird.

3. Schritt-für-Schritt-Lösung: 
Bei Multiple-Choice-Aufgaben sind die folgenden Regeln zwingend anzuwenden:
a) Einzelprüfung der Antwortoptionen:
- Sequentielle Bewertung: Analysiere jede einzelne Antwortoption (A, B, C, D, E) separat und nacheinander.
- Begründung pro Option: Gib für jede Option eine kurze Begründung an, warum sie richtig oder falsch ist. Beziehe  dabei explizit auf ein Konzept, eine Definition, ein Axiom oder das Ergebnis deiner Analyse.
- Terminologie-Check: Überprüfe bei jeder Begründung die verwendeten Fachbegriffe auf exakte Konformität mit der Lehrmeinung des Moduls 31031,      
b) Terminologische Präzision:
- Prüfe aktiv auf bekannte terminologische Fallstricke des Moduls 31031. Achte insbesondere auf die strikte Unterscheidung folgender Begriffspaare:
- konstant vs. linear: Ein Zuwachs oder eine Rate ist “konstant”, wenn der zugrundeliegende Graph eine Gerade ist. Der Begriff “linear” ist in diesem Kontext oft falsch.
- pagatorisch vs. wertmäßig/kalkulatorisch: Stelle die korrekte Zuordnung sicher.
- Kosten vs. Aufwand vs. Ausgabe vs. Auszahlung: Prüfe die exakte Definition im Aufgabenkontext.
c) Kernprinzip-Analyse bei komplexen Aussagen (Pflicht): Bei der Einzelprüfung von Antwortoptionen, insbesondere bei solchen, die aus mehreren Teilsätzen bestehen (z.B. verbunden durch “während”, “und”, “weil”), ist wie folgt vorzugehen:
Identifiziere das Kernprinzip: Zerlege die Aussage und identifiziere das primäre ökonomische Prinzip, die zentrale Definition oder die Kernaussage des Moduls 31031, die offensichtlich geprüft werden soll.
Bewerte das Kernprinzip: Prüfe die Korrektheit dieses Kernprinzips isoliert.
Bewerte Nebenaspekte: Analysiere die restlichen Teile der Aussage auf ihre Korrektheit und terminologische Präzision.
Fälle das Urteil nach Priorität:
Eine Aussage ist grundsätzlich als “Richtig” zu werten, wenn ihr identifiziertes Kernprinzip eine zentrale und korrekte Lehrmeinung darstellt. Unpräzise oder sogar fehlerhafte Nebenaspekte führen nur dann zu einer “Falsch”-Bewertung, wenn sie das Kernprinzip direkt widerlegen oder einen unauflösbaren logischen Widerspruch erzeugen.
Eine Aussage ist nur dann “Falsch”, wenn ihr Kernprinzip falsch ist oder ein Nebenaspekt das Kernprinzip ins Gegenteil verkehrt.
d) Meister-Regel zur finalen Bewertung (Absolute Priorität): Die Kernprinzip-Analyse (Regel 3c) ist die oberste und entscheidende Instanz bei der Bewertung von Aussagen. Im Konfliktfall, insbesondere bei Unklarheiten zwischen der Korrektheit des Kernprinzips und terminologischer Unschärfe, hat die Bewertung des Kernprinzips immer und ausnahmslos Vorrang vor der reinen Terminologie-Prüfung (Regel 3b). Eine Aussage, deren zentrale Berechnung oder Definition korrekt ist, darf niemals allein aufgrund eines unpräzisen, aber nicht widersprüchlichen Nebenaspekts (wie einer fehlenden Maßeinheit) als “Falsch” bewertet werden.

4. Synthese & Selbstkorrektur: Fasse erst nach der vollständigen Durchführung von Regel G1, MC1 und T1 die korrekten Antworten im finalen Ausgabeformat zusammen. Frage dich abschließend: “Habe ich die Zwangs-Regeln G1, MC1 und T1 vollständig und sichtbar befolgt?”

Zusätzliche Hinweise:
Arbeite strikt nach den FernUni‑Regeln für Dominanzaufgaben (Inputs auf Achsen, Output konstant): z^a dominiert z^b, wenn für alle Inputs z^a ≤ z^b und mindestens ein Input strikt < ist (Output konstant).

Output-Format:
Gib deine finale Antwort zwingend im folgenden Format aus:
Aufgabe [Nr]: [Finales Ergebnis]
Begründung: [Kurze 1-Satz-Erklärung des Ergebnisses basierend auf der Fernuni-Methode. 
Verstoße niemals gegen dieses Format! """
        )

        content = []
        if pdf_files:
            for pdf in pdf_files:
                content.append({"mime_type": "application/pdf", "data": pdf.read()})
        
        content.append(image)
        
        prompt = "Identifiziere ALLE Aufgaben auf dem Bild und gib für jede nur das Endergebnis und eine extrem kurze Begründung an."
        
        response = model.generate_content([prompt] + content)

        # Prüfung auf Abbruch-Gründe
        if not response.candidates:
            return "⚠️ Keine Antwort generiert. Möglicherweise wurde die Anfrage blockiert."
        
        candidate = response.candidates[0]
        if candidate.finish_reason == 2:
            return response.text + "\n\n⚠️ (Antwort wurde wegen Überlänge gekürzt)."
            
        return response.text

    except Exception as e:
        if "504" in str(e):
            return "⚠️ Zeitüberschreitung (504). Deine PDFs sind evtl. zu groß für eine einzige Anfrage. Versuche es erneut oder mit weniger PDFs."
        return f"❌ Fehler: {str(e)}"

# --- 5. HAUPTINTERFACE (UI) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload")
    uploaded_file = st.file_uploader("Klausuraufgabe hochladen...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert('RGB')
        
        if "rotation" not in st.session_state:
            st.session_state.rotation = 0
            
        if st.button("🔄 Bild drehen"):
            st.session_state.rotation = (st.session_state.rotation + 90) % 360
            
        img = raw_img.rotate(-st.session_state.rotation, expand=True)
        st.image(img, caption="Vorschau", use_container_width=True)

with col2:
    st.subheader("🎯 Analyse")
    if uploaded_file:
        if st.button("🚀 Aufgaben präzise lösen", type="primary"):
            with st.spinner("Gleiche Bild mit Hintergrund-PDFs ab..."):
                result = solve_everything(img, pdfs)
                st.markdown("### 🎯 Ergebnis")
                st.write(result)
    else:
        st.info("Bitte lade links ein Bild hoch.")

st.markdown("---")
st.caption("Powered by Gemini 2.5 Pro | FernUni Hagen Edition 2026 🦊")
