# streamlit_app_local_ollama.py
import streamlit as st
from pathlib import Path
import json, os, tempfile, time, difflib, re
import importlib

# Import your agent module (must be in same folder or on PYTHONPATH)
# Replace with your module name if different (you used langchain_pandas_agent earlier)
import langchain_pandas_agent as agent_mod

# ---------- Helpers: audio normalization & multi-backend recorder ----------
import base64
import io

def _normalize_audio_bytes(raw):
    """
    Accept many recorder return types and return raw WAV bytes (or None).
    Handles:
      - bytes -> returned directly
      - base64-encoded string -> decode
      - dicts like {'data': ..., 'blob': ...} -> find likely field
      - file-like objects with .read()
      - streamlit UploadedFile (has .getvalue())
      - list/tuple -> normalize first element
    """
    if raw is None:
        return None

    # list/tuple -> first element
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        return _normalize_audio_bytes(raw[0])

    # bytes
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)

    # UploadedFile from Streamlit (has getvalue)
    if hasattr(raw, "getvalue"):
        try:
            return raw.getvalue()
        except Exception:
            pass

    # file-like object
    if hasattr(raw, "read"):
        try:
            b = raw.read()
            if isinstance(b, (bytes, bytearray)):
                return bytes(b)
        except Exception:
            pass

    # dict -> try common keys
    if isinstance(raw, dict):
        # common keys that some components use
        for key in ("data", "audio", "blob", "wav", "file", "bytes"):
            if key in raw and raw[key] is not None:
                return _normalize_audio_bytes(raw[key])
        # also try values
        for v in raw.values():
            out = _normalize_audio_bytes(v)
            if out:
                return out

    # str -> maybe base64 or data URI
    if isinstance(raw, str):
        s = raw.strip()
        # data URI like "data:audio/wav;base64,...."
        if s.startswith("data:"):
            try:
                b64 = s.split(",",1)[1]
                return base64.b64decode(b64)
            except Exception:
                pass
        # plain base64
        try:
            return base64.b64decode(s)
        except Exception:
            pass

    # fallback: can't normalize
    return None

def get_audio_from_any_recorder():
    """
    Try several browser recorder components. Return (backend_name, raw_audio, errors_list).
    raw_audio is the un-normalized value returned by the component.
    """
    recorder_backend = None
    audio_raw = None
    errors = []

    # 1) try streamlit-audiorecorder
    try:
        from streamlit_audiorecorder import audiorecorder  # pip install streamlit-audiorecorder
        recorder_backend = "streamlit-audiorecorder"
        audio_raw = audiorecorder("Record command", "Recording...")
    except Exception as e:
        errors.append(("streamlit-audiorecorder", repr(e)))
        audio_raw = None

    # 2) try streamlit_mic_recorder (user indicated usage)
    if audio_raw is None:
        try:
            from streamlit_mic_recorder import mic_recorder  # pip install streamlit-mic-recorder
            recorder_backend = "streamlit-mic-recorder"
            audio_raw = mic_recorder()
        except Exception as e:
            errors.append(("streamlit-mic-recorder", repr(e)))
            audio_raw = None

    # 3) try st-audiorec
    if audio_raw is None:
        try:
            from st_audiorec import st_audiorec  # pip install st-audiorec
            recorder_backend = "st-audiorec"
            audio_raw = st_audiorec()
        except Exception as e:
            errors.append(("st-audiorec", repr(e)))
            audio_raw = None

    # 4) fallback: file uploader
    if audio_raw is None:
        recorder_backend = "file-uploader"
        uploaded = st.file_uploader("Upload WAV/MP3 (fallback)", type=["wav","mp3","m4a","ogg"])
        if uploaded is not None:
            audio_raw = uploaded  # UploadedFile; normalize will handle .getvalue()

    return recorder_backend, audio_raw, errors

# ---------- Ollama helpers ----------
def init_ollama():
    try:
        try:
            from langchain_community.llms import Ollama
        except Exception:
            from langchain_ollama import OllamaLLM as Ollama
        llm = Ollama(model="llama3")
        return llm, None
    except Exception as e:
        return None, str(e)

def call_ollama_for_text(llm, prompt):
    """Call Ollama/Llama wrapper with a plain string prompt. Return text or (None, err)."""
    if llm is None:
        return None, "No Ollama LLM available"
    try:
        # try several common call styles
        if hasattr(llm, "invoke"):
            out = llm.invoke(prompt)
            return str(out), None
        if hasattr(llm, "predict"):
            out = llm.predict(prompt)
            return str(out), None
        if hasattr(llm, "generate"):
            out = llm.generate([{"role":"user","content":prompt}])
            try:
                return out.generations[0][0].text, None
            except Exception:
                return str(out), None
        # last fallback: call like a function
        out = llm(prompt)
        return str(out), None
    except Exception as e:
        return None, str(e)

def build_json_tool_prompt(user_sentence, csv_path):
    prompt = f"""
Translate the user's natural-language request into exactly one JSON object (and nothing else).
The JSON schema is: {{"tool":"<tool_name>", "input": <input_object>}}

Allowed tools: describe, describe_column, missing, duplicates, outliers, correlations, scatter, run_report

Examples (use path exactly as provided):
{{"tool":"describe","input":{{"path":"{csv_path}"}}}}
{{"tool":"describe_column","input":{{"path":"{csv_path}","col":"Age"}}}}
{{"tool":"outliers","input":{{"path":"{csv_path}","col":"Weight (kg)"}}}}
{{"tool":"correlations","input":{{"path":"{csv_path}","cols":["Age","Weight (kg)"]}}}}
Use full column names if possible; if not, substring matches are acceptable (e.g., 'weight' -> 'Weight (kg)').

Produce exactly one JSON object — nothing else. User request: {user_sentence}
"""
    return prompt

def parse_model_json(raw_text):
    # 1) direct
    try:
        return json.loads(raw_text), None
    except Exception:
        pass
    s = (raw_text or "").strip()
    # 2) strip wrapping quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        try:
            return json.loads(s[1:-1]), None
        except Exception:
            pass
    # 3) replace single quotes with double quotes
    try:
        return json.loads(s.replace("'", '"')), None
    except Exception:
        pass
    # 4) find JSON substring
    m = re.search(r"\{.*\}", raw_text or "", flags=re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate), None
        except Exception:
            try:
                return json.loads(candidate.replace("'", '"')), None
            except Exception:
                pass
    return None, "Could not parse JSON"

# ---------- Execute parsed action safely ----------
TOOL_MAP = {
    "describe": agent_mod.safe_tool_wrapper(agent_mod.tool_describe),
    "describe_column": agent_mod.safe_tool_wrapper(agent_mod.tool_describe_column),
    "missing": agent_mod.safe_tool_wrapper(agent_mod.tool_missing),
    "duplicates": agent_mod.safe_tool_wrapper(agent_mod.tool_duplicates),
    "outliers": agent_mod.safe_tool_wrapper(agent_mod.tool_outliers),
    "correlations": agent_mod.safe_tool_wrapper(agent_mod.tool_correlations),
    "scatter": agent_mod.safe_tool_wrapper(agent_mod.tool_scatter),
    "run_report": agent_mod.safe_tool_wrapper(agent_mod.tool_run_report),
}

def execute_parsed_action(parsed_json, csv_path):
    if not isinstance(parsed_json, dict):
        return {"message":"Parsed result not dict","files":[], "meta":{}}
    tool = parsed_json.get("tool") or parsed_json.get("action") or parsed_json.get("name")
    inp = parsed_json.get("input") or parsed_json.get("inputs") or parsed_json.get("args") or parsed_json.get("data") or {}
    if not tool:
        return {"message":"No 'tool' in parsed JSON","files":[], "meta":{}}
    tool = tool.strip()
    func = TOOL_MAP.get(tool)
    if not func:
        return {"message":f"Unknown tool: {tool}","files":[], "meta":{}}
    try:
        res = func(inp)
        return res if isinstance(res, dict) else {"message": str(res), "files": [], "meta": {}}
    except Exception as e:
        return {"message": f"Tool execution error: {e}", "files": [], "meta": {}}

# ---------- Local transcription attempts ----------
# --- Local transcription attempts (always translate to English) ---
def transcribe_local(audio_path):
    """
    Transcribes and automatically translates speech to English using a local Whisper or Faster-Whisper model.
    Works fully offline.
    """
    # Try faster-whisper first (recommended for local/offline)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cpu")  # or "medium" for better accuracy
        segments, info = model.transcribe(str(audio_path), task="translate", language=None)
        text = " ".join([seg.text for seg in segments]).strip()
        if text:
            return text, None
    except Exception as e:
        print("faster-whisper error:", e)

    # Fallback to openai-whisper if installed
    try:
        import whisper
        model = whisper.load_model("small")
        result = model.transcribe(str(audio_path), task="translate")  # ensures English output
        text = result.get("text", "").strip()
        if text:
            return text, None
    except Exception as e:
        return None, f"No local STT available: {e}"

    return None, "Transcription failed."


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Local Audio → Ollama → Agent", layout="wide")
st.title("Local Audio → Ollama → Agent")

st.sidebar.header("Settings")
csv_path = st.sidebar.text_input("Path to CSV file", value="")
uploaded = st.sidebar.file_uploader("Or upload CSV here", type=["csv"])
use_ollama = st.sidebar.checkbox("Use Ollama to generate tool JSON", value=True)
show_debug = st.sidebar.checkbox("Show debug info (model raw/parsed/import errors)", value=False)

if uploaded is not None:
    p = Path("uploaded_data.csv")
    p.write_bytes(uploaded.getbuffer())
    csv_path = str(p)

if not csv_path:
    st.warning("Please provide CSV path or upload CSV in sidebar.")
    st.stop()

# init Ollama once (store init status)
if 'ollama' not in st.session_state:
    llm, err = init_ollama()
    st.session_state['ollama'] = llm
    st.session_state['ollama_err'] = err

llm = st.session_state['ollama']
if st.session_state.get('ollama_err'):
    st.info(f"Ollama init error: {st.session_state['ollama_err']}. App will use deterministic fallback if Ollama unavailable.")

st.write(f"Using CSV: `{csv_path}`")
st.markdown("### Record a short voice command")
st.write("Press record, speak one command (e.g., 'Describe Age', 'outliers for weight', 'correlation for age and weight'). Press Stop when done.")

# Use the robust recorder that tries multiple backends
backend, raw_audio, rec_errors = get_audio_from_any_recorder()
st.sidebar.write("Audio backend:", backend)
if show_debug and rec_errors:
    st.sidebar.write("Recorder import attempts and errors (first few):")
    for name, err in rec_errors[:6]:
        st.sidebar.write(name, "->", err)

# Normalize
audio_bytes = _normalize_audio_bytes(raw_audio)
if audio_bytes:
    # Save to temp wav (so transcription libraries can read)
    tmp = Path(tempfile.gettempdir()) / f"st_cmd_{int(time.time())}.wav"
    try:
        tmp.write_bytes(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        st.success(f"Saved recording to {tmp}")
    except Exception as e:
        st.error(f"Failed to save normalized audio bytes: {e}")
        audio_bytes = None

if audio_bytes:
    # Transcribe locally
    transcript, terr = transcribe_local(tmp)
    if terr:
        st.error(f"Transcription failed: {terr}")
        manual = st.text_input("Type your command instead (fallback):")
        if manual:
            transcript = manual
    else:
        st.write("Transcription:", transcript)

    if not transcript:
        st.info("No transcription available. Provide text input or try re-recording.")
    else:
        # If using Ollama: build strict JSON prompt and call Ollama
        llm_debug = {}
        parsed = None
        if use_ollama and llm is not None:
            prompt = build_json_tool_prompt(transcript, csv_path)
            raw, err = call_ollama_for_text(llm, prompt)
            llm_debug['raw'] = raw
            llm_debug['err'] = err
            if raw:
                parsed, perr = parse_model_json(raw)
                llm_debug['parsed'] = parsed
                llm_debug['parse_err'] = perr

            # If parsed JSON and valid, execute; otherwise fallback to deterministic
            if parsed and isinstance(parsed, dict) and parsed.get("tool"):
                result = execute_parsed_action(parsed, csv_path)
            else:
                df = agent_mod._read_df_from_arg({"path": csv_path})
                result = agent_mod.rule_based_executor(transcript, csv_path, df)
        else:
            df = agent_mod._read_df_from_arg({"path": csv_path})
            result = agent_mod.rule_based_executor(transcript, csv_path, df)
            llm_debug = {}

        # Show debug if requested
        if show_debug:
            st.subheader("LLM / debug")
            st.json(llm_debug)

        # Display result
        st.subheader("Result")
        if isinstance(result, dict):
            st.success(result.get("message", "Done"))
            files = result.get("files") or []
            if files:
                st.write("Files:")
                for f in files:
                    try:
                        if isinstance(f, (list, tuple)):
                            # flatten odd shapes
                            for ff in f:
                                st.write(ff)
                        elif f.lower().endswith((".png", ".jpg", ".jpeg")):
                            st.image(f, use_column_width=True)
                        else:
                            st.write(f)
                    except Exception:
                        st.write(f)
        else:
            st.write(result)

else:
    st.info("No recording yet — press the record button to start.")
