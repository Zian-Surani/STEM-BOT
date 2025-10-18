# app.py
# STEM Bot — Math (SymPy) + Science/Social (HF Inference API via Hugging Face)
# NOTE: Hard-coding API keys is unsafe. Prefer Streamlit secrets or env vars.

import os, io, json
from typing import List, Optional, Tuple, Dict

import streamlit as st
from PIL import Image
import numpy as np
from pdfminer.high_level import extract_text
import sympy as sp
import sympy.stats as sps
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
import regex as re

# ---------- SymPy parsing with implicit multiplication ----------
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    function_exponentiation,
)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    function_exponentiation,
)

# ---------- Config ----------
# !! WARNING: Hard-coding tokens is insecure. You asked to inline it explicitly:
HF_TOKEN = "YOUR TOKEN"

HF_MODEL_SCI   = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_SOCIO = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_FALLBACK = "HuggingFaceH4/zephyr-7b-beta"  # public conversational fallback

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
MAX_CTX_SNIPPET = 1600
OCR_CACHE_VERSION = "v3"
PDF_CACHE_VERSION = "v2"

# ---------- Page ----------
st.set_page_config(page_title="STEM Bot", layout="wide")

# ---------- OCR backends ----------
@st.cache_resource(show_spinner=False)
def get_ocr(version: str = OCR_CACHE_VERSION):
    fns, names = [], []

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        paddle = PaddleOCR(use_angle_cls=True, lang="en")
        def paddle_run(img: Image.Image) -> str:
            arr = np.array(img.convert("RGB"))
            res = paddle.ocr(arr, cls=True)
            lines = []
            for page in res:
                for _, (txt, conf) in page:
                    if conf is None or conf > 0.5:
                        lines.append(txt)
            return "\n".join(lines)
        fns.append(paddle_run); names.append("PaddleOCR")
    except Exception:
        pass

    # EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        def easy_run(img: Image.Image) -> str:
            arr = np.array(img.convert("RGB"))
            res = reader.readtext(arr, detail=0, paragraph=True)
            return "\n".join([r for r in res if isinstance(r, str) and r.strip()])
        fns.append(easy_run); names.append("EasyOCR")
    except Exception:
        pass

    # Tesseract
    try:
        import pytesseract
        tess_path = os.environ.get("TESSERACT_PATH")
        if tess_path:
            pytesseract.pytesseract.tesseract_cmd = tess_path
        def tess_run(img: Image.Image) -> str:
            return pytesseract.image_to_string(img.convert("RGB"))
        fns.append(tess_run); names.append("Tesseract")
    except Exception:
        pass

    return fns, names

def run_ocr_chain(img: Image.Image) -> Tuple[str, str]:
    fns, names = get_ocr()
    last_err = None
    for fn, name in zip(fns, names):
        try:
            txt = fn(img)
            if txt and txt.strip():
                return txt, name
        except PermissionError as e:
            last_err = e; continue
        except Exception as e:
            last_err = e; continue
    if last_err:
        raise last_err
    return "", "None"

# ---------- PDF text extraction ----------
@st.cache_resource(show_spinner=False)
def get_pdf_backend(version: str = PDF_CACHE_VERSION):
    def pdfminer_extract(pdf_bytes: bytes) -> str:
        bio = io.BytesIO(pdf_bytes)
        return extract_text(bio) or ""
    try:
        from pypdf import PdfReader
        def pypdf_extract(pdf_bytes: bytes) -> str:
            bio = io.BytesIO(pdf_bytes)
            reader = PdfReader(bio)
            out = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    out.append(t)
            return "\n".join(out)
        return pdfminer_extract, pypdf_extract
    except Exception:
        return pdfminer_extract, None

def extract_pdf_text_inmemory(uploaded_file) -> str:
    pdf_bytes = uploaded_file.getvalue()
    primary, fallback = get_pdf_backend()
    try:
        text = primary(pdf_bytes)
        if text and text.strip():
            return text
    except Exception:
        pass
    if fallback:
        try:
            text = fallback(pdf_bytes)
            if text and text.strip():
                return text
        except Exception:
            pass
    return ""

# ---------- Embeddings & LLM ----------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_ID)

@st.cache_resource(show_spinner=False)
def get_llm_client():
    # Use HF conversational API. This avoids the Together/OpenAI router that causes 401 if misconfigured.
    return InferenceClient(api_key=HF_TOKEN)  # token=HF_TOKEN also works

def _is_permissions_error(msg: str) -> bool:
    s = (msg or "").lower()
    return "403" in s and ("forbidden" in s or "insufficient" in s or "provider" in s)

def _is_auth_error(e: Exception) -> bool:
    s = str(e).lower()
    return "401" in s or "unauthorized" in s or "invalid credentials" in s

def chat_once(model_id: str, messages, temperature=0.18, top_p=0.85, max_tokens=900) -> str:
    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content if (resp and resp.choices) else ""
        return (text or "No answer returned.").strip()

    except HfHubHTTPError as e:
        s = str(e).lower()
        if ("401" in s or "unauthorized" in s or "invalid credentials" in s
            or "403" in s or "forbidden" in s or "insufficient" in s):
            try:
                resp = client.chat.completions.create(
                    model=HF_MODEL_FALLBACK,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content if (resp and resp.choices) else ""
                return (text or "No answer returned.").strip()
            except Exception as e2:
                return f"[LLM error] {e2}"
        return f"[LLM error] {e}"

    except Exception as e:
        try:
            resp = client.chat.completions.create(
                model=HF_MODEL_FALLBACK,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content if (resp and resp.choices) else ""
            return (text or "No answer returned.").strip()
        except Exception as e2:
            return f"[LLM error] {e2}"

# ---------- Dedup / cleanup ----------
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
PARA_SPLIT = re.compile(r"\n{2,}")

def clean_repeats_final(text: str) -> str:
    if not text: return text
    for win in range(400, 40, -20):
        if len(text) >= 2*win and text[-2*win:-win] == text[-win:]:
            while len(text) >= 2*win and text[-2*win:-win] == text[-win:]:
                text = text[:-win]
            break
    paras = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    seen_p, out_p = set(), []
    for p in paras:
        if p not in seen_p:
            out_p.append(p); seen_p.add(p)
    cleaned = []
    for p in out_p:
        sents = [s.strip() for s in SENT_SPLIT.split(p) if s.strip()]
        seen_s, out_s = set(), []
        for s in sents:
            if s not in seen_s:
                out_s.append(s); seen_s.add(s)
        cleaned.append(" ".join(out_s))
    return "\n".join(cleaned).strip()

# ---------- OCR/Math normalization ----------
INT_SIGN_PAT = re.compile(r"∫\s*(.*?)\s*d\s*([a-zA-Z])\b", re.S)
INTEGRAL_TEXT_PAT = re.compile(r"(?:integral\s+of|integrate)\s+(.+)", re.I)

def normalize_ocr_math(q: str) -> str:
    if not q:
        return q
    s = q.strip()
    m = INT_SIGN_PAT.search(s)
    if m:
        integrand, var = m.group(1).strip(), m.group(2)
        integrand = re.sub(rf"(?i)\bd\s*{re.escape(var)}\b", "", integrand).strip()
        s = f"integrate {integrand}"
    s = re.sub(r"(\*\*|\^)\s*$", "", s)
    s = re.sub(r"(?i)\bd\s*[a-z]\b", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _clean_expr_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("×", "*").replace("·", "*").replace("∙", "*")
    s = s.replace("÷", "/").replace("√", "sqrt")
    s = s.replace("^", "**")
    s = re.sub(r"\b(log|ln|sin|cos|tan|sec|csc|cot)\s*([A-Za-z0-9(])", r"\1(\2", s)
    s = re.sub(r"\b(log|ln|sin|cos|tan|sec|csc|cot)\(([^()]*?)(?=\s|$|[+\-*/^=)])", r"\1(\2)", s)
    s = re.sub(r"\blog\s+([A-Za-z0-9_.]+)\b", r"log(\1)", s)
    s = re.sub(r"\bln\s+([A-Za-z0-9_.]+)\b", r"log(\1)", s)
    s = re.sub(r"(?i)\bd\s*[a-z]\b", "", s)
    s = re.sub(r"(\*\*|\^)\s*$", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _expr_from_text(text: str) -> sp.Expr:
    return parse_expr(_clean_expr_text(text), transformations=TRANSFORMS, evaluate=True)

# ---------- Prompts ----------
def system_prompt(subject: str) -> str:
    base = ("You are a careful STEM tutor. Provide accurate, concise answers. "
            "Do not repeat yourself; state each point once, then stop.")
    if subject == "socio":
        base += " Use dates, names, places; be neutral and factual."
    elif subject == "sci":
        base += " Use definitions/laws precisely and include a short rationale."
    return base

def user_prompt(subject: str, mode: str, question: str, options: Optional[List[str]], ctx: List[str]) -> str:
    lines=[]
    if ctx:
        lines.append("Helpful context (optional):")
        for c in ctx: lines.append(f"- {c}")
        lines.append("---")
    lines.append(f"Subject: {subject} | Mode: {mode}")
    if options:
        lines.append("Options:")
        for i,opt in enumerate(options): lines.append(f"{chr(65+i)}. {opt}")
    lines.append("Question:"); lines.append(question.strip() if question else "")
    lines.append("---")
    if mode=="mcq":
        lines.append("Return ONLY: chosen option (A/B/C/...) and one-sentence rationale.")
    elif mode=="short":
        lines.append("Return 4–6 crisp sentences.")
    else:
        lines.append("Return a structured answer (short intro → key points → 1–2 line summary).")
    return "\n".join(lines)

# ---------- Retrieval ----------
def build_faiss_index(texts: List[Dict], embedder) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    vecs = embedder.encode([d["text"] for d in texts], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs.astype("float32"))
    return index, texts

def search_index(index, meta, query: str, embedder, k: int = 5) -> List[Tuple[float, Dict]]:
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D,I = index.search(q,k); out=[]
    for s, idx in zip(D[0], I[0]):
        if 0 <= idx < len(meta): out.append((float(s), meta[idx]))
    return out

OPTION_PREFIX_RE = re.compile(r"^\s*([A-Da-d]|[0-9]+|[-*•])[\).\]:\- ]\s*")
def parse_mcq_options(block: str) -> List[str]:
    if not block: return []
    lines=[ln.strip() for ln in block.splitlines() if ln.strip()]
    out=[]
    for ln in lines:
        cleaned = OPTION_PREFIX_RE.sub("", ln).strip()
        if cleaned: out.append(cleaned)
    return out[:6]

# ---------- Math core ----------
X = sp.symbols("x")
LIMIT_TEXT_PAT = re.compile(r"limit(?:\s+of)?\s+(.*?)\s+as\s+x\s*->\s*([^\s]+)", re.I)

def parse_integral_text(q: str) -> Optional[Tuple[str, str]]:
    m = INT_SIGN_PAT.search(q)
    if m:
        integrand, var = m.group(1).strip(), m.group(2)
        integrand = re.sub(rf"(?i)\bd\s*{re.escape(var)}\b", "", integrand).strip()
        return integrand, var
    m2 = INTEGRAL_TEXT_PAT.search(q)
    if m2:
        return m2.group(1).strip(), "x"
    return None

def is_math_question_conservative(q: str) -> bool:
    if not q: return False
    ql = q.strip().lower()
    if any(k in ql for k in ("integral", "integrate", "derivative", "differentiate", "limit", "simplify", "∫", "solve", "prob:")):
        return True
    if re.search(r"[0-9]|[+\-*/^=()]", ql):
        return True
    if re.search(r"\b(sin|cos|tan|log|ln|exp|sqrt)\b", ql):
        return True
    return False

def integrate_with_steps(expr: sp.Expr) -> str:
    out = []
    out.append("Integrate:")
    out.append(f"$$ {sp.latex(expr)}\\, dx $$\n")
    terms = list(expr.as_ordered_terms()) if isinstance(expr, sp.Add) else [expr]
    step_lines = []
    for t in terms:
        try:
            res_t = sp.integrate(t, X)
            if t == sp.log(X) or (t.is_Function and t.func == sp.log and t.args and t.args[0] == X):
                step_lines.append("For $\\int \\log(x)\\,dx$: parts with $u=\\log(x),\\;dv=dx$ ⇒ $x\\log(x)-x$.")
            elif t.is_Function and t.func == sp.log and t.args:
                step_lines.append(f"For $\\int \\log({sp.latex(t.args[0])})\\,dx$: use parts $u=\\log(\\cdot),\\;dv=dx$.")
            elif t.is_Pow and t.base == X and t.exp.is_number:
                n = sp.simplify(t.exp)
                step_lines.append(f"$\\int x^{{{sp.latex(n)}}} dx = \\frac{{x^{{{sp.latex(n+1)}}}}}{{{sp.latex(n+1)}}}$.")
            elif t.is_Mul and any(f.is_Pow and f.base==X for f in t.args):
                a = sp.simplify(t.as_coeff_mul(X)[0])
                pows = [f for f in t.args if (f.is_Pow and f.base==X)]
                if pows:
                    n = sp.simplify(pows[0].exp)
                    step_lines.append(f"$\\int {sp.latex(a)}\\,x^{{{sp.latex(n)}}} dx = {sp.latex(a)}\\,\\frac{{x^{{{sp.latex(n+1)}}}}}{{{sp.latex(n+1)}}}$.")
            step_lines.append(f"$\\int {sp.latex(t)}\\,dx = {sp.latex(res_t)}$")
        except Exception:
            step_lines.append(f"$\\int {sp.latex(t)}\\,dx = {sp.latex(sp.integrate(t, X))}$")
    if step_lines:
        out.append("Steps:")
        out.append("\n".join(f"- {s}" for s in step_lines))
    res = sp.integrate(expr, X)
    out.append("\nResult (indefinite):")
    out.append(f"$$ {sp.latex(res)} + C $$")
    return "\n".join(out)

def solve_equations(text: str) -> str:
    t = text.strip()
    mt = re.search(r"\bsolve\b(.*)", t, re.I)
    if mt:
        t = mt.group(1).strip(": ,")
    parts = [p.strip() for p in re.split(r"[;\n,]+", t) if p.strip()]
    eqs = []
    for p in parts:
        if "=" in p:
            left, right = p.split("=", 1)
            eqs.append(sp.Eq(_expr_from_text(left), _expr_from_text(right)))
        else:
            eqs.append(sp.Eq(_expr_from_text(p), 0))
    vars_set = set()
    for e in eqs:
        vars_set |= e.free_symbols
    pref = [sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")]
    ordered = [v for v in pref if v in vars_set] + [v for v in sorted(vars_set, key=lambda s: s.name) if v not in pref]
    try:
        if len(eqs) == 1:
            sol = sp.solve(eqs[0], ordered or list(vars_set), dict=True)
        else:
            sol = sp.solve(eqs, ordered or list(vars_set), dict=True)
        if not sol:
            return "No solution found (or infinitely many)."
        latex_solutions = []
        for s in sol:
            row = ", ".join(f"{sp.latex(k)} = {sp.latex(sp.simplify(v))}" for k, v in s.items())
            latex_solutions.append(row)
        body = ";\n".join(f"• {row}" for row in latex_solutions)
        return f"Solutions:\n\n{body}"
    except Exception as e:
        return f"❌ Could not solve: {e}"

PROB_HELP = (
    "Probability examples:\n"
    "- `prob: die P(X>=5)` (fair six-sided die)\n"
    "- `prob: coin P(X=1)` or `prob: coin(p=0.3) P(X=1)` (1=heads)\n"
    "- `prob: binomial(n=10,p=0.4) P(X>=3)`\n"
    "- `prob: poisson(l=2) P(X=3)`\n"
    "- `prob: normal(mu=0,sigma=1) P(-1 < X < 2)`\n"
    "- `prob: uniform(a=0,b=10) P(X>7)`\n"
    "- `prob: normal(mu=10,sigma=2) mean,var`"
)

def prob_query(text: str) -> str:
    s = text.strip()
    m = re.match(r"(?is)\s*prob\s*:\s*(.+)", s)
    if not m:
        return "Please start your query with `prob:`.\n\n" + PROB_HELP
    q = m.group(1).strip()

    X = None
    if re.match(r"(?is)^die\b", q):
        X = sps.Die('X', 6)
        tail = q[q.lower().find('die')+3:].strip()
    elif re.match(r"(?is)^coin", q):
        pm = re.search(r"p\s*=\s*([0-9]*\.?[0-9]+)", q)
        p = float(pm.group(1)) if pm else 0.5
        X = sps.Bernoulli('X', p)
        tail = q[q.lower().find('coin')+4:].strip()
    elif re.match(r"(?is)^binomial", q):
        nm = re.search(r"n\s*=\s*([0-9]+)", q); pm = re.search(r"p\s*=\s*([0-9]*\.?[0-9]+)", q)
        if not (nm and pm): return "For binomial, specify n and p. Eg: `prob: binomial(n=10,p=0.4) P(X>=3)`"
        n = int(nm.group(1)); p = float(pm.group(1))
        X = sps.Binomial('X', n, p)
        tail = q[q.lower().find('binomial')+8:].strip()
    elif re.match(r"(?is)^poisson", q):
        lm = re.search(r"(l|lambda)\s*=\s*([0-9]*\.?[0-9]+)", q)
        if not lm: return "For poisson, specify lambda. Eg: `prob: poisson(l=2) P(X=3)`"
        lam = float(lm.group(2))
        X = sps.Poisson('X', lam)
        tail = q[q.lower().find('poisson')+7:].strip()
    elif re.match(r"(?is)^normal", q):
        mu_m = re.search(r"mu\s*=\s*([\-0-9]*\.?[0-9]+)", q)
        sg_m = re.search(r"(sigma|sd)\s*=\s*([\-0-9]*\.?[0-9]+)", q)
        if not (mu_m and sg_m): return "For normal, specify mu and sigma. Eg: `prob: normal(mu=0,sigma=1) P(-1 < X < 2)`"
        mu = float(mu_m.group(1)); sigma = float(sg_m.group(2))
        X = sps.Normal('X', mu, sigma)
        tail = q[q.lower().find('normal')+6:].strip()
    elif re.match(r"(?is)^uniform", q):
        a_m = re.search(r"a\s*=\s*([\-0-9]*\.?[0-9]+)", q)
        b_m = re.search(r"b\s*=\s*([\-0-9]*\.?[0-9]+)", q)
        if not (a_m and b_m): return "For uniform, specify a and b. Eg: `prob: uniform(a=0,b=10) P(X>7)`"
        a = float(a_m.group(1)); b = float(b_m.group(1))
        X = sps.Uniform('X', a, b)
        tail = q[q.lower().find('uniform')+7:].strip()
    else:
        return "Unknown distribution.\n\n" + PROB_HELP

    if re.search(r"\bmean\b|\bE\b", tail, re.I) or re.search(r"\bvar\b|\bvariance\b", tail, re.I):
        rows=[]
        rows.append(f"**Mean**: $ {sp.latex(sps.E(X))} $")
        rows.append(f"**Variance**: $ {sp.latex(sps.Variance(X))} $")
        return "\n\n".join(rows)

    pm = re.search(r"P\s*\((.+)\)\s*$", tail, re.I)
    if not pm:
        return "Please specify a probability like `P(X>=3)` or `P(-1 < X < 2)`.\n\n" + PROB_HELP

    cond = pm.group(1).strip()
    try:
        rng = re.match(r"^\s*([\-0-9]*\.?[0-9]+)\s*<\s*X\s*<\s*([\-0-9]*\.?[0-9]+)\s*$", cond)
        if rng:
            a = float(rng.group(1)); b = float(rng.group(2))
            prob = sps.P(sp.And(X > a, X < b))
        else:
            cond = cond.replace("^", "**").replace("==", "=")
            cond = (cond.replace("≥", ">=").replace("≤", "<=").replace("≠", "!="))
            mrel = re.match(r"^\s*X\s*(<=|>=|!=|=|<|>)\s*([\-0-9]*\.?[0-9]+)\s*$", cond)
            if mrel:
                op, val = mrel.group(1), float(mrel.group(2))
                if op == "=": rel = sp.Eq(X, val)
                elif op == ">=": rel = sp.Ge(X, val)
                elif op == "<=": rel = sp.Le(X, val)
                elif op == ">": rel = sp.Gt(X, val)
                elif op == "<": rel = sp.Lt(X, val)
                else: rel = sp.Ne(X, val)
                prob = sps.P(rel)
            else:
                rel = _expr_from_text(cond)
                prob = sps.P(rel)
        prob_n = sp.N(prob, 8)
        return f"Probability:\n\n$ {sp.latex(prob)} \\approx {sp.latex(prob_n)} $"
    except Exception as e:
        return f"❌ Could not compute probability: {e}\n\n" + PROB_HELP

def solve_math_question(question: str, math_task: str = "Auto") -> str:
    q = (question or "").strip()
    if not q:
        return "❓ Please enter a math question."
    q = normalize_ocr_math(q)
    qlow = q.lower()

    pair = parse_integral_text(q)
    if pair and (math_task in ("Auto","Integrate")):
        integrand_txt, var = pair
        try:
            sym_var = sp.Symbol(var)
            expr = parse_expr(_clean_expr_text(integrand_txt),
                              transformations=TRANSFORMS, evaluate=True)
            res = sp.integrate(expr, sym_var)
            return (
                "Integrate:\n\n"
                f"$$ {sp.latex(expr)}\\, d{sp.latex(sym_var)} $$\n\n"
                "Result (indefinite):\n\n"
                f"$$ {sp.latex(res)} + C $$"
            )
        except Exception as e:
            return f"❌ Could not integrate parsed integrand: {e}"

    if math_task == "Probability" or qlow.startswith("prob:"):
        return prob_query(q)

    if math_task == "Solve" or re.search(r"\bsolve\b", qlow):
        return solve_equations(q)

    m2 = LIMIT_TEXT_PAT.search(q)
    if math_task in ("Auto","Limit") and m2:
        expr_raw, target = m2.group(1), m2.group(2)
        try:
            expr = _expr_from_text(expr_raw)
            a = _expr_from_text(target)
            res = sp.limit(expr, sp.Symbol("x"), a)
            return f"Given limit:\n\n$$\\lim_{{x\\to {sp.latex(a)}}} {sp.latex(expr)}$$\n\nResult:\n\n$$ {sp.latex(res)} $$"
        except Exception as e:
            return f"❌ Could not compute limit: {e}"

    if (math_task in ("Auto","Differentiate")) and any(k in qlow for k in ("differentiate","derivative")):
        try:
            start = qlow.index("differentiate") if "differentiate" in qlow else qlow.index("derivative")
            key = "differentiate" if "differentiate" in qlow else "derivative"
            expr_txt = q[start+len(key):].replace("of","",1)
            expr = _expr_from_text(expr_txt)
            res  = sp.diff(expr, sp.Symbol("x"))
            return f"Differentiate:\n\n$$ {sp.latex(expr)} $$\n\nResult:\n\n$$ {sp.latex(res)} $$"
        except Exception as e:
            return f"❌ Could not differentiate: {e}"

    if (math_task in ("Auto","Simplify")) and ("simplify" in qlow):
        try:
            expr_txt = q[qlow.index("simplify")+len("simplify"):]
            expr = _expr_from_text(expr_txt)
            res  = sp.simplify(expr)
            return f"Simplify:\n\n$$ {sp.latex(expr)} $$\n\nResult:\n\n$$ {sp.latex(res)} $$"
        except Exception as e:
            return f"❌ Could not simplify: {e}"

    try:
        expr = _expr_from_text(q)
        has_symbols = bool(expr.free_symbols)
        if math_task == "Evaluate" or (math_task == "Auto" and not has_symbols):
            exact = sp.simplify(expr)
            decimal = sp.N(exact, 15)
            return (
                "Evaluate:\n\n"
                f"$$ {sp.latex(expr)} $$\n\n"
                "Result:\n\n"
                f"**Exact:** $$ {sp.latex(exact)} $$\n\n"
                f"**Decimal:** $$ {sp.latex(decimal)} $$"
            )
        if math_task in ("Auto","Integrate"):
            return integrate_with_steps(expr)
        elif math_task == "Differentiate":
            res = sp.diff(expr, sp.Symbol("x"))
            return f"Differentiate:\n\n$$ {sp.latex(expr)} $$\n\nResult:\n\n$$ {sp.latex(res)} $$"
        elif math_task == "Simplify":
            res = sp.simplify(expr)
            return f"Simplify:\n\n$$ {sp.latex(expr)} $$\n\nResult:\n\n$$ {sp.latex(res)} $$"
        elif math_task == "Limit":
            return "ℹ️ Provide the limit in the form: `limit <expr> as x->a`"
        else:
            return integrate_with_steps(expr)
    except Exception as e:
        return f"⚠️ I couldn't parse the math expression: {e}\nTry: `Evaluate 2+3`, `solve x^2-5x+6=0`, or `prob: normal(mu=0,sigma=1) P(-1<X<2)`."

# ---------- UI ----------
st.title("🤖 STEM Bot — SymPy Math + LLM (Science/Social)")

with st.sidebar:
    subject = st.radio("Subject", ["math","sci","socio"], horizontal=True, index=0)
    math_task = st.selectbox(
        "Math Task",
        ["Auto", "Evaluate", "Integrate", "Differentiate", "Simplify", "Limit", "Solve", "Probability"],
        index=0
    )
    auto_detect_math = st.checkbox("Auto-detect math in questions (when not in Math)", value=False)
    if subject == "math" and math_task == "Probability":
        st.info("Tip: use the helper syntax.\n\n" + PROB_HELP)

    mode = st.radio("Mode (for Sci/Socio)", ["short","long","mcq"], horizontal=True, index=0)
    temperature = st.slider("Creativity (LLM only; lower = precise)", 0.0, 1.0, 0.18, 0.02)

    use_retrieval = st.checkbox("Use Retrieval (FAISS + bge-small)", value=False)
    top_k = st.slider("Top-k", 1, 10, 5)

    default_model = HF_MODEL_SCI if subject=="sci" else (HF_MODEL_SOCIO if subject=="socio" else "")
    model_id = st.text_input("HF chat model (for Sci/Socio)", value=default_model)

    st.caption("🔐 Using hard-coded HF token (not recommended)")

    if st.button("♻️ Reset OCR cache"):
        get_ocr.clear(); st.success("OCR cache cleared.")
    if st.button("♻️ Reset PDF cache"):
        get_pdf_backend.clear(); st.success("PDF cache cleared.")

# session state
if "messages" not in st.session_state: st.session_state.messages=[]
if "index" not in st.session_state: st.session_state.index=None
if "meta" not in st.session_state: st.session_state.meta=[]
if "retriever_ready" not in st.session_state: st.session_state.retriever_ready=False
if "generating" not in st.session_state: st.session_state.generating=False

# Controls
cA, cB, cC = st.columns([1,1,3])
with cA:
    if st.button("🆕 New chat", use_container_width=True, disabled=st.session_state.generating):
        st.session_state.messages=[]
with cB:
    if st.button("⬇️ Download chat", use_container_width=True, disabled=st.session_state.generating):
        md = "\n\n".join([f"**{m['role'].upper()}**:\n\n{m['content']}" for m in st.session_state.messages])
        st.download_button("Download .md", data=md, file_name="chat.md", mime="text/markdown", use_container_width=True)

st.markdown("---")

left, right = st.columns([2,1], vertical_alignment="top")
with left:
    st.subheader("Ask")
    question = st.text_area(
        "Question",
        placeholder="e.g., '2+3', '∫ (2x^3 + log(x^2)) dx', 'solve x^2-5x+6=0', 'prob: normal(mu=0,sigma=1) P(-1<X<2)', or 'Why is the Earth round?'",
        height=120
    )
    options_block = st.text_area(
        "MCQ Options (single box)",
        placeholder="A) ...\nB) ...\nC) ...\nD) ...",
        height=90
    ) if subject!="math" and mode=="mcq" else None

    st.markdown("**Uploads (optional)**")
    up_pdf = st.file_uploader("PDF", type=["pdf"])
    up_img = st.file_uploader("Image", type=["png","jpg","jpeg"])
    extra_notes = st.text_area("Extra notes/context", height=80)

    go = st.button("Get Answer", type="primary", use_container_width=True, disabled=st.session_state.generating)

    if go:
        st.session_state.generating=True
        try:
            ctx=[]
            if use_retrieval and st.session_state.retriever_ready and (question and question.strip()):
                embedder = get_embedder()
                hits = search_index(st.session_state.index, st.session_state.meta, question, embedder, k=top_k)
                ctx.extend([h[1]["text"][:MAX_CTX_SNIPPET] for h in hits])

            if up_pdf is not None:
                try:
                    text = extract_pdf_text_inmemory(up_pdf) or ""
                    if text.strip():
                        ctx.append(text[:MAX_CTX_SNIPPET])
                        st.success("PDF text extracted in-memory.")
                        if subject == "math" and not (question and question.strip()):
                            question = text.strip().split("\n")[0]
                except Exception as e:
                    st.error(f"PDF parse failed: {e}")

            if up_img is not None:
                try:
                    img = Image.open(up_img).convert("RGB")
                    ocr_text, backend = run_ocr_chain(img)
                    if ocr_text.strip():
                        ctx.append(ocr_text[:MAX_CTX_SNIPPET])
                        st.success(f"OCR extracted text using {backend}.")
                        if subject == "math" and not (question and question.strip()):
                            question = ocr_text.strip().split("\n")[0]
                except Exception as e:
                    st.error(f"OCR failed: {e}")

            user_disp = question or ""
            if options_block and (options_block or "").strip():
                user_disp += "\n\nOptions:\n" + options_block.strip()
            st.session_state.messages.append({"role":"user","content":user_disp})

            auto_math = is_math_question_conservative(question or "") if auto_detect_math else False
            use_math = (subject == "math") or auto_math
            if auto_math and subject != "math":
                st.info("This looks like a math question. Solving with the Math engine for better accuracy.")

            if use_math:
                with st.spinner("🧮 Solving with SymPy…"):
                    final = solve_math_question(question or "", math_task)
            else:
                opts = parse_mcq_options(options_block or "") if (options_block and mode=="mcq") else None
                sys = system_prompt(subject)
                usr = user_prompt(subject, mode, question or "", opts, ctx)
                chosen = (model_id or "").strip() or (HF_MODEL_SCI if subject=="sci" else HF_MODEL_SOCIO)
                with st.spinner("🤔 Thinking…"):
                    text = chat_once(
                        chosen,
                        [{"role":"system","content":sys},{"role":"user","content":usr}],
                        temperature=temperature, top_p=0.85, max_tokens=1000
                    )
                final = clean_repeats_final(text).strip() or "No answer returned."

            st.session_state.messages.append({"role":"assistant","content":final})

        finally:
            st.session_state.generating=False

    st.subheader("Chat")
    for m in st.session_state.messages:
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            txt = m["content"]
            blocks = list(re.finditer(r"\$\$(.+?)\$\$", txt, re.S))
            if not blocks:
                st.markdown(txt)
            else:
                pos = 0
                for b in blocks:
                    pre = txt[pos:b.start()]
                    if pre.strip(): st.markdown(pre)
                    eq = b.group(1).strip()
                    if eq: st.latex(eq)
                    pos = b.end()
                rest = txt[pos:]
                if rest.strip(): st.markdown(rest)

with right:
    st.subheader("Retrieval")
    if st.button("(Re)Build Index", disabled=st.session_state.generating):
        texts=[]
        if extra_notes and extra_notes.strip():
            texts.append({"id":"notes","text":extra_notes.strip()[:50000],"meta":{"source":"notes"}})
        if up_pdf is not None:
            try:
                t = extract_pdf_text_inmemory(up_pdf) or ""
                if t.strip(): texts.append({"id":"pdf","text":t[:50000],"meta":{"source":getattr(up_pdf,'name','pdf')}})
            except Exception as e:
                st.error(f"PDF parse failed: {e}")
        if up_img is not None:
            try:
                img = Image.open(up_img).convert("RGB")
                ocr_text, backend = run_ocr_chain(img)
                if ocr_text.strip(): texts.append({"id":"image","text":ocr_text[:30000],"meta":{"source":getattr(up_img,'name','image')}})
            except Exception as e:
                st.error(f"OCR failed: {e}")
        if texts:
            with st.spinner("Building FAISS…"):
                embedder = get_embedder()
                index, meta = build_faiss_index(texts, embedder)
            st.session_state.index=index; st.session_state.meta=meta; st.session_state.retriever_ready=True
            st.success(f"Index ready with {len(texts)} doc(s).")
        else:
            st.warning("No text to index. Add notes or upload a PDF/Image.")

# ---------- Footer ----------
st.markdown("---")
st.caption("⚡ STEM Bot — SymPy math (Evaluate/Integrate/Differentiate/Simplify/Limit/Solve/Probability), "
           "HF models for Science/Social, in-memory OCR & PDF parsing, optional FAISS retrieval.")
