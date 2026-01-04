import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
import pandas as pd
import numpy as np
import time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Indonesian Humor Detection",
    layout="wide"
)

# ============================================================
# TEXT PREPROCESSOR
# ============================================================
class TextPreprocessor:
    def __init__(self, model_name='indolem/indobert-base-uncased'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
    
    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.!\?,;:\-()]', '', text)
        return text


# ============================================================
# MODEL ARCHITECTURE — ORIGINAL TRAINING VERSION
# ============================================================
class HumorDetectionModel(nn.Module):
    def __init__(self, model_name='indolem/indobert-base-uncased', dropout_rate=0.3):
        super().__init__()

        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except:
            self.bert = AutoModel.from_pretrained('bert-base-multilingual-uncased')

        self.bert_dim = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        # ORIGINAL TRAINING ARCHITECTURE — EXACT MATCH
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = self.dropout(output.pooler_output)
        return self.classifier(pooled)


# ============================================================
# MODEL CONFIG
# ============================================================
MODEL_CONFIGS = {
    "Eksperimen 1": {
        "file": "eksperimen_1.pth",
        "params": {"max_seq_len": 128, "bert_lr": "1e-5", "dropout": 0.1},
        "metrics": {"accuracy": 0.9747, "precision": 0.9776, "recall": 0.9718, "f1_score": 0.9746}
    },
    "Eksperimen 2": {
        "file": "eksperimen_2.pth",
        "params": {"max_seq_len": 128, "bert_lr": "1e-5", "dropout": 0.3},
        "metrics": {"accuracy": 0.9745, "precision": 0.9744, "recall": 0.9747, "f1_score": 0.9745}
    },
    "Eksperimen 3": {
        "file": "eksperimen_3.pth",
        "params": {"max_seq_len": 128, "bert_lr": "2e-5", "dropout": 0.1},
        "metrics": {"accuracy": 0.9755, "precision": 0.9722, "recall": 0.9749, "f1_score": 0.9736}
    },
    "Eksperimen 4": {
        "file": "eksperimen_4.pth",
        "params": {"max_seq_len": 128, "bert_lr": "2e-5", "dropout": 0.3},
        "metrics": {"accuracy": 0.9740, "precision": 0.9757, "recall": 0.9722, "f1_score": 0.9740}
    },
    "Eksperimen 5": {
        "file": "eksperimen_5.pth",
        "params": {"max_seq_len": 128, "bert_lr": "5e-5", "dropout": 0.1},
        "metrics": {"accuracy": 0.9715, "precision": 0.9838, "recall": 0.9587, "f1_score": 0.9711}
    },
    "Eksperimen 6": {
        "file": "eksperimen_6.pth",
        "params": {"max_seq_len": 128, "bert_lr": "5e-5", "dropout": 0.3},
        "metrics": {"accuracy": 0.9720, "precision": 0.9746, "recall": 0.9692, "f1_score": 0.9719}
    },
    "Eksperimen 7": {
        "file": "eksperimen_7.pth",
        "params": {"max_seq_len": 256, "bert_lr": "1e-5", "dropout": 0.1},
        "metrics": {"accuracy": 0.9751, "precision": 0.9753, "recall": 0.9749, "f1_score": 0.9751}
    },
    "Eksperimen 8": {
        "file": "eksperimen_8.pth",
        "params": {"max_seq_len": 256, "bert_lr": "1e-5", "dropout": 0.3},
        "metrics": {"accuracy": 0.9750, "precision": 0.9760, "recall": 0.9740, "f1_score": 0.9750}
    },
    "Eksperimen 9": {
        "file": "eksperimen_9.pth",
        "params": {"max_seq_len": 256, "bert_lr": "2e-5", "dropout": 0.1},
        "metrics": {"accuracy": 0.9751, "precision": 0.9769, "recall": 0.9731, "f1_score": 0.9750}
    },
    "Eksperimen 10": {
        "file": "eksperimen_10.pth",
        "params": {"max_seq_len": 256, "bert_lr": "2e-5", "dropout": 0.3},
        "metrics": {"accuracy": 0.9751, "precision": 0.9732, "recall": 0.9770, "f1_score": 0.9751}
    },
    "Eksperimen 11": {
        "file": "eksperimen_11.pth",
        "params": {"max_seq_len": 256, "bert_lr": "5e-5", "dropout": 0.1},
        "metrics": {"accuracy": 0.9710, "precision": 0.9659, "recall": 0.9765, "f1_score": 0.9712}
    },
    "Eksperimen 12": {
        "file": "eksperimen_12.pth",
        "params": {"max_seq_len": 256, "bert_lr": "5e-5", "dropout": 0.3},
        "metrics": {"accuracy": 0.9728, "precision": 0.9778, "recall": 0.9677, "f1_score": 0.9727}
    }
}

# ============================================================
# LOAD MODELS (PyTorch 2.6 FIX)
# ============================================================
@st.cache_resource
def load_models():
    available = {}
    preprocessor = TextPreprocessor()
    device = torch.device("cpu")

    import os
    for name, cfg in MODEL_CONFIGS.items():
        if os.path.exists(cfg["file"]):
            available[name] = {
                "config": cfg,
                "metrics": cfg["metrics"],
                "loaded": False
            }

    return available, preprocessor, device


def load_single_model(model_name, cfg, device):
    import numpy as np
    import torch

    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])

        raw_state = torch.load(
            cfg["file"],
            map_location="cpu",
            weights_only=False
        )

        model = HumorDetectionModel(dropout_rate=cfg["params"]["dropout"])

        if isinstance(raw_state, dict):
            if "model_state_dict" in raw_state:
                model.load_state_dict(raw_state["model_state_dict"])
            elif "state_dict" in raw_state:
                model.load_state_dict(raw_state["state_dict"])
            else:
                model.load_state_dict(raw_state)
        else:
            model.load_state_dict(raw_state)

        model = model.to(device)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Gagal memuat {model_name}: {e}")
        return None


# ============================================================
# SINGLE TEXT PREDICTION
# ============================================================
def predict_single(text, model, prep, device, max_len):
    cleaned = prep.clean_text(text)
    enc = prep.tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    start = time.time()
    with torch.no_grad():
        out = model(ids, mask)
        prob = torch.sigmoid(out).item()
        pred = prob > 0.5
    latency = time.time() - start

    return pred, prob, latency


# ============================================================
# SUMMARY (NO DIV BY ZERO)
# ============================================================
def create_summary(results):
    if len(results) == 0:
        st.error("Tidak ada model yang berhasil memproses teks.")
        return

    votes = sum(1 for r in results.values() if r["prediction"])
    total = len(results)
    consensus = votes / total if total > 0 else 0

    avg_conf = sum(r["confidence"] for r in results.values()) / total

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if consensus >= 0.6:
            st.success("HUMOR")
        elif consensus >= 0.4:
            st.warning("UNCERTAIN")
        else:
            st.info("NON-HUMOR")

    with col2:
        st.metric("Consensus", f"{consensus:.1%}")

    with col3:
        st.metric("Agreement", f"{votes}/{total}")

    with col4:
        st.metric("Avg Confidence", f"{avg_conf:.2%}")

    table = []
    for name, r in results.items():
        cfg = r["config"]['params']
        table.append({
            "Model": name,
            "Result": "Humor" if r["prediction"] else "Non-Humor",
            "Probability": f"{r['probability']:.3f}",
            "Config": f"Len:{cfg['max_seq_len']} | LR:{cfg['bert_lr']} | Drop:{cfg['dropout']}"
        })

    st.dataframe(pd.DataFrame(table), use_container_width=True)


def create_light_chart(results):
    if len(results) == 0:
        return
    names = list(results.keys())
    probs = [results[n]["probability"] for n in names]
    df = pd.DataFrame({"Model": names, "Probability": probs})
    st.bar_chart(df.set_index("Model"))


# ============================================================
# MODEL STATISTICS
# ============================================================
def create_metrics_table(models):
    rows = []
    for name, data in models.items():
        m = data["metrics"]
        p = data["config"]["params"]
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1_score"],
            "Max Len": p["max_seq_len"],
            "LR": p["bert_lr"],
            "Dropout": p["dropout"],
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN UI
# ============================================================
def main():
    st.title("Indonesian Humor Detection")

    models, prep, device = load_models()

    with st.sidebar:
        st.header("Performance Metrics")
        metrics_df = create_metrics_table(models)
        st.dataframe(metrics_df, height=350)

        if len(models) > 0:
            best = max(models.keys(), key=lambda x: models[x]["metrics"]["f1_score"])
            st.success(f"Best Model: {best}")

    tab1, tab2 = st.tabs(["🔍 Single Text Analysis", "📊 Model Statistics"])

    # ============================================================
    # TAB 1 — SINGLE TEXT ANALYSIS
    # ============================================================
    with tab1:
        input_text = st.text_area(
            "Masukkan teks:",
            placeholder="Contoh: Kenapa kucing suka tidur? Karena dia nggak punya cicilan."
        )

        mode = st.radio(
            "Pilih mode:",
            ["Semua Model (12)", "Model Terpilih"],
            horizontal=True
        )

        if mode == "Model Terpilih":
            selected = st.selectbox("Pilih model:", list(models.keys()))

        if st.button("Analisis", type="primary"):
            if not input_text.strip():
                st.warning("Masukkan teks dulu.")
                return

            results = {}

            if mode == "Semua Model (12)":
                for name, data in models.items():
                    model = load_single_model(name, data["config"], device)
                    if not model:
                        continue

                    cfg = data["config"]["params"]
                    pred, prob, t = predict_single(input_text, model, prep, device, cfg["max_seq_len"])

                    results[name] = {
                        "prediction": pred,
                        "probability": prob,
                        "confidence": prob if pred else (1 - prob),
                        "config": data["config"]
                    }

                if len(results) == 0:
                    st.error("Tidak ada model berhasil diproses.")
                    return

                st.subheader("Hasil 12 Model")
                create_summary(results)
                create_light_chart(results)

            else:  # Model Terpilih
                data = models[selected]
                model = load_single_model(selected, data["config"], device)
                if not model:
                    st.error("Gagal memuat model.")
                    return

                cfg = data["config"]["params"]
                pred, prob, t = predict_single(input_text, model, prep, device, cfg["max_seq_len"])

                st.subheader(f"Hasil {selected}")
                if pred:
                    st.success("HUMOR")
                else:
                    st.info("NON-HUMOR")
                st.metric("Probability", f"{prob:.3f}")

                st.subheader("Configuration")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.caption("Max Seq Length")
                    st.markdown(f"### {cfg['max_seq_len']}")

                with col2:
                    st.caption("BERT LR")
                    st.markdown(f"### {cfg['bert_lr']}")

                with col3:
                    st.caption("Dropout")
                    st.markdown(f"### {cfg['dropout']}")

    # ============================================================
    # TAB 2 — MODEL STATISTICS
    # ============================================================
    with tab2:
        st.subheader("Statistik Lengkap 12 Model")
        metrics_df = create_metrics_table(models)
        st.dataframe(metrics_df, use_container_width=True)

        best_acc = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
        best_prec = metrics_df.loc[metrics_df['Precision'].idxmax()]
        best_rec = metrics_df.loc[metrics_df['Recall'].idxmax()]
        best_f1 = metrics_df.loc[metrics_df['F1 Score'].idxmax()]

        st.subheader("Best Performance Summary")
        st.write(f"**Best Accuracy:** {best_acc['Model']} ({best_acc['Accuracy']:.4f})")
        st.write(f"**Best Precision:** {best_prec['Model']} ({best_prec['Precision']:.4f})")
        st.write(f"**Best Recall:** {best_rec['Model']} ({best_rec['Recall']:.4f})")
        st.write(f"**Best F1 Score:** {best_f1['Model']} ({best_f1['F1 Score']:.4f})")

        st.subheader("Ranking Model (Berdasarkan F1 Score)")
        st.dataframe(metrics_df.sort_values("F1 Score", ascending=False), use_container_width=True)

        st.subheader("Analisis Eksperimen")
        st.info("""
1. Max Length 256 secara konsisten menghasilkan F1-Score yang lebih unggul dibandingkan 128  
2. Learning Rate BERT optimal: 1e-5 hingga 2e-5 memberikan performa terbaik  
3. Dropout Rate: Baik 0.1 maupun 0.3 memberikan hasil yang baik  
4. Learning Rate tinggi (5e-5): Cenderung menurunkan performa secara konsisten  
        """)


# ============================================================
if __name__ == "__main__":
    main()
