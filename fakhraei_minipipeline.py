import os
import json
import gzip
import tarfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, classification_report
from sklearn.manifold import TSNE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "fakhraei")
OUT_DIR = os.path.join(BASE_DIR, "output", "fakhraei")
os.makedirs(OUT_DIR, exist_ok=True)

def load_usersdata(path):
    try:
        with gzip.open(path, "rb") as gb:
            bio = io.BytesIO(gb.read())
        try:
            tf = tarfile.open(fileobj=bio, mode="r:*")
            member = None
            for m in tf.getmembers():
                if os.path.basename(m.name).lower().startswith("usersdata"):
                    member = m
                    break
            if member is None:
                raise RuntimeError("usersdata not found in tar")
            fobj = tf.extractfile(member)
            df = pd.read_csv(io.TextIOWrapper(fobj, encoding="utf-8", errors="ignore"), sep="\t", header=None)
            return df
        except tarfile.ReadError:
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                try:
                    df = pd.read_csv(f, sep="\t")
                except Exception:
                    df = pd.read_csv(f)
            return df
    except Exception:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            df = pd.read_csv(f, sep="\t")
        return df

def find_label_column(df):
    candidates = ["label", "is_spammer", "spammer", "class", "is_spam", "spam"]
    for c in candidates:
        if c in df.columns:
            return c
    if df.shape[1] >= 2:
        last = df.columns[-1]
        vals = df[last]
        if pd.api.types.is_numeric_dtype(vals):
            uniq = pd.Series(vals).dropna().unique()
            if len(uniq) <= 5 and set(pd.Series(uniq).astype(int)) <= {0,1}:
                return last
    return None

def get_numeric_df(df, label_col):
    exclude = {label_col, "user_id", "userid", "id", "screen_name", "name", "created_at", "location", "description"}
    cols = [c for c in df.columns if c not in exclude]
    df2 = df[cols].copy()
    for c in df2.columns:
        if df2[c].dtype == bool:
            df2[c] = df2[c].astype(int)
        elif not np.issubdtype(df2[c].dtype, np.number):
            if df2[c].dtype == object and set(df2[c].dropna().unique()).issubset({"M","F","m","f"}):
                df2[c] = df2[c].map({"M":1,"F":0,"m":1,"f":0})
            else:
                df2[c] = pd.to_numeric(df2[c], errors="ignore")
    num_cols = [c for c in df2.columns if np.issubdtype(df2[c].dtype, np.number)]
    return df2[num_cols], num_cols

def supervised_pipeline(df, label_col):
    y = df[label_col]
    if y.dtype == object:
        y = y.astype("category").cat.codes
    if y.nunique() > 2:
        y = (y == y.max()).astype(int)
    X_df, X_cols = get_numeric_df(df, label_col)
    X = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(X) > 50000:
        df_idx = np.random.RandomState(42).choice(len(X), 50000, replace=False)
        X = X.iloc[df_idx]
        y = y.iloc[df_idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced_subsample')
    lr.fit(X_tr_s, y_tr)
    rf.fit(X_tr, y_tr)
    preds_lr = lr.predict(X_test_s)
    preds_rf = rf.predict(X_test)
    proba_lr = lr.predict_proba(X_test_s)[:, 1] if hasattr(lr, "predict_proba") else None
    proba_rf = rf.predict_proba(X_test)[:, 1] if hasattr(rf, "predict_proba") else None
    proba_rf_val = rf.predict_proba(X_val)[:, 1] if hasattr(rf, "predict_proba") else None
    def eval_metrics(y_true, y_pred, y_score):
        acc = accuracy_score(y_true, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        mpr, mrc, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        roc = roc_auc_score(y_true, y_score) if y_score is not None else None
        prauc = average_precision_score(y_true, y_score) if y_score is not None else None
        return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "macro_precision": mpr, "macro_recall": mrc, "macro_f1": mf1, "roc_auc": roc, "pr_auc": prauc}
    res_lr = eval_metrics(y_test, preds_lr, proba_lr)
    res_rf = eval_metrics(y_test, preds_rf, proba_rf)
    best_thresh = None
    best_f1 = -1.0
    tuned_res = None
    tuned_preds = None
    if proba_rf_val is not None:
        for t in np.linspace(0.05, 0.95, 19):
            p_val = (proba_rf_val >= t).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(y_val, p_val, average="binary")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)
        if best_thresh is not None and proba_rf is not None:
            tuned_preds = (proba_rf >= best_thresh).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(y_test, tuned_preds, average="binary")
            tuned_res = {
                "accuracy": float(accuracy_score(y_test, tuned_preds)),
                "precision": float(pr),
                "recall": float(rc),
                "f1": float(f1),
                "roc_auc": float(roc_auc_score(y_test, proba_rf)),
                "pr_auc": float(average_precision_score(y_test, proba_rf))
            }
    report_lr = classification_report(y_test, preds_lr)
    report_rf = classification_report(y_test, preds_rf)
    vc_train = pd.Series(y_train).value_counts()
    vc_val = pd.Series(y_val).value_counts()
    vc_test = pd.Series(y_test).value_counts()
    class_counts = {
        "train": {str(int(k)): int(v) for k, v in vc_train.items()},
        "val": {str(int(k)): int(v) for k, v in vc_val.items()},
        "test": {str(int(k)): int(v) for k, v in vc_test.items()},
    }
    fig_dir = OUT_DIR
    try:
        plt.figure(figsize=(6,4))
        from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
        if proba_lr is not None:
            RocCurveDisplay.from_predictions(y_test, proba_lr)
            plt.title("LR ROC")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "lr_roc.png"), dpi=160)
            plt.close()
            PrecisionRecallDisplay.from_predictions(y_test, proba_lr)
            plt.title("LR PR")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "lr_pr.png"), dpi=160)
            plt.close()
        if proba_rf is not None:
            RocCurveDisplay.from_predictions(y_test, proba_rf)
            plt.title("RF ROC")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "rf_roc.png"), dpi=160)
            plt.close()
            PrecisionRecallDisplay.from_predictions(y_test, proba_rf)
            plt.title("RF PR")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "rf_pr.png"), dpi=160)
            plt.close()
        ConfusionMatrixDisplay.from_predictions(y_test, preds_rf)
        plt.title("RF Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "rf_cm.png"), dpi=160)
        plt.close()
        if tuned_preds is not None:
            ConfusionMatrixDisplay.from_predictions(y_test, tuned_preds)
            plt.title("RF Confusion Matrix (threshold-tuned)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "rf_cm_tuned.png"), dpi=160)
            plt.close()
        try:
            importances = rf.feature_importances_
            idxs = np.argsort(importances)[::-1][:20]
            names = [str(X_cols[i]) for i in idxs]
            vals = importances[idxs]
            plt.figure(figsize=(8,5))
            sns.barplot(x=vals, y=names, orient='h')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('RF Top-20 Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "rf_importance.png"), dpi=160)
            plt.close()
        except Exception:
            pass
    except Exception:
        pass
    safe_features = []
    for c in X_cols:
        if isinstance(c, (np.integer,)):
            safe_features.append(int(c))
        else:
            safe_features.append(str(c))
    results = {
        "label_col": str(label_col),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": safe_features,
        "logreg": {k:(float(v) if isinstance(v,(np.floating,)) else (None if v is None else v)) for k,v in res_lr.items()},
        "random_forest": {k:(float(v) if isinstance(v,(np.floating,)) else (None if v is None else v)) for k,v in res_rf.items()},
        "random_forest_tuned": (tuned_res if tuned_res is not None else None),
        "random_forest_best_threshold": (best_thresh if best_thresh is not None else None),
        "classification_report_lr": report_lr,
        "classification_report_rf": report_rf,
        "class_counts": class_counts,
    }
    return results

def unsupervised_pipeline(df):
    X_df, X_cols = get_numeric_df(df, None)
    X = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(X) > 100000:
        df_idx = np.random.RandomState(42).choice(len(X), 100000, replace=False)
        X = X.iloc[df_idx]
    iso = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
    scores = -iso.fit_predict(X)
    top_idx = np.argsort(scores)[-50:]
    top = df.iloc[top_idx]
    safe_features = []
    for c in X_cols:
        if isinstance(c, (np.integer,)):
            safe_features.append(int(c))
        else:
            safe_features.append(str(c))
    results = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": safe_features,
        "top_anomalies_preview": top.head(10).to_dict(orient="records"),
    }
    return results

def usersdata_visuals(df, label_col):
    try:
        if label_col is None:
            return
        y = df[label_col]
        X_df, X_cols = get_numeric_df(df, label_col)
        if len(X_df) > 50000:
            idx = np.random.RandomState(42).choice(len(X_df), 50000, replace=False)
            X_df = X_df.iloc[idx]
            y = y.iloc[idx]
        plt.figure(figsize=(6,4))
        if 3 in X_df.columns:
            sns.histplot(x=X_df[3], hue=y, element="step", stat="density", common_norm=False)
            plt.xlabel("Age")
            plt.title("Age distribution by label")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "users_age_hist.png"), dpi=160)
            plt.close()
        if 2 in X_df.columns and 3 in X_df.columns:
            plt.figure(figsize=(6,4))
            sns.scatterplot(x=X_df[2], y=X_df[3], hue=y, s=10, alpha=0.6)
            plt.xlabel("Feature2")
            plt.ylabel("Age")
            plt.title("Feature2 vs Age by label")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "users_scatter.png"), dpi=160)
            plt.close()
        try:
            n = min(len(X_df), 10000)
            idx = np.random.RandomState(123).choice(len(X_df), n, replace=False)
            Xn = X_df.iloc[idx]
            yn = y.iloc[idx]
            ts = TSNE(n_components=2, random_state=42, perplexity=30)
            emb = ts.fit_transform(Xn)
            plt.figure(figsize=(6,4))
            sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=yn, s=8, alpha=0.7, palette="Set1")
            plt.axis('off')
            plt.title("t-SNE of users features")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "users_tsne.png"), dpi=160)
            plt.close()
        except Exception:
            pass
    except Exception:
        pass

def build_graph_and_visualize(rel_path, users_df=None):
    if not os.path.exists(rel_path):
        return None
    G = nx.Graph()
    try:
        edges = []
        import gzip
        with gzip.open(rel_path, "rt", encoding="utf-8", errors="ignore") as f:
            cnt = 0
            for line in f:
                if line.startswith("relations.csv"):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    u = int(parts[1])
                    v = int(parts[3])
                    edges.append((u, v))
                except Exception:
                    continue
                cnt += 1
                if cnt >= 100000:
                    break
        if not edges:
            return None
        G.add_edges_from(edges)
        deg = dict(G.degree())
        nodes_sorted = sorted(deg.keys(), key=lambda n: deg[n], reverse=True)
        nodes = nodes_sorted[:1000]
        H = G.subgraph(nodes).copy()
        pos = nx.spring_layout(H, seed=42)
        dvals = np.array([deg.get(n,0) for n in H.nodes()])
        vmax = max(dvals.max(),1)
        plt.figure(figsize=(8,6))
        nx.draw_networkx_nodes(H, pos, node_size=20, node_color=dvals, cmap=plt.cm.viridis, vmin=0, vmax=vmax)
        nx.draw_networkx_edges(H, pos, alpha=0.15)
        plt.axis('off')
        plt.tight_layout()
        outp = os.path.join(OUT_DIR, "social_graph.png")
        plt.savefig(outp, dpi=180)
        plt.close()
        d = np.array(list(deg.values()))
        plt.figure(figsize=(6,4))
        sns.histplot(d, bins=50)
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.title("Degree distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "degree_hist.png"), dpi=160)
        plt.close()
        plt.figure(figsize=(6,4))
        sns.histplot(d, bins=200)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Degree (log)")
        plt.ylabel("Count (log)")
        plt.title("Degree distribution (log-log)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "degree_loglog.png"), dpi=160)
        plt.close()
        summary = {
            "loaded_edges": int(len(edges)),
            "loaded_nodes": int(len(deg)),
            "subgraph_nodes": int(H.number_of_nodes()),
            "subgraph_edges": int(H.number_of_edges()),
            "avg_degree": float(np.mean(d)) if len(d)>0 else 0.0,
            "max_degree": int(np.max(d)) if len(d)>0 else 0,
            "components": int(nx.number_connected_components(H)),
        }
        try:
            summary["avg_clustering_subgraph"] = float(nx.average_clustering(H))
        except Exception:
            summary["avg_clustering_subgraph"] = None
        with open(os.path.join(OUT_DIR, "graph_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return outp
    except Exception:
        return None

def main():
    users_path = os.path.join(DATA_DIR, "usersdata.csv.gz")
    if not os.path.exists(users_path):
        raise FileNotFoundError(users_path)
    df = load_usersdata(users_path)
    label_col = find_label_column(df)
    if label_col is not None:
        results = supervised_pipeline(df, label_col)
        mode = "supervised"
        usersdata_visuals(df, label_col)
    else:
        results = unsupervised_pipeline(df)
        mode = "unsupervised"
    results["mode"] = mode
    out_json = os.path.join(OUT_DIR, "usersdata_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(out_json)
    rel_path = os.path.join(DATA_DIR, "relations.csv.gz")
    gimg = build_graph_and_visualize(rel_path, df)
    if gimg:
        print(gimg)

if __name__ == "__main__":
    main()
