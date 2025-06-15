# ============================================================
#  fraud_detection_experiments.py
#  Exact-money fraud-detection pipeline + diagnostics
# ============================================================

"""
Models
------
Isolation Forest · Autoencoder · XGBoost · LSTM

Money metrics (per test set)
----------------------------
fn_loss    = Σ(Amount) of all false-negatives
fp_cost    = FP_FLAT × #FP  +  FP_PCT × Σ(Amount of false-positives)
total_cost = fn_loss + fp_cost
"""

# ╭──────────────── Imports ─────────────────╮
import time, random, warnings
from pathlib import Path
from typing  import Tuple, List
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.colors import ListedColormap
warnings.filterwarnings("ignore")

SEED = 69
np.random.seed(SEED); random.seed(SEED)

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (precision_score, recall_score, f1_score,
                                     roc_auc_score, confusion_matrix,
                                     precision_recall_curve,
                                     average_precision_score, roc_curve, auc)
from sklearn.ensemble        import IsolationForest
from xgboost                 import XGBClassifier
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
# ╰────────────────────────────────────────────╯


# ╭────────── Configuration ───────────╮
CONFIG = {
    "DATA_PATH":  r"C:\Users\theka\OneDrive\Pulpit\diploma\creditcard.csv",
    "TEST_SIZE":  0.30,
    "CV_FOLDS":   5,
    "RANDOM_STATE": SEED,

    "ISF":  {"n_estimators":[200], "max_samples":["auto"], "contamination":[0.002]},
    "XGB":  {"n_estimators":[400], "max_depth":[4], "learning_rate":[0.05],
             "subsample":[0.9], "colsample_bytree":[0.9], "gamma":[0]},
    "AE":   {"encoding_dim":16, "epochs":25, "batch_size":256},
    "LSTM": {"timesteps":10, "units":32, "epochs":8, "batch_size":128},

    # Monetary parameters for false-positives
    "FP_FLAT": 5.0,      # flat cost per FP
    "FP_PCT" : 0.02,     # % of legitimate amount (e.g. 0.02 = 2 %)
    "OUTPUT_DIR": Path("outputs")
}
CONFIG["OUTPUT_DIR"].mkdir(exist_ok=True)
# ╰──────────────────────────────────────╯


# ╭──────── Convenience ───────╮
def banner(msg, w: int = 70):
    print(f"\n{'='*w}\n{msg}\n{'='*w}")

def timed(fn,*a,**kw):
    st=time.perf_counter(); out=fn(*a,**kw)
    return out,(time.perf_counter()-st)*1e3
# ╰────────────────────────────╯


# ╭──────── Data pipeline ─────╮
def load_dataset(p): return pd.read_csv(p)

def engineer(df):
    df=df.copy()
    df["Amount_log"]=np.log1p(df["Amount"])
    df["Hour"]      =(df["Time"]//3600)%24
    df["Day"]       = df["Time"]//86400
    df["Amt_roll100"]=df["Amount"].rolling(100,min_periods=1).mean()
    return df

FEATS=[f"V{i}" for i in range(1,29)]+["Time","Amount","Amount_log","Hour","Day","Amt_roll100"]
TARGET="Class"

def split_scale(df)->Tuple:
    X=df[FEATS].astype("float32").values
    y=df[TARGET].astype("int8").values
    amt=df["Amount"].astype("float32").values
    X_tr,X_te,y_tr,y_te,amt_tr,amt_te=train_test_split(
        X,y,amt,test_size=CONFIG["TEST_SIZE"],
        stratify=y,random_state=SEED)
    sc=StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te, amt_tr, amt_te
# ╰────────────────────────────╯


# ╭──────── Model builders ────╮
def build_isf():  return RandomizedSearchCV(
        IsolationForest(random_state=SEED,n_jobs=-1),
        CONFIG["ISF"], scoring="roc_auc", n_iter=1,
        cv=StratifiedKFold(CONFIG["CV_FOLDS"],shuffle=True,random_state=SEED),
        n_jobs=-1, random_state=SEED)

def build_xgb():  return RandomizedSearchCV(
        XGBClassifier(objective="binary:logistic",eval_metric="auc",
                      tree_method="hist",random_state=SEED,n_jobs=-1),
        CONFIG["XGB"], scoring="roc_auc", n_iter=1,
        cv=StratifiedKFold(CONFIG["CV_FOLDS"],shuffle=True,random_state=SEED),
        n_jobs=-1, random_state=SEED)

def build_ae(d:int):
    k=CONFIG["AE"]["encoding_dim"]
    inp=Input(shape=(d,))
    h=Dense(k*4,activation="relu")(inp)
    h=Dense(k*2,activation="relu")(h)
    z=Dense(k,activation="relu")(h)
    h=Dense(k*2,activation="relu")(z)
    h=Dense(k*4,activation="relu")(h)
    out=Dense(d,activation="linear")(h)
    m=Model(inp,out); m.compile("adam","mse"); return m

def build_lstm(d:int):
    ts=CONFIG["LSTM"]["timesteps"]
    m=Sequential([LSTM(CONFIG["LSTM"]["units"],input_shape=(ts,d)),
                  Dropout(0.2), Dense(1,activation="sigmoid")])
    m.compile("adam","binary_crossentropy"); return m
# ╰────────────────────────────╯


# ╭──────── Sequence helper ───╮
def make_seq(X,y,ts):
    Xs,ys=[],[]
    for i in range(len(X)-ts):
        Xs.append(X[i:i+ts]); ys.append(y[i+ts])
    return np.array(Xs),np.array(ys)
# ╰────────────────────────────╯


# ╭──────── Evaluation ────────╮
def evaluate(name,y_true,y_pred,y_score,lat_ms,amt_true):
    cm=confusion_matrix(y_true,y_pred)
    tn,fp,fn,tp=cm.ravel()

    fn_loss = amt_true[(y_true==1)&(y_pred==0)].sum()
    fp_flat = fp * CONFIG["FP_FLAT"]
    fp_pct  = amt_true[(y_true==0)&(y_pred==1)].sum() * CONFIG["FP_PCT"]
    fp_cost = fp_flat + fp_pct
    total_cost = fn_loss + fp_cost

    return {"model":name,"TP":tp,"TN":tn,"FP":fp,"FN":fn,
            "precision":precision_score(y_true,y_pred,zero_division=0),
            "recall":   recall_score(y_true,y_pred,zero_division=0),
            "f1":       f1_score(y_true,y_pred,zero_division=0),
            "auc":      roc_auc_score(y_true,y_score),
            "latency_ms":lat_ms,
            "fn_loss":fn_loss,"fp_cost":fp_cost,"total_cost":total_cost}, cm
# ╰────────────────────────────╯


# ╭──────── Plotting ──────────╮
def plot_conf(cm,name):
    tags=np.array([["TN","FP"],["FN","TP"]])
    annot=np.vectorize(lambda l,v:f"{l}\n{v:,}")(tags,cm)
    mask=np.array([[0,1],[1,0]])
    cmap=ListedColormap(["#c7e9c0","#fdd0c9"])
    plt.figure(figsize=(3.8,3.5))
    sns.heatmap(mask,annot=annot,fmt="",cmap=cmap,cbar=False,
                linewidths=.5,linecolor="black",
                xticklabels=["Pred 0","Pred 1"],
                yticklabels=["True 0","True 1"],vmin=0,vmax=1)
    plt.title(f"Confusion – {name}")
    plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_DIR"]/f"cm_{name}.png",dpi=150); plt.close()

def bar(df,col,color,title,fname):
    (df.set_index("model")[col]
       .plot(kind="bar",figsize=(6,4),color=color,legend=False))
    plt.ylabel(col); plt.title(title)
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_DIR"]/fname,dpi=150); plt.close()

def plot_roc(rocs:List[Tuple[str,np.ndarray,np.ndarray]]):
    plt.figure(figsize=(8,6))
    for n,fpr,tpr in rocs:
        plt.plot(fpr,tpr,label=f"{n} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_DIR"]/ "roc_curves.png",dpi=150); plt.close()

def plot_pr(prs:List[Tuple[str,np.ndarray,np.ndarray]]):
    plt.figure(figsize=(8,6))
    for n,y,s in prs:
        p,r,_=precision_recall_curve(y,s); ap=average_precision_score(y,s)
        plt.plot(r,p,label=f"{n} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_DIR"]/ "precision_recall.png",dpi=150); plt.close()

def plot_latency(df):
    plt.figure(figsize=(7,5)); plt.scatter(df["auc"],df["latency_ms"])
    for _,r in df.iterrows():
        plt.annotate(r["model"],(r["auc"],r["latency_ms"]),
                     xytext=(5,5),textcoords="offset points")
    plt.yscale("log")
    plt.xlabel("ROC-AUC"); plt.ylabel("Latency (ms)")
    plt.title("Latency vs Accuracy"); plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_DIR"]/ "latency_vs_auc.png",dpi=150); plt.close()

def plot_gain(gains:List[Tuple[str,np.ndarray,np.ndarray]]):
    plt.figure(figsize=(8,6))
    for n,y,s in gains:
        order=np.argsort(-s); ys=y[order]
        cum=np.cumsum(ys); pct_tx=np.arange(1,len(ys)+1)/len(ys)
        plt.plot(pct_tx,cum/cum[-1],label=n)
    plt.plot([0,1],[0,1],"--",color="grey")
    plt.xlabel("Proportion Reviewed"); plt.ylabel("Proportion Frauds Found")
    plt.title("Cumulative Gain"); plt.legend(); plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_DIR"]/ "cumulative_gain.png",dpi=150); plt.close()
# ╰────────────────────────────╯


# ╭──────── Training routine ───╮
def train_eval(X_tr,X_te,y_tr,y_te,amt_te):
    rows,rocs,prs,gains=[],[],[],[]

    # 1 Isolation Forest
    isf=build_isf().fit(X_tr,y_tr)
    scr,lat=timed(isf.decision_function,X_te)
    y_pred=(scr<0).astype(int)
    row,cm=evaluate("IsolationForest",y_te,y_pred,-scr,lat,amt_te)
    rows.append(row); plot_conf(cm,"IsolationForest")
    rocs.append(("IsolationForest",*roc_curve(y_te,-scr)[:2]))
    prs.append(("IsolationForest",y_te,-scr)); gains.append(("IsolationForest",y_te,-scr))

    # 2 XGBoost
    xgb=build_xgb().fit(X_tr,y_tr).best_estimator_
    proba,lat=timed(xgb.predict_proba,X_te); proba=proba[:,1]
    y_pred=(proba>=0.5).astype(int)
    row,cm=evaluate("XGBoost",y_te,y_pred,proba,lat,amt_te)
    rows.append(row); plot_conf(cm,"XGBoost")
    rocs.append(("XGBoost",*roc_curve(y_te,proba)[:2]))
    prs.append(("XGBoost",y_te,proba)); gains.append(("XGBoost",y_te,proba))

    # 3 Autoencoder
    ae=build_ae(X_tr.shape[1])
    ae.fit(X_tr[y_tr==0],X_tr[y_tr==0],
           epochs=CONFIG["AE"]["epochs"],batch_size=CONFIG["AE"]["batch_size"],
           validation_split=0.1,verbose=0,
           callbacks=[EarlyStopping("val_loss",patience=3,restore_best_weights=True)])
    rec,lat=timed(lambda X:np.mean((X-ae.predict(X,0))**2,1),X_te)
    thr=np.percentile(rec[y_te==0],99)
    y_pred=(rec>=thr).astype(int)
    row,cm=evaluate("Autoencoder",y_te,y_pred,rec,lat,amt_te)
    rows.append(row); plot_conf(cm,"Autoencoder")
    rocs.append(("Autoencoder",*roc_curve(y_te,rec)[:2]))
    prs.append(("Autoencoder",y_te,rec)); gains.append(("Autoencoder",y_te,rec))

    # 4 LSTM
    ts=CONFIG["LSTM"]["timesteps"]
    Xs_tr,ys_tr=make_seq(X_tr,y_tr,ts)
    Xs_te,ys_te=make_seq(X_te,y_te,ts)
    amt_seq=amt_te[ts:]
    lstm=build_lstm(X_tr.shape[1])
    lstm.fit(Xs_tr,ys_tr,
             epochs=CONFIG["LSTM"]["epochs"],batch_size=CONFIG["LSTM"]["batch_size"],
             validation_split=0.1,verbose=0,
             callbacks=[EarlyStopping("val_loss",patience=2,restore_best_weights=True)])
    proba,lat=timed(lambda X:lstm.predict(X,0).ravel(),Xs_te)
    y_pred=(proba>=0.5).astype(int)
    row,cm=evaluate("LSTM",ys_te,y_pred,proba,lat,amt_seq)
    rows.append(row); plot_conf(cm,"LSTM")
    rocs.append(("LSTM",*roc_curve(ys_te,proba)[:2]))
    prs.append(("LSTM",ys_te,proba)); gains.append(("LSTM",ys_te,proba))

    return pd.DataFrame(rows),rocs,prs,gains
# ╰────────────────────────────╯


# ╭──────── Main ─────────────╮
def main():
    banner("LOAD → PREP")
    df=engineer(load_dataset(CONFIG["DATA_PATH"]))
    X_tr,X_te,y_tr,y_te,amt_tr,amt_te=split_scale(df)

    banner("TRAIN → EVALUATE")
    metrics,rocs,prs,gains=train_eval(X_tr,X_te,y_tr,y_te,amt_te)

    banner("RESULTS"); print(metrics.to_string(index=False))
    metrics.to_csv(CONFIG["OUTPUT_DIR"]/ "metrics_summary.csv",index=False)

    plot_roc(rocs); plot_pr(prs); plot_latency(metrics); plot_gain(gains)
    bar(metrics,"fn_loss"  ,"tomato"   ,"Missed-fraud loss"  ,"fn_loss.png")
    bar(metrics,"fp_cost"  ,"gold"     ,"False-positive cost","fp_cost.png")
    bar(metrics,"total_cost","steelblue","Overall monetary cost","total_cost.png")

    banner("DONE – outputs saved to ./outputs")

if __name__=="__main__":
    main()