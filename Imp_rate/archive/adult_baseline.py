from pac_baseline import *

# ═══════════════════════════════════════════════════════════
# RUN ON ADULT DATASET (FIXED — MINIMAL CHANGES ONLY)
# ═══════════════════════════════════════════════════════════

print("\n[Adult] Loading data...")
X, y = load_and_process_adult('adult.csv')

# ───────── LABEL SANITY ─────────
y = np.array(y)

if y.ndim > 1:
    if y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    else:
        y = y.reshape(-1)

if y.dtype != int:
    unique_vals = np.unique(y)
    if len(unique_vals) > 2:
        y = (y > 0.5).astype(int)
    else:
        y = y.astype(int)

y = y.reshape(-1)

print("y shape:", y.shape)
print("unique labels:", np.unique(y))

# ───────── SPLIT ─────────
X_train, X_test, y_train, y_test = data_split(X, y, 0.3)

# Convert EVERYTHING to numpy float32 (CRITICAL FIX)
X_train = np.asarray(X_train, dtype=np.float32)
X_test  = np.asarray(X_test,  dtype=np.float32)

y_train = np.asarray(y_train).reshape(-1).astype(int)
y_test  = np.asarray(y_test).reshape(-1).astype(int)

print("After split:")
print("y_train shape:", y_train.shape, "unique:", np.unique(y_train))
print("y_test shape:", y_test.shape, "unique:", np.unique(y_test))

# ───────── TRAIN f* ─────────
print("[Adult] Training f*...")
fstar_adult = train_fstar(X_train, y_train)

# Train h on f* labels
y_fstar_train = fstar_adult.predict(X_train)

# ───────── TRAIN h ─────────
print("[Adult] Training h model...")
model = train_h(X_train, y_fstar_train, w_fp=1.0, w_fn=1.0)

# Ensure model is on CPU (important if earlier CUDA used)
model = model.cpu()
model.eval()

# ───────── PARAMETERS ─────────
d = X_train.shape[1]
alpha_np = np.ones(d, dtype=np.float32)
beta = 1.0

# ───────── FINAL SAFE NUMPY ─────────
X_test_np = np.asarray(X_test, dtype=np.float32)
y_test_np = np.asarray(y_test, dtype=int)


# ───────── EXTRACT CORRECT LINEAR WEIGHTS ─────────
with torch.no_grad():
    W1 = model.hidden[0].weight      # (64, d)
    W2 = model.out.weight           # (1, 64)

    w_eff = (W2 @ W1).squeeze().cpu().numpy()   # (d,)
    b_eff = model.out.bias.item()

# ───────── CALL USING CORRECT w, b ─────────
w, b, num_manipulated, num_improved, improvement_rate = compute_manipulated_stats_baseline(
    (w_eff, b_eff),   # ← PASS AS TUPLE INSTEAD OF MODEL
    X_test_np,
    y_test_np,
    fstar_adult,
    alpha_np,
    beta
)


def compute_manipulated_stats_baseline(model, X_test, y_test, eta_model, alpha_np, beta):
    
    if isinstance(model, tuple):   # ← ADD THIS
        w, b = model
    else:
        w = model.coef_.flatten()
        b = model.intercept_[0]

    # strategic features
    X_strat, manipulated_indices = get_strategic_response_batch(X_test, w, b, alpha_np, beta)
    
    num_manipulated = len(manipulated_indices)

    probs = eta_model.predict_proba(X_strat)[:,1]
    y_tilde = np.random.binomial(1, probs)

    improved_mask = (y_test == 0) & (y_tilde == 1)

    num_improved = np.sum(improved_mask[manipulated_indices])

    improvement_rate = (num_improved / num_manipulated if num_manipulated > 0 else 0)

    return w, b, num_manipulated, num_improved, improvement_rate

def get_strategic_response_batch(X_batch_np, w_np, b_np, alpha_np, beta):
    batch_size, d = X_batch_np.shape
    X_manip = X_batch_np.copy()
    
    roi = w_np / (alpha_np + 1e-12)
    k_star = np.argmax(roi)
    
    scores = X_batch_np @ w_np + b_np
    neg_indices = np.where(scores < 0)[0]
    
    manipulated_indices = []

    if len(neg_indices) > 0:
        gap = -scores[neg_indices] 
        w_k = w_np[k_star]
        
        if w_k > 1e-8:
            delta_k = gap / w_k
            cost = alpha_np[k_star] * delta_k
            
            do_manipulate = cost <= beta
            
            idx_to_change = neg_indices[do_manipulate]
            delta_to_apply = delta_k[do_manipulate]
            
            X_manip[idx_to_change, k_star] += (delta_to_apply + 1e-5)
            manipulated_indices = idx_to_change.tolist()

    return X_manip, manipulated_indices


print("\n[Adult Results]")
print(f"Num manipulated: {num_manipulated}")
print(f"Num improved: {num_improved}")
print(f"Improvement rate: {improvement_rate:.4f}")