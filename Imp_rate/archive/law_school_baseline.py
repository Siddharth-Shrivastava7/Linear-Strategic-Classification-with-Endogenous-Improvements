from pac_baseline import *

# ═══════════════════════════════════════════════════════════
# RUN ON LAW SCHOOL DATASET (FIXED — MINIMAL CHANGES ONLY)
# ═══════════════════════════════════════════════════════════

print("\n[Law School] Loading data...")
X, y = load_and_process_lawschool('Imp_rate/law_school_clean.csv')

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
print("[Law School] Training f*...")
fstar_lawschool = train_fstar(X_train, y_train)

# Train h on f* labels
y_fstar_train = fstar_lawschool.predict(X_train)

# ───────── TRAIN h ─────────
print("[Law School] Training h model...")
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
    fstar_lawschool,
    alpha_np,
    beta
)

print("\n[Law School Results]")
print(f"Num manipulated: {num_manipulated}")
print(f"Num improved: {num_improved}")
print(f"Improvement rate: {improvement_rate:.4f}")