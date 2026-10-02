from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

app = Flask(__name__)
app.secret_key = "housing-dashboard-secret"

FEATURES = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]

# ---------- Train the model once at startup ----------
df = pd.read_csv("USA_Housing.csv")
X = df[FEATURES]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
MODEL_STATS = {
    "r2": round(metrics.r2_score(y_test, preds), 4),
    "rmse": round(np.sqrt(metrics.mean_squared_error(y_test, preds)), 2),
    "intercept": round(float(model.intercept_), 2),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "coefficients": dict(zip(FEATURES, [round(float(c), 2) for c in model.coef_])),
    "feature_stats": {
        f: {
            "mean": round(float(df[f].mean()), 2),
            "min": round(float(df[f].min()), 2),
            "max": round(float(df[f].max()), 2),
        }
        for f in FEATURES
    },
}


# ---------- Scatter plot: actual vs predicted ----------
def make_scatter_plot():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, preds, alpha=0.4, color="steelblue", s=12)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted House Prices")
    ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


SCATTER_PLOT = make_scatter_plot()

LAST_UPLOAD = None


@app.route("/", methods=["GET", "POST"])
def dashboard():
    prediction = None
    error = None
    inputs = None

    if request.method == "POST":
        try:
            inputs = {f: float(request.form[f]) for f in FEATURES}
            X_input = pd.DataFrame([inputs], columns=FEATURES)
            X_input_scaled = scaler.transform(X_input)
            prediction = float(model.predict(X_input_scaled)[0])
        except (ValueError, KeyError):
            error = "Please enter valid numeric values for all fields."

    return render_template(
        "dashboard.html",
        stats=MODEL_STATS,
        features=FEATURES,
        prediction=prediction,
        error=error,
        inputs=inputs,
        scatter_plot=SCATTER_PLOT,
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file selected.")
        return redirect(url_for("dashboard"))
    try:
        data = pd.read_csv(file)
    except Exception:
        flash("Could not read the file. Please upload a valid CSV.")
        return redirect(url_for("dashboard"))

    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        flash(f"CSV is missing required columns: {', '.join(missing)}")
        return redirect(url_for("dashboard"))

    try:
        X_new = data[FEATURES].astype(float)
    except ValueError:
        flash("All feature columns must contain numeric values.")
        return redirect(url_for("dashboard"))

    X_new_scaled = scaler.transform(X_new)
    data = data.copy()
    data["Predicted Price"] = model.predict(X_new_scaled).round(2)

    global LAST_UPLOAD
    LAST_UPLOAD = data

    return render_template(
        "upload_result.html",
        tables=[data.head(50).to_html(classes="data", index=False, float_format="%.2f")],
        total=len(data),
        shown=min(len(data), 50),
    )


@app.route("/download")
def download():
    if LAST_UPLOAD is None:
        flash("No upload yet.")
        return redirect(url_for("dashboard"))
    buf = io.BytesIO()
    LAST_UPLOAD.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")


if __name__ == "__main__":
    app.run(debug=True)
