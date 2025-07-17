import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==== 1. Daten laden ====
df = pd.read_csv("Marktkampagne_ohne_NaN_Einkommen.csv")  # Deine Datei nach Vorverarbeitung
df = df.dropna()
# Gesamtausgaben berechnen
ausgaben_spalten = [
    "Ausgaben_Wein", "Ausgaben_Obst", "Ausgaben_Fleisch", "Ausgaben_Fisch",
    "Ausgaben_Süßigkeiten", "Ausgaben_Gold"
]
df["Gesamtausgaben"] = df[ausgaben_spalten].sum(axis=1)
df["Alter"] = 2025 - df["Geburtsjahr"]

# ==== 2. Modelle trainieren ====

# Regressionsmodell
X_reg = df[["Einkommen","Alter", "Kinder_zu_Hause", "Teenager_zu_Hause", "Antwort_Letzte_Kampagne"]]
y_reg = df["Gesamtausgaben"]
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_reg, y_reg)

# Klassifikationsmodell
X_clf = df[["Einkommen", "Alter", "Kinder_zu_Hause", "Teenager_zu_Hause", "Gesamtausgaben"]]
y_clf = df["Antwort_Letzte_Kampagne"]
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_clf, y_clf)

# ==== 3. Dashboard Setup ====
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("📊 Kundenanalyse-Dashboard zur Marktkampagne"),
    
    html.Div([
        html.Label("Alter (max):"),
        dcc.Slider(min=18, max=100, step=1, value=60, id="alter_slider"),

        html.Label("Einkommen (max):"),
        dcc.Slider(min=0, max=150000, step=1000, value=75000, id="einkommen_slider"),

        html.Label("Anzahl Kinder (gesamt):"),
        dcc.Slider(min=0, max=5, step=1, value=2, id="kinder_slider"),
    ], style={"width": "40%", "padding": "10px"}),

    dcc.Graph(id="ausgaben_histogramm"),

    html.Div(id="prognose_output", style={"padding": "20px", "fontSize": "18px", "fontWeight": "bold"})
])

# ==== 4. Callback ====
@app.callback(
    [Output("ausgaben_histogramm", "figure"),
     Output("prognose_output", "children")],
    [Input("alter_slider", "value"),
     Input("einkommen_slider", "value"),
     Input("kinder_slider", "value")]
)
def update_dashboard(max_alter, max_einkommen, kinderanzahl):
    # Filterdaten
    gefiltert = df[
        (df["Alter"] <= max_alter) &
        (df["Einkommen"] <= max_einkommen) &
        ((df["Kinder_zu_Hause"] + df["Teenager_zu_Hause"]) == kinderanzahl)
    ]

    # Histogramm
    fig = px.histogram(gefiltert, x="Gesamtausgaben", nbins=30,
                       title="Verteilung der Gesamtausgaben")

    # Prognose für Reaktion
    if not gefiltert.empty:
        X_pred = gefiltert[["Einkommen", "Alter", "Kinder_zu_Hause", "Teenager_zu_Hause", "Gesamtausgaben"]]
        y_pred = rf_clf.predict(X_pred)
        n_reagiert = sum(y_pred)

        # Prognose für Ausgaben
        X_pred_reg = gefiltert[["Einkommen", "Alter", "Kinder_zu_Hause", "Teenager_zu_Hause", "Antwort_Letzte_Kampagne"]]
        y_pred_ausgabe = rf_reg.predict(X_pred_reg)
        ausgabe_summe = round(np.sum(y_pred_ausgabe), 2)

        prognose = f"📦 Erwartete Rückmeldungen: {n_reagiert} Kunden | 💰 Prognostizierte Gesamtausgaben: {ausgabe_summe} €"
    else:
        prognose = "⚠️ Keine passenden Kunden im Filterbereich."

    return fig, prognose

# ==== 5. Starten ====
if __name__ == "__main__":
    app.run(debug=True)

