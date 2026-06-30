# Front — Dashboard de Maintenance Prédictive (Streamlit)

Dashboard décisionnel autonome, indépendant de l'API et exploitable par un
profil métier. Il permet de visualiser les données, d'explorer les
indicateurs clés, de comparer les modèles entraînés et d'exécuter des
prédictions sur des scénarios saisis par l'utilisateur.

```
Dashboard (Streamlit) ──HTTP──> API (FastAPI) ──joblib──> XGBoost.pkl
```

Le dashboard n'embarque jamais le modèle : la page **Prédiction** appelle
l'API REST (`/health`, `/model-info`, `/predict`). Les pages **Exploration**
et **Comparaison des modèles** lisent directement les fichiers du dossier
`training/` (données et métriques) et restent donc consultables même si
l'API est arrêtée.

## Structure

```
front/
├── app.py                      # Page d'accueil : KPIs globaux, vue d'ensemble
├── pages/
│   ├── 1_Exploration.py        # Données, distributions, corrélations, filtres
│   ├── 2_Comparaison_modeles.py # Métriques comparées, CV, figures SHAP/ROC/CM
│   └── 3_Prediction.py         # Scénario manuel + lot CSV, recommandation métier
├── utils/
│   ├── api_client.py           # Client HTTP vers l'API (health/model-info/predict)
│   ├── data.py                 # Chargement mis en cache des CSV (training/)
│   ├── ui.py                   # Sidebar de connexion API, rendu des résultats
│   └── config.py               # Chemins, valeurs par défaut, recommandations métier
└── requirements.txt
```

## Installation

```bash
cd front
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Lancer le dashboard

L'API doit être démarrée pour utiliser la page **Prédiction** (voir
`API/README.md`). Les autres pages fonctionnent sans l'API.

```bash
streamlit run app.py
```

Le dashboard s'ouvre sur http://localhost:8501.

Par défaut, le dashboard cherche l'API sur `http://127.0.0.1:8000`. Pour
cibler une autre instance (ex. l'API déployée en production), définissez la
variable d'environnement avant de lancer Streamlit :

```bash
set API_URL=https://prediction.spokayhub.top   # Windows
streamlit run app.py
```

L'URL de l'API peut aussi être changée à tout moment depuis le champ dans la
barre latérale, sans relancer le dashboard.

## Pages

- **Accueil** : KPIs globaux (taux de défaillance, nombre de machines,
  période couverte), répartition des défaillances par type et par machine,
  évolution temporelle du taux de panne.
- **📊 Exploration** : filtres (type de machine, mode opératoire, période),
  distributions des features, boxplots par classe, matrice de corrélation,
  valeurs manquantes.
- **🤖 Comparaison des modèles** : tableau et graphiques comparant Recall
  macro / F1 macro / ROC-AUC / PR-AUC entre les 4 modèles entraînés,
  résultats de validation croisée, infos du modèle actuellement déployé sur
  l'API, et visualisations détaillées (matrice de confusion, ROC, feature
  importance, SHAP).
- **🔮 Prédiction** : formulaire de saisie d'un scénario capteur (avec
  exemples préchargés depuis `API/tests/test_values.json`) ou import d'un
  CSV pour des prédictions par lot. Chaque résultat affiche la classe
  prédite, les probabilités par classe et une recommandation d'action
  métier (ex. "Inspecter les roulements dans les 24-48h").
