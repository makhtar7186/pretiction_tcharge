*`PREDICTION TAUX DE CHARGE`*

-   Un projet de prédiction du taux de charge pour chaque names, toutes les 5 minutes, à partir des données récupérées dans ElasticSearch.
-   Le projet utilise un pipeline de prétraitement, un modèle LSTM et une classe pour interroger ElasticSearch.


==> 🚀 `Objectifs`
-   Prédire la charge (target) pour chaque names toutes les 5 minutes.
-   Utiliser les 288 dernières valeurs de chaque names pour alimenter le modèle.
-   Sauvegarder les prédictions ou les afficher.

==> 📂 `Structure du projet`

tauxcharge/
├── src/
│    ├── tauxcharge/   
│    │   ├── __init__.py
│    │   ├── __main__.py
│    │   ├── preprocessing/
│    │   │   ├── sequence_generator.py
│    │   │   ├── splitter.py
│    │   │   ├── custom_preprocessor.py
│    │   ├── model/
│    │   │   ├── regressor.py
│    │   │   ├── all_model/ #dossier ou se trouve l'ensemble des models sauvegarder
│    │   ├── elastic/
│    │   │   └── elasticsearch.py
│    │
├── tests/
│   └── test_pipeline.py
│
├── .env
├── pyproject.toml
├── poetry.lock
├── README.md


=> 🧪 `Lancer les tests`

exécuter  la commande :
-> poetry run pytest -v -s tests/test_pipeline.py  

en exécutant  cette commande tu vas executer le pipeline de test . en effet le pipeline consiste a charger les donnees au niveau de l'index elastiques , faire les pretraitements necessaire , faire une prediction sur cette sequence en suite faire une evaluation des resultats obtenue .


==> 🖥️ `Exécution`

Le point d’entrée est le __main__.py. Il peut être lancé ainsi :


poetry run python -m prediction_tauxcharge
Il exécutera :

-   La récupération des dernières données d’ElasticSearch.
-   La génération des séquences (288 valeurs) pour chaque names.
-   La prédiction du modèle LSTM.
-   L’affichage ou l’enregistrement des résultats.


==> 📄 `Modules`

elastic/
└── elasticsearch.py
-   ElasticFetcher() -> permet de faire de la recuperation de donnees sur elastic
    -   fetch_all() ➝ récupère toutes les valeurs pour tous les names.
    -   fetch_latest_per_name() ➝ récupère uniquement la dernière valeur pour chaque names.


├── preprocessing/
└── sequence_generator.py
-   SequenceGenerator() ->  crée des séquences temporelles (288 pas).

└── splitter.py
-   TrainTestSplitter() ->  divise le jeu de données en train/test.

└── name_encoder.py
-   NameEncoder() -> permet de faire l'encodage de la classe names
└── name_id_mapping.csv
-   mapping des names avec leurs id.

└── custom_preprocessor.py
-   CustomPreprocessor() -> permet de faire le nettoyage des donnnes 


├── model/
└── lstm_regressor.py
-   KerasLSTMRegressor()
    -   Entraîne et charge un modèle LSTM.
    -   Fonction load_model_lstm() pour charger un modèle existant.

└── model_lstm_multiseries.h5
-   Modèle LSTM deja entrainer pour la prédiction des taux de charge

├── pipeline.py 
-   preprocessing_pipe() -> est le pipeline de pretraitement des donnees d'entrer 


**  `reproduction du workflow`
-   Unzip le projet
-   Placez vous sur le repertoire du projet cd/vers/le/projet
-   Taper la commande reproducibilité `poetry install`
-   Creer un sous-shell ```poetry shell```
-   Ensuite lance l'execution# prediction_taux_charge
