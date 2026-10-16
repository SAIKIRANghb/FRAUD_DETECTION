data:
	PYTHONPATH=src python src/data/load_elliptic.py

train:
	PYTHONPATH=src python src/models/train.py

gnn:
	PYTHONPATH=src python src/models/train_gnn.py

drift:
	PYTHONPATH=src python src/mlops/drift_check.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
