# Workflow-CI

Repository ini berisi implementasi Continuous Integration (CI) dan MLflow Project untuk standarisasi pelatihan model machine learning pada Proyek Akhir Dicoding: **Membangun Sistem Machine Learning**.

## Struktur Direktori
```
Workflow-CI
├── .github/workflows/ci.yml
├── .workflow/ci.yml
├── MLProject/
│   ├── MLproject
│   ├── conda.yaml
│   └── modelling.py
├── namadataset_preprocessing.csv
├── Dockerfile
├── Tautan ke Docker Hub.txt
└── requirements.txt
```

## Menjalankan MLProject
Anda dapat menjalankan pipeline MLflow project dengan:
```bash
cd MLProject
python modelling.py --data_path ../namadataset_preprocessing.csv --n_estimators 100 --max_depth 10
```
atau menggunakan MLflow CLI:
```bash
mlflow run MLProject -P n_estimators=100
```
