# FeatSel_Hyperparam_HIGGS/src/main.py
import pandas as pd
import numpy as np
import os # For file paths
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns # Optional, for visualizations

# Import our custom modules
from src.preprocess import preprocess_data # Import preprocess_data function from src/preprocess.py

def run_experiment():
    print("Proje Başlatılıyor: Feature Selection and Hyperparameter Optimization on HIGGS Dataset")

    # --- 0. Veri Yükleme ve Ön Örnekleme ---
    # Path to the HIGGS.csv.gz file
    data_path = os.path.join('data', 'HIGGS.csv.gz')

    try:
        # GEÇİCİ OLARAK SADECE 5.000 SATIR OKUYACAĞIZ.
        # Bu, geliştirme/test sürecini hızlandırmak içindir ve raporda açıklanacaktır.
        df = pd.read_csv(data_path, compression='gzip', nrows=5000, header=None)
        print(f"'{data_path}' başarıyla okundu. {df.shape[0]} örnek.")
    except FileNotFoundError:
        print(f"HATA: '{data_path}' bulunamadı. Lütfen HIGGS.csv.gz dosyasını 'data/' klasörüne indirdiğinizden emin olun.")
        return # Terminate program if file not found

    # The first column is the target variable (label), others are features
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    print(f"\nVeri Seti Boyutu: {X.shape[0]} örnek, {X.shape[1]} özellik.")
    print(f"Hedef Sınıf Dağılımı (0: Arka Plan, 1: Sinyal):\n{y.value_counts()}")
    print(f"Hedef Sınıf Oranları:\n{y.value_counts(normalize=True).round(2)}")

    # --- 1. Veri Ön İşleme (Aykırı Değerler) ---
    # The preprocess_data function will detect and cap outliers.
    X_preprocessed = preprocess_data(X)
    print("Veri ön işleme tamamlandı: Aykırı değerler işlendi.")

    # --- Modelleme ve Değerlendirme için Hazırlıklar ---
    # Settings for Nested Cross-Validation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Models and hyperparameter ranges (UPDATED AS PER INSTRUCTOR'S DIRECTIVES: NARROWED RANGES)
    models_config = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'param_grid': {'classifier__n_neighbors': [5, 9]} # NARROWED: Only 2 combinations
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            # UPDATED AS PER INSTRUCTOR'S DIRECTIVES: C=0.1 and kernel='linear'
            'param_grid': {'classifier__C': [0.1], 'classifier__kernel': ['linear']}
        },
        'MLP': {
            'model': MLPClassifier(random_state=42, max_iter=500), # max_iter can be increased if convergence issues
            'param_grid': {
                'classifier__hidden_layer_sizes': [(100,)], # NARROWED: Only 1 combination
                'classifier__activation': ['relu'] # NARROWED: Only 1 combination
            }
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'param_grid': {
                'classifier__n_estimators': [100], # NARROWED: Only 1 combination
                'classifier__learning_rate': [0.1], # NARROWED: Only 1 combination
                'classifier__max_depth': [3] # NARROWED: Only 1 combination
            }
        }
    }

    all_results = {} # To store average metrics for both Flowcharts
    all_roc_curves = {} # To store ROC curve data for plotting

    # --- Flowchart A Uygulaması ---
    print("\n--- Flowchart A Uygulaması Başlıyor ---")
    results_A = {} # To store results specific to Flowchart A

    for name, config in models_config.items():
        print(f"\nModel: {name} (Flowchart A)")
        model = config['model']
        param_grid = config['param_grid']

        # Create Pipeline: Scaling -> Feature Selection (k=15 fixed) -> Model
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('feature_selector', SelectKBest(score_func=f_classif, k=15)), # k=15 fixed
            ('classifier', model)
        ])

        # GridSearchCV for hyperparameter optimization in the inner loop
        # We use scoring='roc_auc' as ROC-AUC is important as per project directives.
        # verbose=1 will print progress for each fit.
        grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1, verbose=1)

        outer_fold_metrics = []
        outer_fold_roc_data = [] # List to store ROC data for each outer fold

        # Outer loop of Nested CV
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_preprocessed, y)):
            X_train, X_test = X_preprocessed.iloc[train_idx], X_preprocessed.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f"\n  Outer Fold {fold+1}/{outer_cv.n_splits}:")
            print("  İç döngüde hiperparametre optimizasyonu başlatılıyor...")
            grid_search.fit(X_train, y_train) # Inner CV runs here

            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_inner_score = grid_search.best_score_ # Best average ROC-AUC score from inner CV

            print(f"    En iyi iç döngü parametreleri: {best_params}")
            print(f"    İç döngü en iyi ROC AUC: {best_inner_score:.4f}")

            # Evaluate test performance on the outer fold's test set
            y_pred = best_estimator.predict(X_test)
            # predict_proba gives class probabilities. We need probability of the positive class for ROC-AUC.
            y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }
            outer_fold_metrics.append(metrics)
            print(f"    Dış döngü {fold+1} test metrikleri: ROC AUC={roc_auc:.4f}, Accuracy={accuracy:.4f}")

            # Collect ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            outer_fold_roc_data.append({'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})

        # Calculate average metrics across all outer folds
        avg_metrics = pd.DataFrame(outer_fold_metrics).mean()
        results_A[name] = avg_metrics.to_dict()
        all_roc_curves[f'{name}_Flowchart_A'] = outer_fold_roc_data # Store ROC data for this model and Flowchart

        print(f"\n  Ortalama Dış Döngü Metrikleri (Flowchart A - {name}):")
        print(pd.Series(avg_metrics).round(4))

    all_results['Flowchart_A'] = pd.DataFrame(results_A).T
    print("\n--- Flowchart A Nihai Sonuçları (Ortalama Metrikler) ---")
    print(all_results['Flowchart_A'].round(4))


    # --- Flowchart B Uygulaması ---
    print("\n--- Flowchart B Uygulaması Başlıyor ---")
    results_B = {} # Flowchart B'ye özgü sonuçları depolamak için

    # Özellik seçimi için denenecek k değerleri (örneğin, 10, 15, 20 gibi)
    # Hocanızın "min max ve adım aralığını daraltarak deneyebilirsin" direktifi burada da geçerli.
    # Bu aralığı dar tutarak hızlıca sonuç alalım.
    feature_selection_k_range = [10, 15, 20] # Original was 15, adding a few nearby values.

    for name, config in models_config.items(): # Using the same models_config
        print(f"\nModel: {name} (Flowchart B)")
        model = config['model']
        param_grid_B = config['param_grid'].copy() # Copy param_grid

        # Add 'k' parameter for feature selection to param_grid
        # 'feature_selector__k' targets the 'k' parameter of the 'feature_selector' step in the pipeline.
        param_grid_B['feature_selector__k'] = feature_selection_k_range

        # Create Pipeline: Scaling -> Feature Selection -> Model
        # 'k' value will be determined by GridSearchCV here
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('feature_selector', SelectKBest(score_func=f_classif)), # 'k' will be optimized here
            ('classifier', model)
        ])

        # GridSearchCV for hyperparameter and feature selection optimization in the inner loop
        grid_search = GridSearchCV(pipeline, param_grid_B, cv=inner_cv, scoring='roc_auc', n_jobs=-1, verbose=1)

        outer_fold_metrics = []
        outer_fold_roc_data = [] # List to store ROC data for each outer fold

        # Outer loop
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_preprocessed, y)):
            X_train, X_test = X_preprocessed.iloc[train_idx], X_preprocessed.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f"\n  Outer Fold {fold+1}/{outer_cv.n_splits}:")
            print("  İç döngüde hiperparametre ve özellik seçimi optimizasyonu başlatılıyor...")
            grid_search.fit(X_train, y_train) # Inner CV runs here

            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_inner_score = grid_search.best_score_ # Best average ROC-AUC score from inner CV

            print(f"    En iyi iç döngü parametreleri: {best_params}")
            print(f"    İç döngü en iyi ROC AUC: {best_inner_score:.4f}")

            # Evaluate test performance on the outer fold's test set
            y_pred = best_estimator.predict(X_test)
            y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }
            outer_fold_metrics.append(metrics)
            print(f"    Dış döngü {fold+1} test metrikleri: ROC AUC={roc_auc:.4f}, Accuracy={accuracy:.4f}")

            # Collect ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            outer_fold_roc_data.append({'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})

        # Calculate average metrics across all outer folds
        avg_metrics = pd.DataFrame(outer_fold_metrics).mean()
        results_B[name] = avg_metrics.to_dict()
        all_roc_curves[f'{name}_Flowchart_B'] = outer_fold_roc_data # Store ROC data for this model and Flowchart

        print(f"\n  Ortalama Dış Döngü Metrikleri (Flowchart B - {name}):")
        print(pd.Series(avg_metrics).round(4))

    all_results['Flowchart_B'] = pd.DataFrame(results_B).T
    print("\n--- Flowchart B Nihai Sonuçları (Ortalama Metrikler) ---")
    print(all_results['Flowchart_B'].round(4))


    # --- Sonuçların Raporlanması ve Görselleştirilmesi ---
    print("\n--- Raporlama ve Görselleştirme Başlatılıyor ---")

    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True) # Create "reports" folder if it doesn't exist

    # 1. Save Metric Tables
    if 'Flowchart_A' in all_results:
        print("\nFlowchart A Ortalama Metrikleri:")
        print(all_results['Flowchart_A'].round(4))
        all_results['Flowchart_A'].to_csv(os.path.join(reports_dir, 'metrics_table_FlowchartA.csv'))
        print(f"Flowchart A metrikleri '{reports_dir}/metrics_table_FlowchartA.csv' olarak kaydedildi.")
    else:
        print("Flowchart A sonuçları henüz hesaplanmadı veya mevcut değil.")

    if 'Flowchart_B' in all_results:
        print("\nFlowchart B Ortalama Metrikleri:")
        print(all_results['Flowchart_B'].round(4))
        all_results['Flowchart_B'].to_csv(os.path.join(reports_dir, 'metrics_table_FlowchartB.csv'))
        print(f"Flowchart B metrikleri '{reports_dir}/metrics_table_FlowchartB.csv' olarak kaydedildi.")
    else:
        print("Flowchart B sonuçları henüz hesaplanmadı veya mevcut değil.")

    # Create combined comparison table if both Flowcharts have results
    if 'Flowchart_A' in all_results and 'Flowchart_B' in all_results:
        combined_df = pd.concat([
            all_results['Flowchart_A'].add_suffix('_A'),
            all_results['Flowchart_B'].add_suffix('_B')
        ], axis=1)
        print("\nKombine Model Performans Karşılaştırması:")
        print(combined_df.round(4))
        combined_df.to_csv(os.path.join(reports_dir, 'metrics_table_Combined.csv'))
        print(f"Kombine metrikler '{reports_dir}/metrics_table_Combined.csv' olarak kaydedildi.")
    elif 'Flowchart_A' in all_results or 'Flowchart_B' in all_results:
        print("\nTek bir Flowchart için sonuçlar mevcut. Kombine tablo oluşturulamadı.")


    # 2. Plot and Save ROC Curves
    if all_roc_curves:
        plt.figure(figsize=(12, 8))
        for model_id, fold_roc_data in all_roc_curves.items():
            if fold_roc_data:
                # Calculate average AUC score for the label
                avg_auc = np.mean([d['auc'] for d in fold_roc_data])

                # For simplicity and readability on a single plot,
                # we'll plot the ROC curve from the last outer fold for each model/flowchart combo,
                # and use the average AUC in the legend.
                # A more rigorous approach for "average ROC curve" would involve
                # interpolating and averaging FPR/TPR across all folds.
                last_fold_fpr = fold_roc_data[-1]['fpr']
                last_fold_tpr = fold_roc_data[-1]['tpr']
                plt.plot(last_fold_fpr, last_fold_tpr, label=f'{model_id} (AUC = {avg_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Rastgele Sınıflandırıcı (AUC = 0.5)')
        plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
        plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
        plt.title('ROC Eğrileri Karşılaştırması')
        plt.legend(loc='lower right', fontsize='small')
        plt.grid(True)
        roc_curve_path = os.path.join(reports_dir, 'roc_curves_comparison.png')
        plt.savefig(roc_curve_path)
        print(f"ROC eğrileri '{roc_curve_path}' olarak kaydedildi.")
        plt.show() # Display the plot on screen
    else:
        print("ROC eğrisi verisi mevcut değil.")

    print("\nRaporlama ve görselleştirme tamamlandı.")

if __name__ == "__main__":
    run_experiment()