# FeatSel_Hyperparam_HIGGS/src/preprocess.py
import pandas as pd
import numpy as np

def detect_and_cap_outliers(df_column):
    """
    Bir Pandas Serisi'ndeki (sütun) IQR yöntemi kullanarak aykırı değerleri tespit eder
    ve alt/üst sınırlara sabitler (capping).

    Args:
        df_column (pd.Series): İşlenecek veri sütunu.

    Returns:
        pd.Series: Aykırı değerleri işlenmiş veri sütunu.
    """
    Q1 = df_column.quantile(0.25)
    Q3 = df_column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Aykırı değerleri tespit et
    outliers_lower = df_column < lower_bound
    outliers_upper = df_column > upper_bound

    # Aykırı değerleri sınırlara sabitle (capping)
    df_column_processed = df_column.copy()
    df_column_processed.loc[outliers_lower] = lower_bound
    df_column_processed.loc[outliers_upper] = upper_bound

    num_outliers = outliers_lower.sum() + outliers_upper.sum()
    if num_outliers > 0:
        print(f"  '{df_column.name}' sütununda {num_outliers} ({num_outliers / len(df_column) * 100:.2f}%) aykırı değer tespit edildi ve sınırlandı.")

    return df_column_processed

def preprocess_data(X_df):
    """
    Veri çerçevesindeki tüm sayısal sütunlardaki aykırı değerleri işler.
    MinMaxScaler, pipeline içinde kullanılacağı için burada uygulanmaz.

    Args:
        X_df (pd.DataFrame): Özellikleri içeren veri çerçevesi.

    Returns:
        pd.DataFrame: Aykırı değerleri işlenmiş özellik veri çerçevesi.
    """
    print("Veri ön işleme başlatılıyor (Aykırı değer tespiti ve sabitleme)...")
    X_processed = X_df.copy()
    for col in X_processed.columns:
        if pd.api.types.is_numeric_dtype(X_processed[col]): # Sadece sayısal sütunları işle
            X_processed[col] = detect_and_cap_outliers(X_processed[col])
        else:
            print(f"  Uyarı: '{col}' sütunu sayısal değil, aykırı değer işleme atlandı.")
    print("Veri ön işleme tamamlandı.")
    return X_processed

if __name__ == "__main__":
    # Bu bölüm, fonksiyonları test etmek için kullanılabilir.
    # Gerçek projede main.py'den çağrılacaktır.
    print("preprocess.py doğrudan çalıştırıldı. Bir test verisi oluşturuluyor...")
    test_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 2, 3, 4, -50],
        'feature2': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    }
    test_df = pd.DataFrame(test_data)
    print("Orijinal Test Verisi:")
    print(test_df)

    processed_test_df = preprocess_data(test_df)
    print("\nİşlenmiş Test Verisi:")
    print(processed_test_df)