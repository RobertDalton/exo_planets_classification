
# -------------------- CONTENIDO DE ae_utils.py --------------------
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Tuple

from keras.models import Model
from keras import layers

# Define la dimensión de entrada
INPUT_DIM = 1000
HE_INIT = 'he_uniform' 
EPSILON = 1e-6 


class AnomalyDetector(Model):
    # ⭐️ CLAVE 1: Aceptar argumentos genéricos (**kwargs) en el constructor
    # Esto captura 'trainable', 'dtype', etc., que Keras pasa al cargar.
    def __init__(self, **kwargs): 
        super(AnomalyDetector, self).__init__(**kwargs)
        
        # --- ENCODER ---
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu", input_shape=(INPUT_DIM,), kernel_initializer=HE_INIT),
            layers.Dense(16, activation="relu", kernel_initializer=HE_INIT),
            layers.Dense(14, activation="relu", kernel_initializer=HE_INIT) 
        ])

        # --- DECODER ---
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu", kernel_initializer=HE_INIT),
            layers.Dense(32, activation="relu", kernel_initializer=HE_INIT),
            layers.Dense(INPUT_DIM, activation="linear")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    # ⭐️ CLAVE 2: Sobrescribir get_config() para la serialización.
    # Necesitas esto si tu constructor tiene argumentos personalizados,
    # aunque en este caso solo lo defines por buena práctica con **kwargs.
    #def get_config(self):
        # Obtiene la configuración base de la clase Model
        config = super(AnomalyDetector, self).get_config()
        # Puedes añadir aquí argumentos específicos de tu clase si los tuvieras
        # config.update({"my_custom_arg": self.my_custom_arg}) 
        #return config
    
def preprocess_light_curves(df_raw: pd.DataFrame, n_bins: int = 1000) -> pd.DataFrame:
    """
    Normaliza el flujo, aplica binning al tiempo, pivotea el DataFrame para el Autoencoder,
    e imputa NaN con 0.
    
    Args:
        df_raw: DataFrame con columnas ['id', 'time', 'flux'].
        n_bins: Número de bins fijos para la longitud de la serie de tiempo.
        
    Returns:
        DataFrame listo para el entrenamiento del Autoencoder.
    """
    # 1. Normalización del Flujo por cada ID
    
    df_normalized = df_raw.copy()
    
    # Normalización: (Flujo - Mediana) / Desviación Estándar
    df_normalized['flux_normalized'] = df_normalized.groupby('id')['flux'].transform(
        lambda x: (x - x.median()) / x.std()
    )
    
    # 🌟 TRATAMIENTO DE NAN 1: Imputar 0 después de la normalización.
    # Esto maneja los casos donde std=0 (línea plana) o datos anómalos.
    df_normalized['flux_normalized'] = df_normalized['flux_normalized'].fillna(0)
    
    # 2. Binning del Tiempo (Crear el índice de columna fija)

    def calculate_bins(series: pd.Series, max_bins: int) -> pd.Series:
        """Calcula el índice del bin [0, max_bins-1] para cada punto de tiempo."""
        time_min = series.min()
        time_max = series.max()
        time_range = time_max - time_min
        
        if time_range == 0 or pd.isna(time_range):
            return pd.Series(0, index=series.index)
        
        # Mapea el tiempo relativo [0, 1] al índice del bin [0, N_BINS-1]
        bin_index = ((series - time_min) / time_range) * (max_bins - 1)
        return np.floor(bin_index).astype(int)

    # Aplica la función de binning para obtener la columna 'bin_index'
    df_normalized['bin_index'] = df_normalized.groupby('id')['time'].transform(
        lambda x: calculate_bins(x, n_bins)
    )
    
    # 3. Agregación (Remuestreo) y Pivot
    
    # Agrupar por ID y BIN_INDEX y tomar la media del flujo normalizado
    df_binned = df_normalized.groupby(['id', 'bin_index'])['flux_normalized'].mean().reset_index()
    
    # Pivotear la tabla: 'id' como índice, 'bin_index' como columnas
    df_final = df_binned.pivot(index='id', columns='bin_index', values='flux_normalized')
    
    # 🌟 TRATAMIENTO DE NAN 2: Rellenar los NaNs restantes.
    # Estos NaNs se generan porque algunos bins no tenían ninguna observación.
    df_final = df_final.fillna(0)

    expected_bins = range(n_bins)

    # Renombrar las columnas para tener nombres limpios (ej. 'flux_0', 'flux_1'...)
    #df_final.columns = [f'flux_{i}' for i in range(n_bins)]

    # Aplica la reindexación al eje de columnas (axis=1)
    # Esto asegura que el DataFrame tenga EXACTAMENTE 1000 columnas (0 a 999)
    df_final = df_final.reindex(columns=expected_bins, fill_value=0)

    # ⭐️ AHORA, la asignación de nombres es segura:
    df_final.columns = [f'flux_{i}' for i in range(n_bins)]

    
    # Resetear el índice si quieres 'id' como una columna normal
    df_final = df_final.reset_index()

    return df_final