import streamlit as st
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns


@st.cache_resource
def load_model_components():
    """
    Carga el modelo y todos los transformadores guardados con joblib.
    """
    scaler = joblib.load(r'scaler_exoplanet.joblib')
    inputer = joblib.load(r'nan_imputer.joblib')

    return scaler, inputer

@st.cache_resource
def load_model_xgb():
    """
    Carga el modelo y todos los transformadores guardados con joblib.
    """
    model = joblib.load(r'xgb_exoplanet.joblib')

    return model




def get_prediction(data:pd.DataFrame,model,scaler, inputer):

    all_obs = data.to_numpy() 

    obs_imputed = inputer.transform(all_obs)

    obs_scaled = scaler.transform(obs_imputed)

    prediction = model.predict(obs_scaled)
    probability_prediction = model.predict_proba(obs_scaled)

    exo_proba = probability_prediction[:, 1]
    non_exo_proba = probability_prediction[:, 0]


    data['prediction'] = prediction
    data['exo_planet_probability'] = exo_proba
    data['non_exo_planet_probability'] = non_exo_proba

    new_first_cols = [
    'prediction', 
    'exo_planet_probability', 
    'non_exo_planet_probability'
    ]

    original_cols = [col for col in data.columns if col not in new_first_cols]

    new_col_order = new_first_cols + original_cols

    data = data[new_col_order]

    return data


#st.set_page_config(layout="wide")



with st.sidebar:
    st.header("Upload prediction file")

    #st.markdown("<br>", unsafe_allow_html=True)
    
    #expected_colums = [
                #'teff', 'radius', 'logg', 'period', 'transit_time_t0',
                #'planet_radius', 'duration', 'depth', 'insolation', 'temperature'
            #]

    #df_features = pd.DataFrame(columns=expected_colums)

    #st.markdown("<br>", unsafe_allow_html=True)

    st.write("Your File needs to include these columns")
    #st.dataframe(df_features)

    st.markdown("""
        **Measurements:**

        - `teff`: Stellar effective temperature (Kelvin)
        - `radius`: Stellar radius (Solar radii)
        - `logg`: Surface gravity (log g)
        - `period`: Orbital period (days)
        - `transit_time_t0`: Transit epoch (BJD - 2457000)
        - `planet_radius`: Planetary radius (Earth radii)
        - `duration`: Transit duration (hours)
        - `depth`: Transit depth (ppm)
        - `insolation`: Insolation flux (Earth units)
        - `temperature`: Planet equilibrium temperature (Kelvin)
        """)
    st.markdown("[See Examples](https://shorturl.at/2dif3)")
 
    st.markdown("<br>", unsafe_allow_html=True)


    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    
    st.markdown("---")
    
    # Botón de predicción
    if st.button("🚀 PREDICT", type="primary"):


        st.session_state['run_prediction'] = True
    else:
        st.session_state['run_prediction'] = False

col1, col2 = st.columns([2, 2])

if st.session_state.get('run_prediction'):
    
    # El spinner envuelve toda la lógica de cálculo
    with st.spinner("Running Predictions and Analysis...", show_time=True): 
        
        try:
            scaler, inputer = load_model_components() 
            model = load_model_xgb()
        except NameError:
            st.error("Error: Asegúrate de que las funciones 'load_model_components' y 'load_model' estén definidas.")

        # 1. Ejecutar Predicciones y Análisis de Datos
        df = pd.read_csv(csv_file) # Asume que 'csv_file' contiene el archivo subido
        df_for_shap = df.copy() # Copia para el análisis SHAP

        # Obtener predicciones
        predictions = get_prediction(df, model, scaler, inputer)
        
        # 2. Análisis de Conteo
        pred_count = predictions['prediction'].value_counts()
        count_exoplanet = pred_count.get(1, 0)
        count_non_exo = pred_count.get(0, 0)
        total_samples = len(predictions)

        # -----------------------------------------------------------
        # Columna 1: DataFrame, Métricas y Descarga
        # -----------------------------------------------------------

        with col1:
            st.markdown("<h2 style='text-align: center;'>💡 Model Explanation</h2>", unsafe_allow_html=True)

            # --- CORRECCIÓN CRUCIAL DE PREPROCESAMIENTO ---
            all_obs = df_for_shap.to_numpy() 

            # 1. Imputar (Necesario para manejar NaN)
            obs_imputed = inputer.transform(all_obs) 

            # 2. Escalar (Necesario para poner los datos en el rango de entrenamiento)
            obs_scaled = scaler.transform(obs_imputed) # <-- ¡CORREGIDO!

            # Preparar DataFrame para SHAP
            X_train_shap = pd.DataFrame(obs_scaled, columns=df_for_shap.columns) 
            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_train_shap)

            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(10, 6))

            shap.summary_plot(shap_values, X_train_shap, show=False, color_bar=False)
            
            
            ax.tick_params(axis='y', colors='white', labelsize=10)
            ax.tick_params(axis='x', colors='white')

            fig.patch.set_facecolor('black')  # Fondo del área fuera del gráfico
            plt.gca().set_facecolor('black')  # Fondo del área del gráfico (donde están los puntos)
            
            st.pyplot(fig)

            st.session_state['show_bot1'] = True
            
      
        with col2:

            #st.markdown("<h2 style='text-align: center;'> 📋 Predicted Data</h2>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'> 📈 Metrics</h2>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            col_metric_1, col_metric_2 = st.columns(2)# metric columns
            
            with col_metric_1:
                st.metric(
                    label="⭐ Exoplanets (Class 1)", 
                    value=f"{count_exoplanet:,}",
                    delta=f"{(count_exoplanet / total_samples) * 100:.1f}%  from total"
                )
            
            with col_metric_2:
                st.metric(
                    label="⚪ Non-Exoplanets (Class 0)",
                    value=f"{count_non_exo:,}",
                    delta=f"{(count_non_exo / total_samples) * 100:.1f}% from total",
                    delta_color="inverse",
                )
            
            st.markdown("<br>", unsafe_allow_html=True)

            SELECTED_FEATURE = "duration"

            fig, ax = plt.subplots(figsize=(8, 4))
            FIG_COLOR = '#0E1117' 
            fig.patch.set_facecolor(FIG_COLOR)
            ax.set_facecolor(FIG_COLOR)
            #ax.tick_params(axis='x', colors='white')  # Ticks del eje X
            #ax.tick_params(axis='y', colors='white')  # Ticks del eje Y
            #ax.set_ylabel(f"{SELECTED_FEATURE}", color='white')


            # Generar el gráfico de densidad (KDE)
            df_plot = df.copy()
            df_plot['prediction'] = df_plot['prediction'].astype(str)
            sns.kdeplot(
                data=df_plot, 
                x=SELECTED_FEATURE, 
                hue='prediction', # Colorear por la etiqueta de predicción
                fill=True, 
                ax=ax,
                palette={'1':  '#6495ED' , '0': '#FF6347'}
            )
            st.pyplot(fig)


            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.session_state['show_bot2'] = True



    if st.session_state['show_bot1'] and st.session_state['show_bot2']:

        # this was changed to not have cost from openai

        mock_answer = """
            This plot shows the global impact of all features on the model's predictions. Y-Axis Order: Features are ranked by importance, with the most influential feature at the top. Color Scale: The color of each point represents the actual feature value (Red = High value, Blue = Low value). X-Axis: The SHAP value indicates the magnitude and direction of the impact: Points to the Right push the prediction toward Exoplanet (Class 1). Points to the Left push the prediction toward Non-Exoplanet (Class 0). By analyzing the color pattern, you can quickly determine how high or low values of a feature affect the final classification.

            """
        
        prediction_agent= st.chat_message("assistant")
        prediction_agent.write(mock_answer)

        new_csv_data = predictions.to_csv(index=False).encode('utf-8')
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
                    label="Download Predictions",
                    data=new_csv_data,
                    file_name='exoplanet_predictions.csv',
                    mime='text/csv',
                    key='centered_download_btn' 
                )
