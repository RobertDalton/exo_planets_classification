import streamlit as st
import pandas as pd
from autoencoder_pipelone import AnomalyDetector, preprocess_light_curves
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error



@st.cache_resource
def load_autoencoder():
    """
    Carga el modelo y todos los transformadores guardados con joblib.
    """
    MODEL_PATH = r"autoencoder_exo_planet.keras"
    autoencoder = load_model(
        MODEL_PATH, 
        custom_objects={'AnomalyDetector': AnomalyDetector},
        #compile=False
    )


    autoencoder.compile(optimizer='adam', loss='mse') 

    return autoencoder



#st.set_page_config(layout="wide")



with st.sidebar:
    st.header("Upload file")
    
    #df_lightcurve = pd.DataFrame(columns=["id", "time", "flux"])

    #st.write("This DataFrame represents the columns of the light curve.")
    #st.dataframe(df_lightcurve)

    st.markdown("""
    **Light curve columns:**

    - `id`: Candidate or star identifier for id tracking
    - `time`: Observation time (seconds)
    - `flux`: Observed flux
    """)

    st.markdown("[See Examples](https://shorturl.at/2dif3)")

    st.markdown("<br>", unsafe_allow_html=True)


    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    
    st.markdown("---")
    
    # Botón de predicción
    if st.button("Analyze Light Curve", type="primary"):


        st.session_state['run_prediction'] = True
    else:
        st.session_state['run_prediction'] = False


if st.session_state.get('run_prediction'):

    st.markdown("<h2 style='text-align: center;'> 📈 Autoencoder </h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("Running Predictions...", show_time=True): 

        autoencoder = load_autoencoder()
        data_to_pred = pd.read_csv(csv_file)
        processed_df = preprocess_light_curves(data_to_pred,n_bins=1000)
        processed_df = processed_df.drop(columns=['id'],axis=1)
        X_predict = processed_df.values.astype(np.float32)

        encoded_imgs = autoencoder.encoder(X_predict).numpy()
        decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

        normalized = (decoded_imgs[0] - np.mean(decoded_imgs[0])) / np.std(decoded_imgs[0])

        reconstructions = autoencoder.predict(X_predict)
        thresshold = 0.544

        mae = mean_absolute_error(X_predict, reconstructions)

        classification = "Exo Planet Candidate" if mae>thresshold else "Non Exo Planet Candidate"


        fig, ax = plt.subplots(figsize=(10, 4))

        # Fondo oscuro
        fig.patch.set_facecolor('#1e1e1e')     # Fondo fuera del gráfico
        ax.set_facecolor('#2e2e2e')    

        ax.plot(X_predict[0], color='blue', label='Input')
        ax.plot(normalized, color='red', label='Reconstruction')

        # Rellenar área entre curvas
        ax.fill_between(np.arange(1000), normalized, X_predict[0], color='lightcoral', alpha=0.5)

        # Estética de texto y ticks
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.legend(loc="upper right", facecolor='#2e2e2e', edgecolor='white', labelcolor='white')

        # Etiquetas
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Value")

        plt.tight_layout()

        # Mostrar en Streamlit
        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        if classification == "Exo Planet Candidate":
            st.info("Anomaly Detected, this might be an exoplanet!")
        else:
            st.info("This doesn't look like an exoplanet")
 



