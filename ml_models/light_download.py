import lightkurve as lk
import matplotlib as plt
import plotly.express as px
import numpy as np
import pandas as pd

k2 = pd.read_csv(r'nasa_data\k2.csv')
kepler = pd.read_csv(r'nasa_data\kepler.csv')
tess = pd.read_csv(r'nasa_data\tess.csv')


k2_dict = {
    "id": "tic_id",
    "ra": "ra",
    "dec": "dec",
    "teff": "st_teff",
    "radius": "st_rad",
    "logg": "st_logg",
    "period": "pl_orbper",
    "transit_time_t0": "pl_tranmid",
    "planet_radius": "pl_rade",
    "duration": "pl_trandur",
    "depth": "pl_trandep",
    "insolation": "pl_insol",
    "temperature": "pl_eqt",
    "planet_count": "sy_pnum",
    "disposition": "disposition"
}

tess_dict = {
    "id": "tid",
    "ra": "ra",
    "dec": "dec",
    "teff": "st_teff",
    "radius": "st_rad",
    "logg": "st_logg",
    "period": "pl_orbper",
    "transit_time_t0": "pl_tranmid",
    "planet_radius": "pl_rade",
    "duration": "pl_trandurh",
    "depth": "pl_trandep",
    "insolation": "pl_insol",
    "temperature": "pl_eqt",
    "planet_count": "pl_pnum",
    "disposition": "tfopwg_disp"
}

kepler_dict = {
    "id": "kepid",
    "ra": "ra",
    "dec": "dec",
    "teff": "koi_steff",
    "radius": "koi_srad",
    "logg": "koi_slogg",
    "period": "koi_period",
    "transit_time_t0": "koi_time0bk",
    "planet_radius": "koi_prad",
    "duration": "koi_duration",
    "depth": "koi_depth",
    "insolation": "koi_insol",
    "temperature": "koi_teq",
    "planet_count": "koi_count",
    "disposition": "koi_pdisposition"
}


def standarize_dataframe(df,columns):

    df = df[columns]
    df.columns = list(tess_dict.keys())

    return  df
    
    

def get_samples(df, column, columns,n):

    df_filtrado = df[df[column].isin(columns)]
    samples = df_filtrado.sample(n=n, random_state=42)

    return samples

def generate_light_curve_df(id:str,author:str,mission:str):
    time, flux = extract_light_curve(str(id),author,mission)
    if len(time)>0:
        df =  pd.DataFrame({
            "id": [str(id)] * len(flux),
            "time": time,
            "flux": flux
        })

    else:
        print(f"Failed generation on {id}")
        df =  pd.DataFrame({
            "id": str(id) ,
            "time": [],
            "flux": []
        })

    print("Dataframe generated")

    return df
                

def extract_light_curve(id:str,author:str,mission)-> tuple :
    id_prefix  = 'TIC' if author == 'SPOC' or author == 'K2' else 'KIC'
    print("Processing light curve")

    try:
        available_data = lk.search_lightcurve(id_prefix + ' ' + id,exptime='short' ,author=author,mission=mission)
        lc = available_data.download()
        time = np.array(lc.time.value)
        flux = np.array(lc.flux.value)
    except:
        time = np.array([])
        flux = np.array([])
        print(f"could not get data for id {id}")
    
    return  time,flux


def process_light_curves(dataset,author,mission):
    df_total = pd.DataFrame(columns=["id", "time", "flux"])

    for i, id in enumerate(dataset.id):
        print(f"Processing sample {i}")
        df_temp = generate_light_curve_df(str(id),author,mission)
    
        df_total = pd.concat([df_total,df_temp], ignore_index=True)
    
    print("Generating csv")
    df_total.to_csv(f"{mission}_lightcurves.csv",index=False)


#let's try tess first
tess_standarized = standarize_dataframe(tess, list(tess_dict.values()))
df_tess_non_exo = get_samples(tess_standarized,'disposition',['FP'],100) 
df_tess_exo = get_samples(tess_standarized,'disposition',['CP'],100)

kepler_standarized = standarize_dataframe(kepler, list(kepler_dict.values()))
df_kepler_non_exo = get_samples(kepler_standarized,'disposition',['FALSE POSITIVE'],10)
df_kepler_exo = get_samples(kepler_standarized,'disposition',['CANDIDATE'],50)

#process_non_exo 

process_light_curves(df_tess_non_exo,"SPOC","TESS")
#process_light_curves(df_kepler_non_exo,"Kepler","Kepler")


"""

df_kepler_non_exo = get_samples(kepler,kepler_dict.get('disposition'),['FALSE POSITIVE'],50)
df_k2_non_exo = get_samples(k2,k2_dict.get('disposition'),['FALSE POSITIVE'],50)

df_kepler_exo = get_samples(kepler,kepler_dict.get('disposition'),['CANDIDATE'],50)
df_k2_exo = get_samples(k2,k2_dict.get('disposition'),['CONFIRMED'],50)


"""


