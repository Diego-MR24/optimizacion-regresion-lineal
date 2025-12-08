import pandas as pd
import os

try:
    # Carga de la base de datos
    df = pd.read_csv(os.path.join('data', 'raw', 'Fish.csv'))
    print(df.head())

    df = df[['Length1', 'Weight']]
    print(df.head())

    correlation = df['Length1'].corr(df['Weight'])

    # Normalización
    df['Length1_norm'] = (df['Length1'] - df['Length1'].min()) / (df['Length1'].max() - df['Length1'].min())
    df['Weight_norm'] = (df['Weight'] - df['Weight'].min()) / (df['Weight'].max() - df['Weight'].min())

    print(f"Correlación de Pearson : {correlation:.4f}")
    print(f"Número de registros: {len(df)}")

    # Guardar base de datos limpia
    output_path = os.path.join('data', 'processed', 'clean_fish_data.csv')
    df.to_csv(output_path, index=False)

    print("Se limpio correctamente la base de datos")
    print(df.head())

except FileNotFoundError:
    print("Base de datos no encontrado")
    
