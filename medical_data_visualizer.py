import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos
df = pd.read_csv('medical_examination.csv')

# Agregar columna 'overweight' (sobrepeso)
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# Normalizar datos (0 para buenos, 1 para malos)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Dibujar el gráfico categórico
def draw_cat_plot():
    # Convertir datos a formato largo
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Dibujar el gráfico usando seaborn's catplot()
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count', height=5, aspect=1)
    fig.set_axis_labels('variable', 'total')
    fig = fig.fig
    
    return fig

# Dibujar el mapa de calor
def draw_heat_map():
    # Limpiar los datos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # Calcular la matriz de correlación
    corr = df_heat.corr()

    # Generar una máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurar la figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibujar el mapa de calor
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', ax=ax)

    return fig

# Ejecutar el código para probar el gráfico categórico
if __name__ == "__main__":
    cat_plot_fig = draw_cat_plot()
    heat_map_fig = draw_heat_map()
    cat_plot_fig.savefig('catplot.png')
    heat_map_fig.savefig('heatmap.png')
