import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importa os dados do arquivo CSV para um DataFrame
df = pd.read_csv('medical_examination.csv')

# 2. Adiciona a coluna 'overweight' (acima do peso) com base no cálculo do IMC
# IMC = peso (kg) / (altura (m))^2. Se IMC > 25, considera acima do peso (1), senão (0)
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3. Normaliza as colunas 'cholesterol' e 'gluc':
# 0 = bom, 1 = ruim. Se valor for 1, vira 0 (bom); se maior que 1, vira 1 (ruim)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4
def draw_cat_plot():
    # 5. Transforma o DataFrame para formato longo (long format) para plotagem categórica
    # Seleciona as colunas de interesse e mantém 'cardio' como variável identificadora
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Agrupa e reordena os dados para contar as ocorrências de cada categoria
    df_cat = (
        df_cat.groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # 7. Cria o gráfico categórico (catplot) usando seaborn
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    # 8. Obtém a figura do gráfico
    fig = g.fig

    # 9. Salva a figura em arquivo
    fig.savefig('catplot.png')
    return fig
def draw_heat_map():
    # 11. Limpa os dados removendo casos inconsistentes:
    # - Pressão diastólica maior que sistólica
    # - Altura e peso fora dos percentis 2.5 e 97.5
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12. Calcula a matriz de correlação entre as variáveis
    corr = df_heat.corr()

    # 13. Gera uma máscara para ocultar a metade superior da matriz (triângulo superior)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Prepara a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Plota o heatmap da matriz de correlação usando seaborn
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,      # Mostra os valores na célula
        fmt='.1f',       # Formato dos números
        center=0,        # Centraliza o gradiente de cor no zero
        vmax=0.32,       # Valor máximo da barra de cor
        vmin=-0.16,      # Valor mínimo da barra de cor
        square=True,     # Mantém as células quadradas
        linewidths=0.5,  # Linhas divisórias
        cbar_kws={'shrink': 0.5}, # Barra de cor menor
        ax=ax
    )

    # 16. Salva a figura em arquivo
    fig.savefig('heatmap.png')
    return fig
