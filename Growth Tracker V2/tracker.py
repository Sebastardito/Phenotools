#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:44:48 2024
@author: Sebastardito!
"""

import sys
import importlib
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from flask import Flask, render_template, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from io import BytesIO
import base64
import webbrowser
import seaborn as sns
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import concurrent.futures
import logging
from functools import lru_cache
import time
import re
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.platypus.flowables import KeepTogether

# Configuración de características opcionales
DEFAULT_STD_DEV = 2
MAX_STD_DEV = 4
DEFAULT_SHOW_ANNOTATIONS = False
DEFAULT_SHOW_CORRELATIONS = True

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Verificar dependencias
required_libraries = ['flask', 'pandas', 'matplotlib', 'scipy', 'seaborn', 'statsmodels', 'reportlab']

def check_dependencies():
    missing_libraries = []
    for lib in required_libraries:
        try:
            importlib.import_module(lib)
        except ImportError:
            missing_libraries.append(lib)
    return missing_libraries

missing = check_dependencies()
if missing:
    print("The following libraries are required but not installed:")
    for lib in missing:
        print(lib)
    print("Please install the missing libraries using 'pip install <library_name>' and then try again.")
    sys.exit(1)

# Configuración de Flask
app = Flask(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
template_path = 'index.html'
df = None
variable_columns = []
metrics_cache = {}

# Función para cargar datos con caché
@lru_cache(maxsize=1)
def load_data(csv_file):
    global variable_columns
    logger.info(f"Loading data from: {csv_file}")
    start_time = time.time()
    
    try:
        df = pd.read_csv(csv_file, delimiter=';')
        
        # Convertir columnas numéricas
        age_days_idx = df.columns.get_loc('Age_Days')
        numeric_cols = df.columns[age_days_idx+1:]
        
        for col in ['Age_Days'] + list(numeric_cols):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        variable_columns = list(numeric_cols)
        
        # Preprocesamiento adicional
        df['combined'] = df['Population'] + '-' + df['Cycle'] + '-' + df['Group']
        
        logger.info(f"Data loaded successfully in {time.time() - start_time:.2f} seconds")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_base_plot(selected_populations, selected_cycles, selected_groups, variable, show_annotations):
    plt.figure(figsize=(12, 8))
    y_max_values = []
    average_groups = not bool(selected_groups)
    has_data = False

    # Plotear datos seleccionados
    for population in selected_populations:
        for cycle in selected_cycles:
            groups_to_process = selected_groups if not average_groups else ['All Groups']
            
            for group in groups_to_process:
                if average_groups:
                    data = df[(df['Population'] == population) & 
                            (df['Cycle'] == cycle)]
                else:
                    data = df[(df['Population'] == population) & 
                            (df['Cycle'] == cycle) & 
                            (df['Group'] == group)]
                
                data = data[data['Age_Days'] <= 40]
                if not data.empty:
                    has_data = True
                    data = data.groupby('Age_Days')[variable].mean().reset_index()
                    last_point = data.iloc[-1]
                    last_data_date = df[(df['Population'] == population) & 
                                      (df['Cycle'] == cycle) &
                                      (df['Age_Days'] == last_point['Age_Days'])]['Data_Date'].iloc[0]
                    
                    label = f'{population}, {cycle}, {group}' if not average_groups else f'{population}, {cycle}, All Groups'
                    label += f', Last Data: {last_data_date}'
                    
                    line = plt.plot(data['Age_Days'], data[variable], label=label, marker='o', linewidth=2)[0]
                    y_max_values.append(data[variable].max())
                    
                    if show_annotations:
                        plt.annotate(f"{last_data_date}\n({int(last_point['Age_Days'])} Days)",
                                   (last_point['Age_Days'], last_point[variable]),
                                   xytext=(5, 5), textcoords='offset points',
                                   ha='left', va='bottom',
                                   color=line.get_color(),
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Configuración común del gráfico
    plt.xlabel('Age (Days)', fontsize=12)
    plt.ylabel(variable, fontsize=12)
    plt.title(f'{variable} Growth Curves', fontsize=16, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Solo agregar leyenda si hay datos
    if has_data:
        plt.legend(loc='best', fontsize=10)
    
    # Ajustar límites del eje Y si hay valores
    if y_max_values:
        y_max = max(y_max_values) * 1.1
        plt.ylim(bottom=0, top=y_max)
    else:
        y_max = 0
    
    plt.tight_layout()
    
    return plt, y_max

def add_std_deviation(plt, variable, num_std, y_max):
    # Filtrar datos válidos para desviación estándar
    valid_data = df[(df['Age_Days'] <= 40) & (df[variable].notna())]
    if valid_data.empty:
        return plt
    
    avg_data = valid_data.groupby('Age_Days')[variable].mean()
    std_dev = valid_data.groupby('Age_Days')[variable].std()
    
    plt.plot(avg_data.index, avg_data.values, color='black', linestyle='--', linewidth=1.5, label='Overall Mean')
    
    if num_std > 0:
        colors = ['blue', 'green', 'orange', 'red']
        num_std = min(num_std, MAX_STD_DEV)
        std_max = (avg_data + num_std * std_dev).max()
        
        # Ajustar y_max solo si hay datos válidos
        if not np.isnan(std_max):
            if y_max == 0:
                y_max = std_max * 1.1
            else:
                y_max = max(y_max, std_max * 1.1)
            
            for i in range(1, num_std + 1):
                # Áreas de desviación
                if i == 1:
                    plt.fill_between(avg_data.index, 
                                   avg_data - i*std_dev, 
                                   avg_data + i*std_dev,
                                   color=colors[i-1], alpha=0.1, 
                                   label=f'±{i} Std. Dev.')
                else:
                    plt.fill_between(avg_data.index,
                                   avg_data - i*std_dev,
                                   avg_data - (i-1)*std_dev,
                                   color=colors[i-1], alpha=0.1)
                    plt.fill_between(avg_data.index,
                                   avg_data + (i-1)*std_dev,
                                   avg_data + i*std_dev,
                                   color=colors[i-1], alpha=0.1,
                                   label=f'±{i} Std. Dev.' if i == num_std else None)
                
                # Líneas de desviación
                plt.plot(avg_data.index, avg_data + i*std_dev, 
                        color=colors[i-1], linestyle='-', linewidth=0.8)
                plt.plot(avg_data.index, avg_data - i*std_dev, 
                        color=colors[i-1], linestyle='-', linewidth=0.8)
            
            # Ajustar límites del eje Y con desviaciones
            plt.ylim(bottom=0, top=y_max)

    return plt

def calculate_statistics_all_levels(selected_populations, selected_cycles, selected_groups, variable):
    all_stats = []
    
    # 1. Estadísticas por Población
    for population in selected_populations:
        data = df[(df['Population'] == population) & 
                (df['Cycle'].isin(selected_cycles)) &
                (df['Group'].isin(selected_groups))]
        data = data[data['Age_Days'] <= 40]
        if not data.empty:
            var_data = data[variable].dropna()
            if len(var_data) > 0:
                stats = {
                    'level': 'Population',
                    'population': population,
                    'cycle': 'All',
                    'group': 'All',
                    'count': len(var_data),
                    'mean': np.mean(var_data),
                    'std': np.std(var_data),
                    'min': np.min(var_data),
                    '25%': np.percentile(var_data, 25),
                    '50%': np.median(var_data),
                    '75%': np.percentile(var_data, 75),
                    'max': np.max(var_data)
                }
                all_stats.append(stats)
    
    # 2. Estadísticas por Ciclo
    for cycle in selected_cycles:
        data = df[(df['Cycle'] == cycle) & 
                (df['Population'].isin(selected_populations)) &
                (df['Group'].isin(selected_groups))]
        data = data[data['Age_Days'] <= 40]
        if not data.empty:
            var_data = data[variable].dropna()
            if len(var_data) > 0:
                stats = {
                    'level': 'Cycle',
                    'population': 'All',
                    'cycle': cycle,
                    'group': 'All',
                    'count': len(var_data),
                    'mean': np.mean(var_data),
                    'std': np.std(var_data),
                    'min': np.min(var_data),
                    '25%': np.percentile(var_data, 25),
                    '50%': np.median(var_data),
                    '75%': np.percentile(var_data, 75),
                    'max': np.max(var_data)
                }
                all_stats.append(stats)
    
    # 3. Estadísticas por Grupo
    for group in selected_groups:
        data = df[(df['Group'] == group) & 
                (df['Population'].isin(selected_populations)) &
                (df['Cycle'].isin(selected_cycles))]
        data = data[data['Age_Days'] <= 40]
        if not data.empty:
            var_data = data[variable].dropna()
            if len(var_data) > 0:
                stats = {
                    'level': 'Group',
                    'population': 'All',
                    'cycle': 'All',
                    'group': group,
                    'count': len(var_data),
                    'mean': np.mean(var_data),
                    'std': np.std(var_data),
                    'min': np.min(var_data),
                    '25%': np.percentile(var_data, 25),
                    '50%': np.median(var_data),
                    '75%': np.percentile(var_data, 75),
                    'max': np.max(var_data)
                }
                all_stats.append(stats)
    
    # 4. Estadísticas por Población x Ciclo
    for population in selected_populations:
        for cycle in selected_cycles:
            data = df[(df['Population'] == population) & 
                    (df['Cycle'] == cycle) &
                    (df['Group'].isin(selected_groups))]
            data = data[data['Age_Days'] <= 40]
            if not data.empty:
                var_data = data[variable].dropna()
                if len(var_data) > 0:
                    stats = {
                        'level': 'Population x Cycle',
                        'population': population,
                        'cycle': cycle,
                        'group': 'All',
                        'count': len(var_data),
                        'mean': np.mean(var_data),
                        'std': np.std(var_data),
                        'min': np.min(var_data),
                        '25%': np.percentile(var_data, 25),
                        '50%': np.median(var_data),
                        '75%': np.percentile(var_data, 75),
                        'max': np.max(var_data)
                    }
                    all_stats.append(stats)
    
    # 5. Estadísticas por Población x Ciclo x Grupo
    for population in selected_populations:
        for cycle in selected_cycles:
            for group in selected_groups:
                data = df[(df['Population'] == population) & 
                        (df['Cycle'] == cycle) &
                        (df['Group'] == group)]
                data = data[data['Age_Days'] <= 40]
                if not data.empty:
                    var_data = data[variable].dropna()
                    if len(var_data) > 0:
                        stats = {
                            'level': 'Population x Cycle x Group',
                            'population': population,
                            'cycle': cycle,
                            'group': group,
                            'count': len(var_data),
                            'mean': np.mean(var_data),
                            'std': np.std(var_data),
                            'min': np.min(var_data),
                            '25%': np.percentile(var_data, 25),
                            '50%': np.median(var_data),
                            '75%': np.percentile(var_data, 75),
                            'max': np.max(var_data)
                        }
                        all_stats.append(stats)
    
    return all_stats

def calculate_correlations(selected_populations, selected_cycles, selected_groups, variable):
    logger.info("Calculating correlations...")
    start_time = time.time()
    
    all_correlations = []
    
    # 1. Correlaciones entre poblaciones
    if len(selected_populations) > 1:
        pop_data = {}
        for pop in selected_populations:
            data = df[(df['Population'] == pop) & 
                    (df['Cycle'].isin(selected_cycles)) &
                    (df['Group'].isin(selected_groups))]
            data = data[data['Age_Days'] <= 40]
            if not data.empty:
                grouped = data.groupby('Age_Days')[variable].mean().reset_index()
                pop_data[pop] = grouped.set_index('Age_Days')[variable]
        
        for pop1, pop2 in combinations(selected_populations, 2):
            if pop1 in pop_data and pop2 in pop_data:
                combined = pd.concat([pop_data[pop1], pop_data[pop2]], axis=1, join='inner')
                combined.columns = [pop1, pop2]
                combined = combined.dropna()
                if len(combined) > 1:
                    # Pearson
                    corr_pearson, p_value_pearson = stats.pearsonr(combined[pop1], combined[pop2])
                    # Spearman
                    corr_spearman, p_value_spearman = stats.spearmanr(combined[pop1], combined[pop2])
                    
                    all_correlations.append({
                        'level': 'Between Populations',
                        'item1': pop1,
                        'item2': pop2,
                        'correlation_pearson': corr_pearson,
                        'p_value_pearson': p_value_pearson,
                        'correlation_spearman': corr_spearman,
                        'p_value_spearman': p_value_spearman,
                        'n': len(combined)
                    })
    
    # 2. Correlaciones entre ciclos dentro de poblaciones
    for population in selected_populations:
        if len(selected_cycles) > 1:
            cycle_data = {}
            for cycle in selected_cycles:
                data = df[(df['Population'] == population) & 
                        (df['Cycle'] == cycle) &
                        (df['Group'].isin(selected_groups))]
                data = data[data['Age_Days'] <= 40]
                if not data.empty:
                    grouped = data.groupby('Age_Days')[variable].mean().reset_index()
                    cycle_data[cycle] = grouped.set_index('Age_Days')[variable]
            
            for cyc1, cyc2 in combinations(selected_cycles, 2):
                if cyc1 in cycle_data and cyc2 in cycle_data:
                    combined = pd.concat([cycle_data[cyc1], cycle_data[cyc2]], axis=1, join='inner')
                    combined.columns = [cyc1, cyc2]
                    combined = combined.dropna()
                    if len(combined) > 1:
                        # Pearson
                        corr_pearson, p_value_pearson = stats.pearsonr(combined[cyc1], combined[cyc2])
                        # Spearman
                        corr_spearman, p_value_spearman = stats.spearmanr(combined[cyc1], combined[cyc2])
                        
                        all_correlations.append({
                            'level': 'Between Cycles within Population',
                            'population': population,
                            'item1': cyc1,
                            'item2': cyc2,
                            'correlation_pearson': corr_pearson,
                            'p_value_pearson': p_value_pearson,
                            'correlation_spearman': corr_spearman,
                            'p_value_spearman': p_value_spearman,
                            'n': len(combined)
                        })
    
    # 3. Correlaciones entre grupos
    if len(selected_groups) > 1:
        group_data = {}
        for group in selected_groups:
            data = df[(df['Group'] == group) & 
                    (df['Population'].isin(selected_populations)) &
                    (df['Cycle'].isin(selected_cycles))]
            data = data[data['Age_Days'] <= 40]
            if not data.empty:
                grouped = data.groupby('Age_Days')[variable].mean().reset_index()
                group_data[group] = grouped.set_index('Age_Days')[variable]
        
        for grp1, grp2 in combinations(selected_groups, 2):
            if grp1 in group_data and grp2 in group_data:
                combined = pd.concat([group_data[grp1], group_data[grp2]], axis=1, join='inner')
                combined.columns = [grp1, grp2]
                combined = combined.dropna()
                if len(combined) > 1:
                    # Pearson
                    corr_pearson, p_value_pearson = stats.pearsonr(combined[grp1], combined[grp2])
                    # Spearman
                    corr_spearman, p_value_spearman = stats.spearmanr(combined[grp1], combined[grp2])
                    
                    all_correlations.append({
                        'level': 'Between Groups',
                        'item1': grp1,
                        'item2': grp2,
                        'correlation_pearson': corr_pearson,
                        'p_value_pearson': p_value_pearson,
                        'correlation_spearman': corr_spearman,
                        'p_value_spearman': p_value_spearman,
                        'n': len(combined)
                    })
    
    logger.info(f"Correlations calculated in {time.time() - start_time:.2f} seconds")
    return all_correlations

def calculate_trends(selected_populations, selected_cycles, selected_groups, variable):
    trend_results = []
    
    for population in selected_populations:
        for cycle in selected_cycles:
            for group in selected_groups:
                data = df[(df['Population'] == population) & 
                        (df['Cycle'] == cycle) & 
                        (df['Group'] == group)]
                data = data[data['Age_Days'] <= 40]
                if not data.empty and len(data) > 1:
                    grouped = data.groupby('Age_Days')[variable].mean().reset_index()
                    if len(grouped) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            grouped['Age_Days'], grouped[variable]
                        )
                        trend_results.append({
                            'population': population,
                            'cycle': cycle,
                            'group': group,
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'trend_strength': 'Strong' if abs(r_value) > 0.7 else 
                                             'Moderate' if abs(r_value) > 0.5 else 
                                             'Weak'
                        })
    return trend_results

def calculate_comparative_analysis(selected_populations, selected_cycles, selected_groups, variable):
    comparative_results = []
    
    # 1. Comparación entre poblaciones
    pop_values = {}
    for pop in selected_populations:
        data = df[(df['Population'] == pop) & 
                (df['Cycle'].isin(selected_cycles)) &
                (df['Group'].isin(selected_groups))]
        data = data[data['Age_Days'] <= 40]
        if not data.empty:
            last_age = data['Age_Days'].max()
            last_value = data[data['Age_Days'] == last_age][variable].mean()
            pop_values[pop] = last_value
    
    if len(pop_values) > 1:
        populations = list(pop_values.keys())
        values = list(pop_values.values())
        max_val = max(values)
        min_val = min(values)
        max_pop = populations[values.index(max_val)]
        min_pop = populations[values.index(min_val)]
        diff_percent = ((max_val - min_val) / min_val) * 100 if min_val != 0 else 0
        
        comparative_results.append({
            'type': 'Population',
            'items': ', '.join(populations),
            'max_item': max_pop,
            'min_item': min_pop,
            'max_value': max_val,
            'min_value': min_val,
            'diff_percent': diff_percent
        })
    
    # 2. Comparación entre ciclos
    cycle_values = {}
    for cycle in selected_cycles:
        data = df[(df['Cycle'] == cycle) & 
                (df['Population'].isin(selected_populations)) &
                (df['Group'].isin(selected_groups))]
        data = data[data['Age_Days'] <= 40]
        if not data.empty:
            last_age = data['Age_Days'].max()
            last_value = data[data['Age_Days'] == last_age][variable].mean()
            cycle_values[cycle] = last_value
    
    if len(cycle_values) > 1:
        cycles = list(cycle_values.keys())
        values = list(cycle_values.values())
        max_val = max(values)
        min_val = min(values)
        max_cycle = cycles[values.index(max_val)]
        min_cycle = cycles[values.index(min_val)]
        diff_percent = ((max_val - min_val) / min_val) * 100 if min_val != 0 else 0
        
        comparative_results.append({
            'type': 'Cycle',
            'items': ', '.join(cycles),
            'max_item': max_cycle,
            'min_item': min_cycle,
            'max_value': max_val,
            'min_value': min_val,
            'diff_percent': diff_percent
        })
    
    # 3. Comparación entre grupos
    group_values = {}
    for group in selected_groups:
        data = df[(df['Group'] == group) & 
                (df['Population'].isin(selected_populations)) &
                (df['Cycle'].isin(selected_cycles))]
        data = data[data['Age_Days'] <= 40]
        if not data.empty:
            last_age = data['Age_Days'].max()
            last_value = data[data['Age_Days'] == last_age][variable].mean()
            group_values[group] = last_value
    
    if len(group_values) > 1:
        groups = list(group_values.keys())
        values = list(group_values.values())
        max_val = max(values)
        min_val = min(values)
        max_group = groups[values.index(max_val)]
        min_group = groups[values.index(min_val)]
        diff_percent = ((max_val - min_val) / min_val) * 100 if min_val != 0 else 0
        
        comparative_results.append({
            'type': 'Group',
            'items': ', '.join(groups),
            'max_item': max_group,
            'min_item': min_group,
            'max_value': max_val,
            'min_value': min_val,
            'diff_percent': diff_percent
        })
    
    # 4. Comparación entre ciclos dentro de poblaciones
    for population in selected_populations:
        pop_cycle_values = {}
        for cycle in selected_cycles:
            data = df[(df['Population'] == population) & 
                    (df['Cycle'] == cycle) &
                    (df['Group'].isin(selected_groups))]
            data = data[data['Age_Days'] <= 40]
            if not data.empty:
                last_point = data['Age_Days'].max()
                last_value = data[data['Age_Days'] == last_point][variable].mean()
                pop_cycle_values[cycle] = last_value
        
        if len(pop_cycle_values) > 1:
            cycles = list(pop_cycle_values.keys())
            values = list(pop_cycle_values.values())
            max_val = max(values)
            min_val = min(values)
            max_cycle = cycles[values.index(max_val)]
            min_cycle = cycles[values.index(min_val)]
            diff_percent = ((max_val - min_val) / min_val) * 100 if min_val != 0 else 0
            
            comparative_results.append({
                'type': 'Cycle within Population',
                'population': population,
                'items': ', '.join(cycles),
                'max_item': max_cycle,
                'min_item': min_cycle,
                'max_value': max_val,
                'min_value': min_val,
                'diff_percent': diff_percent
            })
    
    return comparative_results

def calculate_daily_growth(selected_populations, selected_cycles, selected_groups, variable):
    growth_results = []
    
    for population in selected_populations:
        for cycle in selected_cycles:
            for group in selected_groups:
                data = df[(df['Population'] == population) & 
                        (df['Cycle'] == cycle) & 
                        (df['Group'] == group)]
                data = data[data['Age_Days'] <= 40]
                if not data.empty and len(data) > 1:
                    grouped = data.groupby('Age_Days')[variable].mean().reset_index()
                    daily_growth = np.diff(grouped[variable]) / np.diff(grouped['Age_Days'])
                    avg_daily_growth = np.mean(daily_growth) if len(daily_growth) > 0 else 0
                    
                    growth_results.append({
                        'population': population,
                        'cycle': cycle,
                        'group': group,
                        'avg_daily_growth': avg_daily_growth,
                        'max_daily_growth': np.max(daily_growth) if len(daily_growth) > 0 else 0,
                        'min_daily_growth': np.min(daily_growth) if len(daily_growth) > 0 else 0
                    })
    
    return growth_results

def perform_tukey_hsd(selected_populations, selected_cycles, selected_groups, variable):
    tukey_results = []
    
    # Preparar datos para Tukey (último punto de datos)
    tukey_data = []
    
    for population in selected_populations:
        for cycle in selected_cycles:
            for group in selected_groups:
                data = df[(df['Population'] == population) & 
                        (df['Cycle'] == cycle) & 
                        (df['Group'] == group)]
                data = data[data['Age_Days'] <= 40]
                if not data.empty:
                    last_age = data['Age_Days'].max()
                    last_data = data[data['Age_Days'] == last_age]
                    for _, row in last_data.iterrows():
                        tukey_data.append({
                            'population': population,
                            'cycle': cycle,
                            'group': group,
                            'variable': row[variable],
                            'combined': f"{population}-{cycle}-{group}"
                        })
    
    if not tukey_data:
        return tukey_results
    
    tukey_df = pd.DataFrame(tukey_data)
    
    # 1. ANOVA y Tukey para poblaciones
    if len(selected_populations) > 1:
        anova_result = stats.f_oneway(
            *[tukey_df[tukey_df['population'] == pop]['variable'] for pop in selected_populations]
        )
        
        if anova_result.pvalue < 0.05:
            tukey = pairwise_tukeyhsd(
                tukey_df['variable'],
                tukey_df['population'],
                alpha=0.05
            )
            tukey_summary = tukey.summary()
            
            tukey_results.append({
                'factor': 'Population',
                'p_value': anova_result.pvalue,
                'tukey_summary': str(tukey_summary)
            })
    
    # 2. ANOVA y Tukey para ciclos
    if len(selected_cycles) > 1:
        anova_result = stats.f_oneway(
            *[tukey_df[tukey_df['cycle'] == cyc]['variable'] for cyc in selected_cycles]
        )
        
        if anova_result.pvalue < 0.05:
            tukey = pairwise_tukeyhsd(
                tukey_df['variable'],
                tukey_df['cycle'],
                alpha=0.05
            )
            tukey_summary = tukey.summary()
            
            tukey_results.append({
                'factor': 'Cycle',
                'p_value': anova_result.pvalue,
                'tukey_summary': str(tukey_summary)
            })
    
    # 3. ANOVA y Tukey para grupos
    if len(selected_groups) > 1:
        anova_result = stats.f_oneway(
            *[tukey_df[tukey_df['group'] == grp]['variable'] for grp in selected_groups]
        )
        
        if anova_result.pvalue < 0.05:
            tukey = pairwise_tukeyhsd(
                tukey_df['variable'],
                tukey_df['group'],
                alpha=0.05
            )
            tukey_summary = tukey.summary()
            
            tukey_results.append({
                'factor': 'Group',
                'p_value': anova_result.pvalue,
                'tukey_summary': str(tukey_summary)
            })
    
    # 4. ANOVA y Tukey para combinaciones
    if len(selected_populations) > 0 and len(selected_cycles) > 0 and len(selected_groups) > 0:
        anova_result = stats.f_oneway(
            *[tukey_df[tukey_df['combined'] == combo]['variable'] 
              for combo in tukey_df['combined'].unique()]
        )
        
        if anova_result.pvalue < 0.05:
            tukey = pairwise_tukeyhsd(
                tukey_df['variable'],
                tukey_df['combined'],
                alpha=0.05
            )
            tukey_summary = tukey.summary()
            
            tukey_results.append({
                'factor': 'Combination',
                'p_value': anova_result.pvalue,
                'tukey_summary': str(tukey_summary)
            })
    
    return tukey_results

def calculate_advanced_statistics(selected_populations, selected_cycles, selected_groups, variable):
    logger.info("Calculating advanced statistics...")
    start_time = time.time()
    
    results = {
        'correlations': [],
        'hypothesis_tests': [],
        'cluster_analysis': {}
    }
    
    # 1. Cálculo de correlaciones entre métricas
    # Precalcular métricas por grupo
    group_metrics = {}
    for pop in selected_populations:
        for cyc in selected_cycles:
            for grp in selected_groups:
                key = f"{pop}-{cyc}-{grp}"
                group_data = df[(df['Population'] == pop) & 
                              (df['Cycle'] == cyc) & 
                              (df['Group'] == grp) &
                              (df['Age_Days'] <= 40)]
                
                if not group_data.empty:
                    # Último punto
                    last_age = group_data['Age_Days'].max()
                    last_point = group_data[group_data['Age_Days'] == last_age][variable].mean()
                    
                    # Pendiente (slope)
                    ages = group_data['Age_Days']
                    values = group_data[variable]
                    if len(ages) > 1:
                        slope, _, _, _, _ = stats.linregress(ages, values)
                    else:
                        slope = np.nan
                    
                    # Crecimiento diario
                    daily_growth = calculate_daily_growth([pop], [cyc], [grp], variable)
                    avg_daily = daily_growth[0]['avg_daily_growth'] if daily_growth else np.nan
                    
                    group_metrics[key] = {
                        'last_point': last_point,
                        'slope': slope,
                        'daily_growth': avg_daily
                    }
    
    # Convertir a DataFrame para correlaciones
    metrics_df = pd.DataFrame.from_dict(group_metrics, orient='index')
    metrics_df = metrics_df.dropna()
    
    if not metrics_df.empty:
        # Correlaciones de Pearson
        pearson_corr, pearson_p = stats.pearsonr(metrics_df['slope'], metrics_df['daily_growth'])
        results['correlations'].append({
            'type': 'Pearson',
            'metric1': 'slope',
            'metric2': 'daily_growth',
            'correlation': pearson_corr,
            'p_value': pearson_p
        })
        
        # Correlaciones de Spearman
        spearman_corr, spearman_p = stats.spearmanr(metrics_df['slope'], metrics_df['daily_growth'])
        results['correlations'].append({
            'type': 'Spearman',
            'metric1': 'slope',
            'metric2': 'daily_growth',
            'correlation': spearman_corr,
            'p_value': spearman_p
        })
    
    # 2. Pruebas de hipótesis
    # Preparar datos para pruebas
    test_data = []
    for pop in selected_populations:
        for cyc in selected_cycles:
            for grp in selected_groups:
                data_point = group_metrics.get(f"{pop}-{cyc}-{grp}", {})
                if data_point:
                    test_data.append({
                        'population': pop,
                        'cycle': cyc,
                        'group': grp,
                        'last_point': data_point['last_point'],
                        'slope': data_point['slope'],
                        'daily_growth': data_point['daily_growth']
                    })
    
    test_df = pd.DataFrame(test_data)
    
    if not test_df.empty:
        # ANOVA para población
        if len(selected_populations) > 1:
            try:
                model = ols('last_point ~ C(population)', data=test_df).fit()
                anova_table = anova_lm(model)
                results['hypothesis_tests'].append({
                    'factor': 'Population',
                    'metric': 'last_point',
                    'test': 'ANOVA',
                    'f_value': anova_table['F'][0],
                    'p_value': anova_table['PR(>F)'][0]
                })
            except Exception as e:
                logger.error(f"ANOVA error for population: {str(e)}")
        
        # ANOVA para ciclo
        if len(selected_cycles) > 1:
            try:
                model = ols('slope ~ C(cycle)', data=test_df).fit()
                anova_table = anova_lm(model)
                results['hypothesis_tests'].append({
                    'factor': 'Cycle',
                    'metric': 'slope',
                    'test': 'ANOVA',
                    'f_value': anova_table['F'][0],
                    'p_value': anova_table['PR(>F)'][0]
                })
            except Exception as e:
                logger.error(f"ANOVA error for cycle: {str(e)}")
    
    # 3. Análisis de clustering
    try:
        # Preparar datos para clustering
        cluster_data = []
        for pop in selected_populations:
            for cyc in selected_cycles:
                key = f"{pop}-{cyc}"
                pop_cyc_data = test_df[(test_df['population'] == pop) & (test_df['cycle'] == cyc)]
                if not pop_cyc_data.empty:
                    cluster_data.append({
                        'group': key,
                        'last_point': pop_cyc_data['last_point'].mean(),
                        'slope': pop_cyc_data['slope'].mean(),
                        'daily_growth': pop_cyc_data['daily_growth'].mean()
                    })
        
        cluster_df = pd.DataFrame(cluster_data).set_index('group')
        
        if len(cluster_df) > 1:
            # Normalizar datos
            cluster_df_norm = (cluster_df - cluster_df.mean()) / cluster_df.std()
            
            # Calcular matriz de distancia
            dist_matrix = pdist(cluster_df_norm, metric='euclidean')
            dist_square = squareform(dist_matrix)
            
            # Realizar clustering jerárquico
            Z = linkage(dist_matrix, method='ward')
            
            # Generar dendrograma
            plt.figure(figsize=(12, 6))
            dendrogram(Z, labels=cluster_df.index.tolist(), leaf_rotation=90)
            plt.title('Cluster Analysis of Population-Cycle Groups')
            plt.ylabel('Distance')
            plt.tight_layout()
            
            # Guardar imagen
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            plt.close()
            img.seek(0)
            cluster_plot = base64.b64encode(img.getvalue()).decode()
            
            results['cluster_analysis']['plot'] = cluster_plot
            results['cluster_analysis']['distance_matrix'] = dist_square.tolist()
            results['cluster_analysis']['labels'] = cluster_df.index.tolist()
    except Exception as e:
        logger.error(f"Cluster analysis error: {str(e)}")
    
    logger.info(f"Advanced statistics calculated in {time.time() - start_time:.2f} seconds")
    return results

def generate_correlation_matrix_plot(correlation_data):
    if not correlation_data:
        return None
    
    # Crear matriz de correlación
    corr_df = pd.DataFrame(correlation_data)
    
    # Preparar datos para el heatmap
    heatmap_data = corr_df.pivot_table(index='item1', columns='item2', values='correlation_pearson')
    
    # Ordenar las filas y columnas alfabéticamente
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.index), axis=0)
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, 
                linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Group Correlation Matrix (Pearson)')
    plt.tight_layout()
    
    # Guardar imagen
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_trend_plot(trend_data, variable):
    if not trend_data:
        return None
    
    plt.figure(figsize=(12, 8))
    
    # Crear gráfico de barras para las pendientes
    labels = []
    slopes = []
    for result in trend_data:
        labels.append(f"{result['population']}-{result['cycle']}-{result['group']}")
        slopes.append(result['slope'])
    
    # Ordenar por pendiente
    sorted_idx = np.argsort(slopes)
    labels = [labels[i] for i in sorted_idx]
    slopes = [slopes[i] for i in sorted_idx]
    
    colors = ['green' if s > 0 else 'red' for s in slopes]
    plt.barh(labels, slopes, color=colors)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('Growth Rate (Slope)')
    plt.title(f'Trend Analysis: {variable}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Agregar valores
    for i, v in enumerate(slopes):
        plt.text(v, i, f" {v:.4f}", color='black', va='center')
    
    # Guardar imagen
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_comparative_plot(comparative_data, variable):
    if not comparative_data:
        return None
    
    plt.figure(figsize=(14, 8))
    
    # Preparar datos
    labels = []
    max_values = []
    min_values = []
    max_items = []
    min_items = []
    comparison_types = []
    
    for result in comparative_data:
        #label = f"{result['type']}: {result['items']}"
        label = f"{result['type']}"
        labels.append(label)
        max_values.append(result['max_value'])
        min_values.append(result['min_value'])
        max_items.append(result['max_item'])
        min_items.append(result['min_item'])
        comparison_types.append(result['type'])
    
    # Crear gráfico de barras horizontales
    y = np.arange(len(labels))
    height = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.barh(y - height/2, max_values, height, label='Max Value', color='#2ca02c')
    rects2 = ax.barh(y + height/2, min_values, height, label='Min Value', color='#d62728')
    
    ax.set_xlabel(variable)
    ax.set_title('Comparative Analysis: Max vs Min Values')
    
    # FUNCIÓN PARA ROMPER ETIQUETAS LARGAS
    def wrap_labels(labels, max_len=15):
        wrapped = []
        for label in labels:
            words = label.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_len:
                    current_line += (word + " ")
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word + " "
            
            if current_line:
                lines.append(current_line.strip())
                
            wrapped.append('\n'.join(lines))
        return wrapped
    
    # Aplicar wrap a las etiquetas
    wrapped_labels = wrap_labels(labels)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_labels)
    ax.legend()
    
    # Calcular el valor máximo global para ajustar límites
    global_max = max(max_values) if max_values else 0
    
    # AJUSTAR LÍMITES CON MÁRGEN ADICIONAL (25%)
    ax.set_xlim(0, global_max * 1.25)
    
    # Agregar valores y nombres a las barras
    for i, rect in enumerate(rects1):
        width = rect.get_width()
        
        # Valor numérico FUERA de la barra
        ax.annotate(f'{width:.2f}',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center')
        
        # Nombre del ítem DENTRO de la barra
        # Usar posición relativa (3% del valor máximo) para mejor ubicación
        text_x = rect.get_x() + global_max * 0.03
        ax.annotate(max_items[i],
                    xy=(text_x, rect.get_y() + rect.get_height() / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    color='white', fontweight='bold', fontsize=9)

    for i, rect in enumerate(rects2):
        width = rect.get_width()
        
        # Valor numérico FUERA de la barra
        ax.annotate(f'{width:.2f}',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center')
        
        # Nombre del ítem DENTRO de la barra
        text_x = rect.get_x() + global_max * 0.03
        ax.annotate(min_items[i],
                    xy=(text_x, rect.get_y() + rect.get_height() / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    color='white', fontweight='bold', fontsize=9)
    
    # Ajustar diseño automáticamente
    fig.tight_layout()
    
    # Añadir margen extra en la parte derecha
    plt.subplots_adjust(right=0.85)  # Ajustar según necesidad
    
    # Guardar imagen
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_plot(selected_populations, selected_cycles, selected_groups, variable, num_std, show_annotations):
    start_time = time.time()
    logger.info("Generating plot and statistics...")
    
    plt, y_max = create_base_plot(selected_populations, selected_cycles, selected_groups, variable, show_annotations)
    
    try:
        num_std = int(num_std)
    except (ValueError, TypeError):
        num_std = DEFAULT_STD_DEV
    
    plt = add_std_deviation(plt, variable, num_std, y_max)
    
    # Guardar en múltiples formatos
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    for fmt, ext in [('svg', 'svg'), ('pdf', 'pdf')]:
        output_path = os.path.join(dir_path, f'plot.{ext}')
        plt.savefig(output_path, format=fmt)
    
    plt.close()
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Calcular estadísticas por niveles
    stats = calculate_statistics_all_levels(selected_populations, selected_cycles, selected_groups, variable)
    
    # Calcular correlaciones por niveles
    correlations = calculate_correlations(selected_populations, selected_cycles, selected_groups, variable)
    
    # Calcular tendencias
    trends = calculate_trends(selected_populations, selected_cycles, selected_groups, variable)
    
    # Calcular análisis comparativo
    comparative = calculate_comparative_analysis(selected_populations, selected_cycles, selected_groups, variable)
    
    # Calcular crecimiento diario
    daily_growth = calculate_daily_growth(selected_populations, selected_cycles, selected_groups, variable)
    
    # Calcular pruebas de Tukey
    tukey_results = perform_tukey_hsd(selected_populations, selected_cycles, selected_groups, variable)
    
    # Calcular estadísticas avanzadas
    advanced_stats = calculate_advanced_statistics(selected_populations, selected_cycles, selected_groups, variable)
    
    # Generar gráficos adicionales
    correlation_plot = generate_correlation_matrix_plot(correlations) if correlations else None
    trend_plot = generate_trend_plot(trends, variable) if trends else None
    comparative_plot = generate_comparative_plot(comparative, variable) if comparative else None
    
    # Obtener pruebas de hipótesis
    hypothesis_tests = advanced_stats.get('hypothesis_tests', []) if advanced_stats else []
    
    logger.info(f"Plot and statistics generated in {time.time() - start_time:.2f} seconds")
    
    return (plot_url, stats, correlations, trends, comparative, 
            daily_growth, tukey_results, correlation_plot, trend_plot, 
            comparative_plot, advanced_stats, hypothesis_tests)

def generate_pdf_report(selected_populations, selected_cycles, selected_groups, variable, num_std, show_annotations):
    logger.info("Generating PDF report...")
    start_time = time.time()
    
    # Generar todos los componentes del reporte
    (plot_url, stats, correlations, trends, comparative, 
     daily_growth, tukey_results, correlation_plot, trend_plot, 
     comparative_plot, advanced_stats, hypothesis_tests) = generate_plot(
        selected_populations, 
        selected_cycles, 
        selected_groups, 
        variable, 
        num_std,
        show_annotations
    )
    
    # Crear el PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        title="Larvae Growth Analysis Report",
        author="LarvaeTracker",
        leftMargin=0.25*inch,
        rightMargin=0.25*inch,
        topMargin=0.25*inch,
        bottomMargin=0.25*inch
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Título del reporte
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=18,
        alignment=1,
        spaceAfter=12
    )
    elements.append(Paragraph("Larvae Growth Analysis Report", title_style))
    
    # Información de la selección
    selection_info = f"""
    <b>Variable:</b> {variable}<br/>
    <b>Populations:</b> {', '.join(selected_populations)}<br/>
    <b>Cycles:</b> {', '.join(selected_cycles)}<br/>
    <b>Groups:</b> {', '.join(selected_groups)}<br/>
    <b>Standard Deviations:</b> {num_std}<br/>
    <b>Show Annotations:</b> {'Yes' if show_annotations else 'No'}
    """
    elements.append(Paragraph(selection_info, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Gráfico principal
    elements.append(Paragraph("Growth Visualization", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Convertir imagen base64 a objeto de imagen para PDF
    plot_img_data = base64.b64decode(plot_url)
    plot_img = Image(BytesIO(plot_img_data), width=7*inch, height=5*inch)
    plot_img.hAlign = 'CENTER'
    elements.append(plot_img)
    elements.append(Spacer(1, 0.2*inch))
    
    # Estadísticas descriptivas
    if stats:
        elements.append(Paragraph("Descriptive Statistics", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Preparar datos para la tabla
        table_data = [['Level', 'Population', 'Cycle', 'Group', 'Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']]
        for stat in stats:
            row = [
                stat['level'],
                stat['population'],
                stat['cycle'],
                stat['group'],
                str(stat['count']),
                f"{stat['mean']:.4f}",
                f"{stat['std']:.4f}",
                f"{stat['min']:.4f}",
                f"{stat['25%']:.4f}",
                f"{stat['50%']:.4f}",
                f"{stat['75%']:.4f}",
                f"{stat['max']:.4f}"
            ]
            table_data.append(row)
        
        # Crear tabla
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#FF7B00")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#DEE2E6")),
            ('FONTSIZE', (0,1), (-1,-1), 7),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(KeepTogether([table]))
        elements.append(Spacer(1, 0.2*inch))
    
    # Análisis comparativo
    if comparative:
        elements.append(Paragraph("Comparative Analysis", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Preparar datos para la tabla
        table_data = [['Type', 'Items', 'Max Item', 'Max Value', 'Min Item', 'Min Value', 'Diff %']]
        for comp in comparative:
            row = [
                comp['type'],
                comp['items'],
                comp['max_item'],
                f"{comp['max_value']:.4f}",
                comp['min_item'],
                f"{comp['min_value']:.4f}",
                f"{comp['diff_percent']:.2f}%"
            ]
            table_data.append(row)
        
        # Crear tabla
        table = Table(table_data, repeatRows=1, colWidths=[1.2*inch, 1.5*inch, 1*inch, 0.8*inch, 1*inch, 0.8*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#28A745")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#DEE2E6")),
            ('FONTSIZE', (0,1), (-1,-1), 7),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(KeepTogether([table]))
        elements.append(Spacer(1, 0.2*inch))
    
    # Análisis de tendencias
    if trends:
        elements.append(Paragraph("Growth Trends", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Preparar datos para la tabla
        table_data = [['Population', 'Cycle', 'Group', 'Slope', 'Intercept', 'R²', 'P-value', 'Trend Strength']]
        for trend in trends:
            row = [
                trend['population'],
                trend['cycle'],
                trend['group'],
                f"{trend['slope']:.6f}",
                f"{trend['intercept']:.4f}",
                f"{trend['r_squared']:.4f}",
                f"{trend['p_value']:.6f}",
                trend['trend_strength']
            ]
            table_data.append(row)
        
        # Crear tabla
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E88E5")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#DEE2E6")),
            ('FONTSIZE', (0,1), (-1,-1), 7),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(KeepTogether([table]))
        elements.append(Spacer(1, 0.2*inch))
    
    # Gráficos adicionales
    if comparative_plot:
        elements.append(Paragraph("Comparative Visualization", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Convertir imagen base64 a objeto de imagen para PDF
        comp_plot_img_data = base64.b64decode(comparative_plot)
        comp_plot_img = Image(BytesIO(comp_plot_img_data), width=7*inch, height=4*inch)
        comp_plot_img.hAlign = 'CENTER'
        elements.append(comp_plot_img)
        elements.append(Spacer(1, 0.2*inch))
    
    if trend_plot:
        elements.append(Paragraph("Trend Visualization", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Convertir imagen base64 a objeto de imagen para PDF
        trend_plot_img_data = base64.b64decode(trend_plot)
        trend_plot_img = Image(BytesIO(trend_plot_img_data), width=7*inch, height=4*inch)
        trend_plot_img.hAlign = 'CENTER'
        elements.append(trend_plot_img)
        elements.append(Spacer(1, 0.2*inch))
    
    # Construir el documento PDF
    doc.build(elements)
    buffer.seek(0)
    
    logger.info(f"PDF report generated in {time.time() - start_time:.2f} seconds")
    return buffer

@app.route('/generate_plot', methods=['POST'])
def api_generate_plot():
    data = request.get_json()
    
    selected_populations = data.get('populations', [])
    selected_cycles = data.get('cycles', [])
    selected_groups = data.get('groups', [])
    variable = data.get('variable', 'Indv_Weight')
    num_std = data.get('num_std', DEFAULT_STD_DEV)
    show_annotations = data.get('show_annotations', DEFAULT_SHOW_ANNOTATIONS)
    show_correlations = data.get('show_correlations', DEFAULT_SHOW_CORRELATIONS)
    
    try:
        (plot_url, stats, correlations, trends, comparative, 
         daily_growth, tukey_results, correlation_plot, 
         trend_plot, comparative_plot, advanced_stats, hypothesis_tests) = generate_plot(
            selected_populations, 
            selected_cycles, 
            selected_groups, 
            variable, 
            num_std,
            show_annotations
        )
        
        response = {
            'status': 'success', 
            'plot_url': plot_url,
            'stats': stats,
            'correlations': correlations,
            'trends': trends,
            'comparative': comparative,
            'daily_growth': daily_growth,
            'tukey_results': tukey_results,
            'correlation_plot': correlation_plot,
            'trend_plot': trend_plot,
            'comparative_plot': comparative_plot,
            'advanced_stats': advanced_stats,
            'hypothesis_tests': hypothesis_tests
        }
            
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/generate_pdf_report', methods=['POST'])
def api_generate_pdf_report():
    data = request.get_json()
    
    selected_populations = data.get('populations', [])
    selected_cycles = data.get('cycles', [])
    selected_groups = data.get('groups', [])
    variable = data.get('variable', 'Indv_Weight')
    num_std = data.get('num_std', DEFAULT_STD_DEV)
    show_annotations = data.get('show_annotations', DEFAULT_SHOW_ANNOTATIONS)
    
    try:
        pdf_buffer = generate_pdf_report(
            selected_populations, 
            selected_cycles, 
            selected_groups, 
            variable, 
            num_std,
            show_annotations
        )
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='growth_analysis_report.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    global df
    if df is None:
        return "Error: File not loaded. Please go back and reload the file."
    
    return render_template(template_path,
                         populations=df['Population'].unique(),
                         cycles=df['Cycle'].unique(),
                         groups=df['Group'].unique(),
                         variables=variable_columns,
                         std_options=range(0, MAX_STD_DEV + 1),
                         selected_std=DEFAULT_STD_DEV,
                         show_annotations=DEFAULT_SHOW_ANNOTATIONS,
                         show_correlations=DEFAULT_SHOW_CORRELATIONS)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file> OR tracker.exe <csv_file>")
        csv_file = input("Enter file path: ").strip()
        if not csv_file:
            print("No file supplied...")
            sys.exit(1)
    else:
        csv_file = sys.argv[1]

    if not os.path.isfile(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)

    df = load_data(csv_file)
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True)
