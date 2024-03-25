#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:44:48 2024

@author: Sebastardito!
"""

import sys
import importlib

required_libraries = ['flask', 'pandas', 'matplotlib']

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
else:        
    pass

import os
import pandas as pd
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import webbrowser

dir_path = os.path.dirname(os.path.realpath(__file__))
template_path = 'index.html'

app = Flask(__name__)

df = None

def load_data(csv_file):
    return pd.read_csv(csv_file, delimiter=';')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file> OR tracker.exe <csv_file> ")
        print("You can add your path/to/file.csv here -->")
        csv_file = input()
        if not csv_file.strip():
            print("No file has been supplied...")
            sys.exit(1)
    else:
        csv_file = sys.argv[1]

    if not os.path.isfile(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)

    df = load_data(csv_file)

def generate_plot(selected_populations, selected_cycles, selected_groups, variable):
    plt.figure(figsize=(10, 6))
    for population in selected_populations:
        for cycle in selected_cycles:
            for group in selected_groups:
                data = df[(df['Population'] == population) & 
                          (df['Cycle'] == cycle) & 
                          (df['Group'] == group)]
                data = data[data['Age_Days'] <= 40]  # Limitar a 40 días
                data = data.groupby('Age_Days')[variable].mean().reset_index()  # Promediar valores por día
                last_point = data.iloc[-1]
                last_data_date = df[(df['Population'] == population) & 
                                    (df['Cycle'] == cycle) & 
                                    (df['Group'] == group) &
                                    (df['Age_Days'] == last_point['Age_Days'])]['Data_Date'].iloc[0]
                label = f'{population}, {cycle}, {group}, Last Data: {last_data_date}'
                plt.plot(data['Age_Days'], data[variable], label=label, marker='o')  # Mostrar puntos de datos

                # Añadir etiqueta al último punto
                #plt.annotate(f"{last_data_date}\n ({int(last_point['Age_Days'])} Days)", (int(last_point['Age_Days']), last_point[variable]), xytext=(5, 5), textcoords='offset points', ha='left', va='bottom')


    # Calcular desviación estándar para avg_data
    avg_data = df[df['Age_Days'] <= 40].groupby('Age_Days')[variable].mean()
    std_dev = df[df['Age_Days'] <= 40].groupby('Age_Days')[variable].std()
    plt.plot(avg_data.index, avg_data.values, color='black', linestyle='--', linewidth=0.5, label='Overall Mean')

    # Plotear áreas basadas en la desviación estándar
    plt.fill_between(avg_data.index, avg_data - std_dev, avg_data + std_dev, color='blue', alpha=0.1, label='Std. Deviation 1')
    plt.fill_between(avg_data.index, avg_data - 2 * std_dev, avg_data - std_dev, color='green', alpha=0.1, label='Std. Deviation 2')
    plt.fill_between(avg_data.index, avg_data + std_dev, avg_data + 2 * std_dev, color='green', alpha=0.1)
    # plt.fill_between(avg_data.index, avg_data - 3 * std_dev, avg_data - 2 * std_dev, color='orange', alpha=0.1, label='3 std deviations')
    # plt.fill_between(avg_data.index, avg_data + 2 * std_dev, avg_data + 3 * std_dev, color='orange', alpha=0.1)
    # plt.fill_between(avg_data.index, avg_data - 4 * std_dev, avg_data - 3 * std_dev, color='red', alpha=0.1, label='4 std deviations')
    # plt.fill_between(avg_data.index, avg_data + 3 * std_dev, avg_data + 4 * std_dev, color='red', alpha=0.1)

    # Plotear líneas delgadas para cada desviación estándar, range es el numero de desviaciones estandar (de 1 a 5 significa 1,2,3,4sd)
    for i in range(1, 3):
        plt.plot(avg_data.index, avg_data + i * std_dev, color=['blue', 'green', 'orange', 'red'][i-1], linestyle='-', linewidth=0.5)
        plt.plot(avg_data.index, avg_data - i * std_dev, color=['blue', 'green', 'orange', 'red'][i-1], linestyle='-', linewidth=0.5)

    plt.xlabel('Age (Days)')
    plt.ylabel(variable)
    plt.title(f'{variable} Curves for Selected Populations')
    plt.legend()
    plt.ylim(bottom=0)  
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Guardar el gráfico como imagen SVG
    svg_path = os.path.join(dir_path,'plot.svg')
    plt.figure(figsize=(10, 6))
    for population in selected_populations:
        for cycle in selected_cycles:
            for group in selected_groups:
                data = df[(df['Population'] == population) & 
                          (df['Cycle'] == cycle) & 
                          (df['Group'] == group)]
                data = data[data['Age_Days'] <= 40]  # Limitar a 40 días
                data = data.groupby('Age_Days')[variable].mean().reset_index()  # Promediar valores por día
                last_point = data.iloc[-1]
                last_data_date = df[(df['Population'] == population) & 
                                    (df['Cycle'] == cycle) & 
                                    (df['Group'] == group) &
                                    (df['Age_Days'] == last_point['Age_Days'])]['Data_Date'].iloc[0]
                label = f'{population}, {cycle}, {group}, Last Data: {last_data_date}'
                plt.plot(data['Age_Days'], data[variable], label=label, marker='o')  # Mostrar puntos de datos

                # Añadir etiqueta al último punto
                #plt.annotate(f"{last_data_date}\n ({int(last_point['Age_Days'])} Days)", (int(last_point['Age_Days']), last_point[variable]), xytext=(5, 5), textcoords='offset points', ha='left', va='bottom')


    # Calcular desviación estándar para avg_data
    avg_data = df[df['Age_Days'] <= 40].groupby('Age_Days')[variable].mean()
    std_dev = df[df['Age_Days'] <= 40].groupby('Age_Days')[variable].std()
    plt.plot(avg_data.index, avg_data.values, color='black', linestyle='--', linewidth=0.5, label='Overall Mean')

    # Plotear áreas basadas en la desviación estándar
    plt.fill_between(avg_data.index, avg_data - std_dev, avg_data + std_dev, color='blue', alpha=0.1, label='Std. Deviation 1')
    plt.fill_between(avg_data.index, avg_data - 2 * std_dev, avg_data - std_dev, color='green', alpha=0.1, label='Std. Deviation 2')
    plt.fill_between(avg_data.index, avg_data + std_dev, avg_data + 2 * std_dev, color='green', alpha=0.1)
    # plt.fill_between(avg_data.index, avg_data - 3 * std_dev, avg_data - 2 * std_dev, color='orange', alpha=0.1, label='3 std deviations')
    # plt.fill_between(avg_data.index, avg_data + 2 * std_dev, avg_data + 3 * std_dev, color='orange', alpha=0.1)
    # plt.fill_between(avg_data.index, avg_data - 4 * std_dev, avg_data - 3 * std_dev, color='red', alpha=0.1, label='4 std deviations')
    # plt.fill_between(avg_data.index, avg_data + 3 * std_dev, avg_data + 4 * std_dev, color='red', alpha=0.1)

    # Plotear líneas delgadas para cada desviación estándar, range es el numero de desviaciones estandar (de 1 a 5 significa 1,2,3,4sd)
    for i in range(1, 3):
        plt.plot(avg_data.index, avg_data + i * std_dev, color=['blue', 'green', 'orange', 'red'][i-1], linestyle='-', linewidth=0.5)
        plt.plot(avg_data.index, avg_data - i * std_dev, color=['blue', 'green', 'orange', 'red'][i-1], linestyle='-', linewidth=0.5)

    plt.xlabel('Age (Days)')
    plt.ylabel(variable)
    plt.title(f'{variable} Curves for Selected Populations')
    plt.legend()
    plt.ylim(bottom=0)  # Establecer límite inferior del eje Y en 0
    plt.tight_layout()
    plt.savefig(svg_path, format='svg')
    plt.close()

    # Guardar el gráfico como PDF
    pdf_path = os.path.join(dir_path,'plot.pdf')
    plt.figure(figsize=(10, 6))
    for population in selected_populations:
        for cycle in selected_cycles:
            for group in selected_groups:
                data = df[(df['Population'] == population) & 
                          (df['Cycle'] == cycle) & 
                          (df['Group'] == group)]
                data = data[data['Age_Days'] <= 40]  # Limitar a 40 días
                data = data.groupby('Age_Days')[variable].mean().reset_index()  # Promediar valores por día
                last_point = data.iloc[-1]
                last_data_date = df[(df['Population'] == population) & 
                                    (df['Cycle'] == cycle) & 
                                    (df['Group'] == group) &
                                    (df['Age_Days'] == last_point['Age_Days'])]['Data_Date'].iloc[0]
                label = f'{population}, {cycle}, {group}, Last Data: {last_data_date}'
                plt.plot(data['Age_Days'], data[variable], label=label, marker='o')  # Mostrar puntos de datos

                # Añadir etiqueta al último punto
                #plt.annotate(f"{last_data_date}\n ({int(last_point['Age_Days'])} Days)", (int(last_point['Age_Days']), last_point[variable]), xytext=(5, 5), textcoords='offset points', ha='left', va='bottom')


    # Calcular desviación estándar para avg_data
    avg_data = df[df['Age_Days'] <= 40].groupby('Age_Days')[variable].mean()
    std_dev = df[df['Age_Days'] <= 40].groupby('Age_Days')[variable].std()
    plt.plot(avg_data.index, avg_data.values, color='black', linestyle='--', linewidth=0.5, label='Overall Mean')

    # Plotear áreas basadas en la desviación estándar
    plt.fill_between(avg_data.index, avg_data - std_dev, avg_data + std_dev, color='blue', alpha=0.1, label='Std. Deviation 1')
    plt.fill_between(avg_data.index, avg_data - 2 * std_dev, avg_data - std_dev, color='green', alpha=0.1, label='Std. Deviation 2')
    plt.fill_between(avg_data.index, avg_data + std_dev, avg_data + 2 * std_dev, color='green', alpha=0.1)
    # plt.fill_between(avg_data.index, avg_data - 3 * std_dev, avg_data - 2 * std_dev, color='orange', alpha=0.1, label='3 std deviations')
    # plt.fill_between(avg_data.index, avg_data + 2 * std_dev, avg_data + 3 * std_dev, color='orange', alpha=0.1)
    # plt.fill_between(avg_data.index, avg_data - 4 * std_dev, avg_data - 3 * std_dev, color='red', alpha=0.1, label='4 std deviations')
    # plt.fill_between(avg_data.index, avg_data + 3 * std_dev, avg_data + 4 * std_dev, color='red', alpha=0.1)

    # Plotear líneas delgadas para cada desviación estándar, range es el numero de desviaciones estandar (de 1 a 5 significa 1,2,3,4sd)
    for i in range(1, 3):
        plt.plot(avg_data.index, avg_data + i * std_dev, color=['blue', 'green', 'orange', 'red'][i-1], linestyle='-', linewidth=0.5)
        plt.plot(avg_data.index, avg_data - i * std_dev, color=['blue', 'green', 'orange', 'red'][i-1], linestyle='-', linewidth=0.5)

    plt.xlabel('Age (Days)')
    plt.ylabel(variable)
    plt.title(f'{variable} Curves for Selected Populations')
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(pdf_path, format='pdf')
    plt.close()

    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    global df  # Acceder a la variable global df
    if df is None:  # Si el DataFrame no está cargado
        return "Error: File not loaded. Please go back and reload the file."
    
    populations = df['Population'].unique()
    cycles = df['Cycle'].unique()
    groups = df['Group'].unique()
    variables = ['Indv_Weight', 'Mean_Length', 'Length StdDev', 'Length CoeffVar', 'Min Length', 'Max Length', 'Mean_Width', 'Width StdDev', 'Width CoeffVar', 'Min Width', 'Max Width', 'Mean_Area', 'Area StdDev', 'Area CoeffVar', 'Mean_L/W Ratio', 'L/W Ratio StdDev', 'L/W Ratio CoeffVar', 'Min L/W Ratio', 'Max L/W Ratio', 'Mean_Circularity', 'Min Circularity', 'Max Circularity', 'Greyness_I-Mean']
    
    plot_url = None

    if request.method == 'POST':
        selected_populations = request.form.getlist('population')
        selected_cycles = request.form.getlist('cycle')
        selected_groups = request.form.getlist('group')
        variable = request.form['variable']

        plot_url = generate_plot(selected_populations, selected_cycles, selected_groups, variable)

    return render_template(template_path, populations=populations, cycles=cycles, groups=groups, variables=variables, plot_url=plot_url)

# Abrir automáticamente la página web en el navegador predeterminado
webbrowser.open('http://127.0.0.1:5000')

app.run()
