
# HLRS-project

  

# Einrichtung der Arbeitsumgebung unter Windows

  

## 1. Installation von Entwicklungswerkzeugen

  

### Visual Studio Code

  

-  [Download Visual Studio Code](https://code.visualstudio.com/docs/setup/windows)

-  Einfach bitte den Schritten auf der Website folgen

  

### PrePoMax (FEM-Softwarepaket)

  

-  [Download PrePoMax](https://prepomax.fs.um.si/)

### bitte beachten, dass das noch nicht funktioniert. Ich regel das gerade mit unserem Betreuer

-   Einfach bitte den Schritten auf der Website folgen

  

### ParaView (Wissenschaftliche Visualisierungen)

  

-  [Download ParaView](https://www.paraview.org/download/)

-   Einfach bitte den Schritten auf der Website folgen

  

### Python und Anaconda

  

-  [Installation von Python und Anaconda](https://www.elab2go.de/demo-py1/installation-python-anaconda.php)

-  Folgt einfach den Anweisungen auf der Website, bei Fragen dürft ihr mich gerne Anrufen und/oder schreiben. 

  

## 2. Erstellen und Aktivieren einer Conda-Umgebung

  

### Conda Umgebung erstellen

```bash

conda  create  --name  py3818  python==3.8.18

```

## Conda Umgebung aktivieren
Name lautet "py3818", also:
 ````bash
 
 conda activate py3818
 
 ````

## Installation der notwendigen und empfohlenen Packete

 ````bash 
 conda install -c conda-forge numpy
conda install -c conda-forge scipy
conda install -c conda-forge scikit-image
conda install -c conda-forge pypng
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
conda install -c conda-forge tifffile
conda install -c conda-forge itk
conda install -c conda-forge vtk
conda install -c conda-forge ccx2paraview
conda install -c conda-forge meshio==5.0.0
conda install -c conda-forge pygalmesh
conda install -c conda-forge dxchange
pip install ciclope[all]
pip install PyMCubes
 ````

### wenn es Probleme bei der Installation gibt, einfach mich Fragen :) 
