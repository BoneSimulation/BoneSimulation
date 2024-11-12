
# HLRS-project

  

# Einrichtung der Arbeitsumgebung unter Windows

  

## 1. Installation von Entwicklungswerkzeugen

  

### Visual Studio Code

  

-  [Download Visual Studio Code](https://code.visualstudio.com/docs/setup/windows)

-  Einfach bitte den Schritten auf der Website folgen

  

### PrePoMax (FEM-Softwarepaket)
### ".zip" Datei für Installation kann in diesem Ordner gefunden werden
  

-  [Download PrePoMax](https://prepomax.fs.um.si/)

-   Einfach bitte den Schritten auf der Website folgen


### Alternative für Linux: CalculiX

- [Download CalculiX](https://https://www.dhondt.de/)

- Einfach bitte den Schritten auf der Website folgen


### ParaView (Wissenschaftliche Visualisierungen)

  

-  [Download ParaView](https://www.paraview.org/download/)

-   Einfach bitte den Schritten auf der Website folgen

  

### Python und Anaconda

  

-  [Installation von Python und Anaconda](https://www.elab2go.de/demo-py1/installation-python-anaconda.php)

-  Folgt einfach den Anweisungen auf der Website, bei Fragen dürft ihr mich gerne Anrufen und/oder schreiben. 

  

# Anleitung zur Einrichtung der Python-Umgebung

Diese Anleitung führt durch die Erstellung und Aktivierung einer Conda-Umgebung sowie die Installation der benötigten Pakete. 

Ich hoffe ich habe sie für Windows Nutzer optimiert :)



## 2. Erstellen und Aktivieren einer Conda-Umgebung

### Conda-Umgebung erstellen

Öffne das Anaconda Prompt (auf Windows empfohlen) und führe folgenden Befehl aus, um eine neue Umgebung namens `py3818` mit Python 3.12.7 zu erstellen:

```bash
conda create --name py3818 python=3.12.7
```
### Conda-Umgebung aktivieren

Um die Umgebung zu aktivieren:

```bash
conda activate py3818
```
> **Hinweis**: Nach Aktivierung der Umgebung sollte (py3818) links im Prompt angezeigt werden, was signalisiert, dass die Umgebung aktiv ist.


## 3. Anleitung zur Nutzung der environment.yml- und requirements.txt-Dateien




## 4. Manuelle Installation der notwendigen und empfohlenen Pakete
Sobald die Umgebung aktiv ist, installiere die benötigten Pakete. Die Installation ist zweigeteilt: Einige Pakete sind über Conda verfügbar, andere müssen mit pip installiert werden.

Führe die folgenden Befehle aus:

```bash
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
conda install -c conda-forge meshio=5.0.0
conda install -c conda-forge pygalmesh
conda install -c conda-forge dxchange
```


### Zusätzliche Pakete mit pip installieren

Manche Pakete sind möglicherweise nicht in Conda verfügbar. Installiere sie mit pip:

```bash
pip install ciclope[all]
pip install PyMCubes
```
> **Hinweis**: Wenn pip-Pakete wie ciclope oder PyMCubes nicht kompatibel sind, versuche, pip-Pakete immer nach der Installation der Conda-Pakete zu installieren, um Abhängigkeitskonflikte zu vermeiden.

## 4. Überprüfung der Installation
Nach der Installation kannst du die Installation durch ein einfaches Skript überprüfen, das alle installierten Pakete importiert. Erstelle eine Python-Datei (test_installation.py) mit folgendem Inhalt:

```python
import numpy
import scipy
import skimage
import png
import tqdm
import matplotlib
import tifffile
import itk
import vtk
import ccx2paraview
import meshio
import pygalmesh
import dxchange
import ciclope
import PyMCubes

print("Alle Pakete wurden erfolgreich installiert.")
```
Führe das Skript in der aktivierten Conda-Umgebung aus:

```bash
python test_installation.py
```
Wenn keine Fehler auftreten und die Nachricht "Alle Pakete wurden erfolgreich installiert." erscheint, ist die Installation erfolgreich abgeschlossen.


## ich habe aber bereits eine ".bat"-Datei für Windows erstellt, sodass ihr das Anaconda Zeugs nicht manuell installieren müsst! ;)
### wenn es Probleme bei der Installation gibt, einfach mich Fragen :) 

