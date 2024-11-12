@echo off

REM Erstellt Conda-Umgebung
conda create --name py3818 python==3.12.7 -y

REM Aktiviert Conda-Umgebung
call conda activate py3818

REM Installiert benötigten Pakete
conda install -c conda-forge numpy -y
conda install -c conda-forge scipy -y
conda install -c conda-forge scikit-image -y
conda install -c conda-forge pypng -y
conda install -c conda-forge tqdm -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge tifffile -y
conda install -c conda-forge itk -y
conda install -c conda-forge vtk -y
conda install -c conda-forge ccx2paraview -y
conda install -c conda-forge meshio==5.0.0 -y
pip install PyMCubes
conda install -c conda-forge pygalmesh -y
conda install -c conda-forge dxchange -y
pip install ciclope[all]
