ENV_NAME="py3818"

conda env create -f environment.yml
if [ $? -ne 0 ]; then
    echo "Fehler beim Erstellen der Conda-Umgebung."
    exit 1
fi

source activate $ENV_NAME
if [ $? -ne 0 ]; then
    echo "Fehler beim Aktivieren der Conda-Umgebung."
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Fehler beim Installieren der Pakete aus requirements.txt."
    exit 1
fi

echo "Alle Pakete wurden erfolgreich installiert."
