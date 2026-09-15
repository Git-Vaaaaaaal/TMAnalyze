# Installation dépendances

python -m venv venv venv

source venv/bin/activate

# Installation des modules Python nécessaires
pip install -r requirements.txt

# Installation librairie python trident
git clone https://github.com/mahmoodlab/trident.git
cd trident
pip install -e ".[full]"
