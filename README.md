# Tenha o python instalado
python --version
pip --version

# 1. Criar ambiente virtual (pasta "venv" será criada)
python -m venv venv

# 2. Ativar o ambiente virtual:
venv\Scripts\activate

# 3. Instalar as dependências dentro do ambiente:
pip install -r requirements.txt

# 4. Rodar o script
python chatbot_pdf.py

# 5. Sair do ambiente
deactivate