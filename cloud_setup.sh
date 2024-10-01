sudo apt-get update
sudo apt-get install python3-venv python3-pip git
python3 -m venv venv
source venv/bin/activate
git clone https://github.com/taterowney/TateGPT.git
mkdir ./TateGPT/cleaned_data ./TateGPT/models ~/TateGPT/raw_data
pip install -r requirements.txt