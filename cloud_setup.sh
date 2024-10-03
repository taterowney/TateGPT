sudo apt-get update
sudo apt-get install python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
git clone https://github.com/taterowney/TateGPT.git
mkdir ./TateGPT/cleaned_data ./TateGPT/models ~/TateGPT/raw_data
pip3 install -r ./TateGPT/requirements.txt