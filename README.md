How to install, record, train, and run:

pip install -r requirements.txt
python utils.py --device laptop --count 15 --seconds 5
python utils.py --device iphone --count 15 --seconds 5
python train.py
streamlit run app.py


Notes: keep classes balanced; record in quiet and noisy rooms; accuracy improves with more devices and clips.
