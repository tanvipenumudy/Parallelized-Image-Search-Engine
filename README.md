# Parallelized-Image-Search-Engine
Final Version of Spring Semester (2020-2021) Project in the Course - High Performance Computing (Team Tech Phoenix)

Image retrieval in itself is a strenuous task. In recent years, there has been a rapid exponential leap in the growth of multimedia databases - chiefly those that are maintained and operated by web-search engine giants such as Google, Yahoo, Bing, etc. Furthermore, millions of images are published to various social networking platforms on a daily basis and it is extremely taxing to even get started on searching for a relevant image from these massive global archives.

To that end, the work presented herein is carried out with the goal of automating the process by enhancing the underlying methodologies, making them as efficient and lightweight as possible subject to the fundamental concepts of High-Performance Computing.

Dataset: [ImageNet LSVRC 2012 Validation Set (Object Detection)](https://academictorrents.com/collection/imagenet-lsvrc-2015)

## Deployment (on Windows)

Download Project Code
```bash
git clone https://github.com/tanvipenumudy/Parallelized-Image-Search-Engine.git
```
Deploy Virtual Environment (optional)
```
python -m venv project_env
project_env\Scripts\activate.bat
```
WSL Ubuntu
```
wsl
```
Install Requirements (WSL Ubuntu)
```bash 
pip3 install -r requirements.txt
```
Run Flask Application
```
python3 directory\offline.py
python3 directory\server.py
```

