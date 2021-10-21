import os

dirs = [
    os.path.join('data','raw'),
    os.path.join('data','processed'),
    'data_given',
    'notebooks',
    'saved_models',
    'src',
    'report',
    'tests',
    'prediction_service',
    'webapp',
    os.path.join('prediction_service','model'),
    os.path.join('webapp','static'),
    os.path.join('webapp','templates'),
    os.path.join('webapp\\static','css'),
    os.path.join('webapp\\static','script')
]


for dir_ in dirs:
    os.makedirs(dir_, exist_ok= True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f:
        pass

files = [
    "setup.py",
    ".gitignore",
    os.path.join('src', '__init__.py'),
    os.path.join('src', 'make_data.py'),
    "tox.ini",
    "app.py",
    os.path.join('tests', 'test.py'),
    os.path.join('tests', 'test_config.py'),
    os.path.join('tests', '__init__.py'),
    os.path.join('tests', 'conftest.py'),
    os.path.join('prediction_service','__init__.py'),
    os.path.join('prediction_service','prediction_service.py'),
    os.path.join('webapp\\templates', 'index.html'),
    os.path.join('webapp\\templates', 'base.html'),
    os.path.join('webapp\\templates', '404.html'),
    os.path.join('webapp\\static\\script', 'index.js'),
    os.path.join('webapp\\static\\css', 'main.css')


]

for file_ in files:
    with open(file_, 'w') as f:
        pass

