import os

dirs = [
    os.path.join('data','raw'),
    os.path.join('data','processed'),
    'data_given',
    'notebooks',
    'saved_models',
    'src',
    'report',
    'tests'
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
    os.path.join('tests', 'test.py'),
    os.path.join('tests', 'test_config.py'),
    os.path.join('tests', '__init__.py'),
    os.path.join('tests', 'conftest.py')


]

for file_ in files:
    with open(file_, 'w') as f:
        pass

