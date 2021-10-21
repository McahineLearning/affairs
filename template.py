import os

dirs = [
    os.path.join('data','raw'),
    os.path.join('data','processed'),
    'data_given',
    'notebooks',
    'saved_models',
    'src'
]


for dir_ in dirs:
    os.makedirs(dir_, exist_ok= True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f:
        pass

files = [

    ".gitignore",
    os.path.join('src', '__init__.py'),
    os.path.join('src', 'make_data.py')


]

for file_ in files:
    with open(file_, 'w') as f:
        pass
