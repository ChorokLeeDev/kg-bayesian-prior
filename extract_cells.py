import json

with open('notebooks/colab_neurips_experiments.ipynb', 'r') as f:
    nb = json.load(f)

cells = nb['cells']
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'download_dataset' in src or 'load_triples_temporal' in src:
            print(f'\n{"="*60}')
            print(f'CELL {i}')
            print(f'{"="*60}')
            print(src)
