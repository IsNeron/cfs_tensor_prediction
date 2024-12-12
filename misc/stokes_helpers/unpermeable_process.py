from pathlib import Path



data_path = Path('D:\Work\KT\iranian_generated_data\permeability')


for file in data_path.glob('**/*'):
    file_