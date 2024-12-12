import json
from pathlib import Path

conf_path = '/home/ktolstygin/StokesTest/configs/iranian_generated'
data_path = Path('/mnt/moredata/tolstygin/generated_data/data')



default_config_x_path = Path('default_config_x.json')
default_config_y_path = Path('default_config_y.json')
default_config_z_path = Path('default_config_z.json')

for file in data_path.glob('**/*'):
    name = file.name.split('.')[0]
    gen, iteration = name.split('_')

    with default_config_x_path.open() as x:
        config = json.load(x)
        config["domain"]["grid_size_x_y_z"] = [300, 300, 300]
        config["domain"]["domian_size_x_y_z"] = [300.0, 300.0, 300.0]

        config["files"]["folder"] = '/mnt/moredata/tolstygin/generated_data/data'
        config["files"]["raw"] = file.name
        config["files"]["save_permeability"] = f'/mnt/moredata/tolstygin/generated_data/stokes/{gen}_{iteration}_x.dat'

        with open(f'{conf_path}/{name}_x.json', 'w', encoding='utf-8') as x_e:
            json.dump(config, x_e, ensure_ascii=False, indent=4)

    with default_config_y_path.open() as y:
        config = json.load(y)
        config["domain"]["grid_size_x_y_z"] = [300, 300, 300]
        config["domain"]["domian_size_x_y_z"] = [300.0, 300.0, 300.0]

        config["files"]["folder"] = '/mnt/moredata/tolstygin/generated_data/data'
        config["files"]["raw"] = file.name
        config["files"]["save_permeability"] = f'/mnt/moredata/tolstygin/generated_data/stokes/{gen}_{iteration}_y.dat'

        with open(f'{conf_path}/{name}_y.json', 'w', encoding='utf-8') as y_e:
            json.dump(config, y_e, ensure_ascii=False, indent=4)

    with default_config_z_path.open() as z:
        config = json.load(z)
        config["domain"]["grid_size_x_y_z"] = [300, 300, 300]
        config["domain"]["domian_size_x_y_z"] = [300.0, 300.0, 300.0]

        config["files"]["folder"] = '/mnt/moredata/tolstygin/generated_data/data'
        config["files"]["raw"] = file.name
        config["files"]["save_permeability"] = f'/mnt/moredata/tolstygin/generated_data/stokes/{gen}_{iteration}_z.dat'

        with open(f'{conf_path}/{name}_z.json', 'w', encoding='utf-8') as z_e:
            json.dump(config, z_e, ensure_ascii=False, indent=4)