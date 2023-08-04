# WRITE ALL MEAN AVERAGES OF FIELD INTO FILE TO FILTER UNCONVERGED FILES
# COPY CODE INTO preprocess.py if needed.
#
with open('data.txt', 'w') as f:
    for sample in tqdm(dataset, desc='Sample', position=0):
        sim_config = sample[0].rsplit('.', 1)[0]
        sim_config = sim_config.split('_')

        airfoil, angle, mach = sim_config
        angle, mach = float(angle), float(mach)

        vtu_dir = os.path.join(vtu_folder, sample[0])
        stl_dir = os.path.join(stl_folder, sample[1])

        try:
            x, y = vtk_to_tfTensor(config, vtu_dir, stl_dir, mach)

            print(sample[0], ',', y[:, :, 0].mean(), ',',
                  y[:, :, 1].mean(), ',', y[:, :, 2].mean(), file=f)
        except ValueError:
            print('ValueError: ', vtu_dir)
            continue
        except AttributeError:
            print('Attribute error: ', vtu_dir)
            continue

    ######## MOVE UNCONVERGED FILES INTO SEPARATE DIRECTORY
    #########################

    import pandas as pd
    import numpy as np
    import shutil

    data = pd.read_csv('data.txt', delimiter=',')
    data = data.set_axis(['name', 'p', 'ux', 'uy'], axis=1, copy=False)

    data['name'] = data['name'].astype(str)
    data['p'] = data['p'].astype(float)
    data['ux'] = data['ux'].astype(float)
    data['uy'] = data['uy'].astype(float)

    idx = np.where((data['ux'] < 50) | (data['ux'] > 262))
    data = data.loc[idx]

    for name in data.name:
        src = os.path.join(config.preprocess.readdir, 'vtu', name)
        out = os.path.join(config.preprocess.readdir, 'vtu_unconverged')

        shutil.move(src, out)
