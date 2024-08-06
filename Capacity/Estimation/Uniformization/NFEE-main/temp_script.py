import os
import misc_CPDSSS.entropy_util as ent

N=4
T=1
model_dir = 'temp_data/saved_models'

for folder in os.listdir(model_dir):
    base_folder = os.path.join(model_dir,folder)
    T=int(folder[:-1])

    name = f'CPDSSS_hxc_{folder}'
    model = ent.load_model(None,name,base_folder)
    ent.save_model(model,name,base_folder)

    name = f'CPDSSS_joint_{folder}'
    model = ent.load_model(None,name,base_folder)
    ent.save_model(model,name,base_folder)

    name = f'CPDSSS_Xcond_{folder}'
    model = ent.load_model(None,name,base_folder)
    ent.save_model(model,name,base_folder)

    name = f'CPDSSS_xxc_{folder}'
    model = ent.load_model(None,name,base_folder)
    ent.save_model(model,name,base_folder)