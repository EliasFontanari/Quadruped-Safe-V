import mujoco
from mujoco import viewer
model = mujoco.MjModel.from_xml_path('aliengo/random_scene.xml')
data = mujoco.MjData(model)



# simulate and render
with mujoco.viewer.launch(model, data) as viewer:
    while(True):
        if viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()   
            print(data.contact)
        else:
            break