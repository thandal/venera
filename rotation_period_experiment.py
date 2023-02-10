import os
import numpy as np

filenames = (
    "/home/than/.local/lib/python3.10/site-packages/poliastro/core/fixed.TEMPLATE",
    "/home/than/.local/lib/python3.10/site-packages/poliastro/constants/rotational_elements.TEMPLATE",
)

#R = np.concatenate((np.linspace(-243.01, -243.03, 101),
#                    np.linspace(-243.019, -243.022, 101)))
#R = np.linspace(-243.020, -243.023, 11)
R = -243.02 + 0.0003 * np.arange(1, 6)
#R = (243.01848398589195, )   # Official IAU value
#print(R)

for r in R:
    # Kludge to manipulate the poliastro library
    for filename in filenames:
        content = open(filename, "r").read()
        new_content = content.replace("VENERA_ROT_PER", str(r))
        new_filename = filename.replace("TEMPLATE", "py")
        open(new_filename, "w").write(new_content)
   
    os.system("python3 process_radar_images.py")