import numpy as np
import subprocess
import time

filenames = (
    "/home/than/.local/lib/python3.10/site-packages/poliastro/core/fixed.TEMPLATE",
    "/home/than/.local/lib/python3.10/site-packages/poliastro/constants/rotational_elements.TEMPLATE",
)

#R = np.concatenate((np.linspace(-243.01, -243.03, 101),
#                    np.linspace(-243.019, -243.022, 101)))
#R = np.linspace(-243.0185, -243.023, 46)
#R = np.linspace(-243.021, -243.023, 21)
#R = (-243.0185, )   # Official IAU value

#R = np.unique(np.concatenate((
#    -243.018 + np.linspace(-0.002, 0.002, 81),
#    )))
#RA = (272.76, )  # Official IAU value
#DEC = (67.16, )  # Official IAU value

R = (-243.0208, )   # My best value for 1988 to 2017
RA = np.unique(np.concatenate((
    272.76 + np.linspace(-0.02, 0.02, 11),
    )))
DEC = np.unique(np.concatenate((
    67.16 + np.linspace(-0.02, 0.02, 11),
    )))

print(f"{R=}")
print(f"{RA=}")
print(f"{DEC=}")
print(len(R) * len(RA) * len(DEC))

NUM_SUBPROCESSES = 6
SUBPROCESSES = []

if 1:
    for r in R:
        for ra in RA:
            for dec in DEC:
                print(r, ra, dec)
                if len(SUBPROCESSES) >= NUM_SUBPROCESSES:
                    # Wait until the earliest subprocess is done
                    SUBPROCESSES.pop(0).communicate()
                # Kludge to manipulate the poliastro library
                for filename in filenames:
                    content = open(filename, "r").read()
                    content = content.replace("VENERA_RA", str(ra))
                    content = content.replace("VENERA_DEC", str(dec))
                    content = content.replace("VENERA_ROT_PER", str(r))
                    new_filename = filename.replace("TEMPLATE", "py")
                    open(new_filename, "w").write(content)
                # This code automatically skips images that have already been processed.
                s = subprocess.Popen("python3 process_radar_images.py", shell=True)
                SUBPROCESSES.append(s)
                # Sleep a little to allow the subprocess to successfully import poliastro
                time.sleep(2)  