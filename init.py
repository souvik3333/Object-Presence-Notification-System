import subprocess
import sys
import clone_repo


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install("tensorflow")
install("opencv-python")
install("pillow")
install("lxml")
install("jupyter")
install("matplotlib")
install("GitPython")
execfile(clone_repo)
