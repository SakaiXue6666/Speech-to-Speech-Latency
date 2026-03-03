import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "SEGALE"))

from SEGALE.segale_align import main




if __name__ == "__main__":
    main()