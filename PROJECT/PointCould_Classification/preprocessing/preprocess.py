import os
import numpy as np
from src.pp_pointMLP import pp_pointMLP


def main():
    do_pp_pointMLP = True

    if do_pp_pointMLP:
        pp_pointMLP("./data", './data_models/pointMLP', .8)


if __name__ == "__main__":
    main()
