"""EMG Studio: multi-window workbench for the Sichiray EMG PRO 8-channel armband.

  python emg_studio.py                                  # pick the port in the launcher
  python emg_studio.py --port COM4 --camera 0 --open all
  python emg_studio.py --demo hex --open all            # no device needed

See README.md.
"""
from studio.ui.app import main

if __name__ == "__main__":
    main()
