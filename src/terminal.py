import sys

import IPython

def _main():

    print("Starting interative terminal")

    noise = globals().keys() | locals().keys()

    from lib import image
    from lib import interface
    from lib import minimize
    from lib.interface import imshow
    import main
    import numpy as np

    libs = (globals().keys() | locals().keys()) - noise
    print(libs)

    # Interative terminal
    terminal = IPython.terminal.embed.InteractiveShellEmbed()
    terminal.extension_manager.load_extension("autoreload")
    terminal.run_line_magic("autoreload", "2")
    terminal.mainloop()


if __name__ == "__main__":
    sys.exit(_main())
