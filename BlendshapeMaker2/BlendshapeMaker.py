import imp
import tkinter as tk
from os import path

APP_ROOT = path.dirname(path.abspath(__file__))


def main():
    ''' Create Menu to load data '''
    M = imp.load_source('Menu', APP_ROOT + '/lib/Menu.py')
    root = tk.Tk()
    root.title("BlendshapeMaker")
    menu_app = M.Menu(root)
    menu_app.mainloop()

    '''　If a data path has been selected:
        Create the main application and start the loop'''
    if (len(menu_app.filename) > 0):
        A = imp.load_source('BMApp', APP_ROOT+'/lib/BMApp.py')
        root = tk.Tk()
        app = A.BMApp(menu_app.filename, root)
        app.mainloop()


if __name__ == '__main__':
    main()
