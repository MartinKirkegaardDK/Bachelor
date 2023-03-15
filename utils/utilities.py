import os

def naming(path: str, name: str, extension: str) -> str:
    return path + name + "_" + str(len(os.listdir(path))) +"." + extension