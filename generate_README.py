import subprocess

if __name__ == "__main__":
    command = "jupyter nbconvert --to markdown README.ipynb"
    subprocess.run(command, shell=True, check=True)