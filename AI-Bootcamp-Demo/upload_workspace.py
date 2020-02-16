from azureml.core.model import Model
from azureml.core.webservice import Webservice, AciWebservice
from azureml.core.image import ContainerImage, Image
from azureml.core import Workspace, Run
from azureml.core.conda_dependencies import CondaDependencies
from enums.dependencies import DEPENDENCIES
import os

def main():

    ws = Workspace.from_config()

    conda = CondaDependencies()
    conda.add_conda_package("python==3.5")
    conda.add_pip_package("h5py==2.8.0")
    conda.add_pip_package("html5lib==1.0.1")
    conda.add_pip_package("keras==2.2.0")
    conda.add_pip_package("Keras-Applications==1.0.2")
    conda.add_pip_package("Keras-Preprocessing==1.0.1")
    conda.add_pip_package("matplotlib==2.2.2")
    conda.add_pip_package("numpy==1.14.5")
    conda.add_pip_package("opencv-python==3.3.0.9")
    conda.add_pip_package("pandas==0.23.3")
    conda.add_pip_package("Pillow==5.2.0")
    conda.add_pip_package("requests==2.19.1")
    conda.add_pip_package("scikit-image==0.14.0")
    conda.add_pip_package("scikit-learn==0.19.2")
    conda.add_pip_package("scipy==1.1.0")
    conda.add_pip_package("sklearn==0.0")
    conda.add_pip_package("tensorflow==1.9.0")
    conda.add_pip_package("urllib3==1.23")
    conda.add_pip_package("azureml-sdk")

    with open("environment.yml", "w") as f:
        f.write(conda.serialize_to_string())

    with open("environment.yml", "r") as f:
        print(f.read())

    image_config = ContainerImage.image_configuration(execution_script="score.py",
                                                      runtime="python",
                                                      conda_file="environment.yml",
                                                      docker_file="Dockerfile",
                                                      dependencies=DEPENDENCIES)

    webservices = ws.webservices(compute_type='ACI')

    image = ContainerImage.create(name="ai-bootcamp",
                                  models=[],
                                  image_config=image_config,
                                  workspace=ws)

    image.wait_for_creation(show_output=True)

    webservices_list = []
    for key in webservices:
        webservices_list.append(key)

    service_name = webservices_list[0]

    aciwebservice = AciWebservice(ws, service_name)

    aciwebservice.update(image=image)

if __name__ == '__main__':

    main()