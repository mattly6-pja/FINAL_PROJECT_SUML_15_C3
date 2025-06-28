# run_kedro.py

from kedro.framework.context import KedroContext
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import configure_project

project_path = "./diabetes-predictor"

bootstrap_project(project_path)

configure_project("diabetes-predictor")

context = KedroContext.__new__(KedroContext)
context.__init__(project_path=project_path)

context.run()
