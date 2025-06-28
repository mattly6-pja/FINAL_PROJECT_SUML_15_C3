from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from pathlib import Path

PROJECT_NAME = "diabetes_predictor"

if __name__ == "__main__":
    configure_project(PROJECT_NAME)
    project_path = Path(__file__).resolve().parent

    with KedroSession.create(
        package_name=PROJECT_NAME,
        project_path=project_path,
    ) as session:
        context = session.load_context()
        context.run()
