
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

PROJECT_PATH = Path(__file__).parent.resolve()
ENVIRONMENT = None  # lub "base"/"local" zgodnie z Twoim projektem

if __name__ == "__main__":
    bootstrap_project(PROJECT_PATH)
    with KedroSession.create(
        project_path=PROJECT_PATH,
        env=ENVIRONMENT,
        save_on_close=True,
    ) as session:
        session.run()  # odpala domyślny pipeline __default__
