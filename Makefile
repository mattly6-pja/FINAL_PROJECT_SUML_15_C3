install:
	pip install --upgrade pip && pip install -r ./requirements.txt kedro kedro-datasets[pandas]

format:
	black *.py

train:
	python run_kedro.py

eval:
	echo "## Model Metrics" > report.md; \
	echo "" >> report.md; \
	echo "| Metric | Value |" >> report.md; \
	echo "|--------|-------|" >> report.md; \
	tail -n +2 "data/08_reporting/best_models_metrics.csv" | while IFS=, read -r metric value; do echo "| $$metric | $$value |" >> report.md; done; \
	cml comment create report.md

update-branch:
	git config --global user.name "auto_ci_cd_bot"
	git config --global user.email "spam@pjatk.edu.pl"
	git add -A
	git commit -am "Update with new results"
	git push --force origin HEAD:refs/heads/ci_update

hf-login:
	git config --global user.name "auto_ci_cd_bot"
	git config --global user.email "spam@pjatk.edu.pl"
	git pull --no-rebase --allow-unrelated-histories origin ci_update
	git switch ci_update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload SUML15c3/FINAL_CICD_TEST ./streamlit_app.py                          --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload SUML15c3/FINAL_CICD_TEST ./diabetes-predictor/src            /src    --repo-type=space --commit-message="Sync Src"
	huggingface-cli upload SUML15c3/FINAL_CICD_TEST ./diabetes-predictor/data/06_models /models --repo-type=space --commit-message="Sync Models"
	huggingface-cli upload SUML15c3/FINAL_CICD_TEST ./diabetes-predictor/tests          /tests  --repo-type=space --commit-message="Sync Tests"

deploy: hf-login push-hub
