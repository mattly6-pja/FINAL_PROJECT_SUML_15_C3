install:
	pip install --upgrade pip && pip install -r ./requirements.txt #copied the new requirements here, left the requirements inside PROJECT intact

format:
	black *.py #do we even need this?

train:
	run_kedro.py

eval: #this should more or less work fine
	echo "## Model Metrics" > report.md
	echo "" >> report.md
	echo "| Metric | Value |" >> report.md
	echo "|--------|-------|" >> report.md
	tail -n +2 /PROJECT/diabetes-predictor/data/08_reporting/best_models_metrics.csv | while IFS=, read -r metric value; do
	  echo "| $metric | $value |" >> report.md
	done

	cml comment create report.md

update-branch: #this is OK leave it
	git config --global user.name "auto_ci_cd_bot"
	git config --global user.email "spam@pjatk.edu.pl"
	git commit -am "Update with new results"
	git push --force origin HEAD:ci_update

hf-login: #this is OK leave it
	git config --global user.name "auto_ci_cd_bot"
	git config --global user.email "spam@pjatk.edu.pl"
	git pull --no-rebase --allow-unrelated-histories origin ci_update
	git switch ci_update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: #this is OK leave it
	huggingface-cli upload SUML15c3/FINAL_CICD_TEST ./ --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload SUML15c3/FINAL_CICD_TEST ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload SUML15c3/FINAL_CICD_TEST ./Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub #this is OK leave it