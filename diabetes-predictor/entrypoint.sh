#!/bin/sh

if [ "$TRAIN" = "1" ]; then
    echo "Executing `kedro run` command"
    kedro run
else
    echo "Skipping training."
fi

echo "Starting Streamlit on port 80..."
streamlit run streamlit_app.py --server.port=80 --server.address=0.0.0.0
