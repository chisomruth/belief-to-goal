# BGT

    pip install -r requirements.txt
    modal setup                                              # once: log in to Modal
    modal secret create bgt-keys ANTHROPIC_API_KEY=sk-ant-... HF_TOKEN=hf_...

    modal run modal_app.py::check                            # does the model load and answer?
    modal run --detach modal_app.py::main --config configs/main.yaml --out main.jsonl
    modal volume get bgt-results main.jsonl ./results/       # download the results
    python -m bgt.metrics results/main.jsonl                 # tables
